import os
import json
import time
import sys

import ml_collections
import wandb

import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils, multihost_utils
from jax.sharding import Mesh, PartitionSpec as P
from flax.traverse_util import flatten_dict

from function_diffusion.models import Encoder, Decoder

from function_diffusion.utils.model_utils import (
    create_optimizer,
    create_autoencoder_state,
    compute_total_params,
)
from function_diffusion.utils.checkpoint_utils import (
    create_checkpoint_manager,
    save_checkpoint,
)
from function_diffusion.utils.h5_data_utils import H5BatchParser, create_dataloader, worker_init_fn

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from usct_sos.data_utils import create_dataset
from model_utils import create_train_step


def _format_grad_norm_logs(grad_norms, prefix):
    flat_grad_norms = flatten_dict(grad_norms, sep="/")
    return {
        f"{prefix}/{name}": float(value)
        for name, value in flat_grad_norms.items()
    }


def train_and_evaluate(config: ml_collections.ConfigDict):
    # Initialize model
    encoder = Encoder(**config.model.encoder)
    decoder = Decoder(**config.model.decoder)
    # Create learning rate schedule and optimizer
    lr, tx = create_optimizer(config)

    # Create train state
    state = create_autoencoder_state(config, encoder, decoder, tx)
    num_params = compute_total_params(state)
    print(f"Model storage cost: {num_params * 4 / 1024 / 1024:.2f} MB of parameters")

    # Device count
    num_local_devices = jax.local_device_count()
    num_devices = jax.device_count()
    print(f"Number of devices: {num_devices}")
    print(f"Number of local devices: {num_local_devices}")

    # Create sharding for data parallelism
    mesh = Mesh(mesh_utils.create_device_mesh((jax.device_count(),)), "batch")
    #state = multihost_utils.host_local_array_to_global_array(state, mesh, P())

    # Create loss and train step functions
    encoder_cfg = config.model.encoder
    grad_norm_cfg = config.logging.grad_norm
    step_timing_cfg = config.logging.step_timing
    if grad_norm_cfg.enabled and grad_norm_cfg.log_interval <= 0:
        raise ValueError("config.logging.grad_norm.log_interval must be positive when grad norm logging is enabled")
    train_step = create_train_step(
        encoder,
        decoder,
        mesh,
        use_pde=config.training.use_pde,
        drop_prob=encoder_cfg.get("drop_prob", 0.2),
        training=encoder_cfg.get("training", True),
        force_cond=encoder_cfg.get("force_cond", False),
        return_grad_norms=grad_norm_cfg.enabled,
    )
    step_timing_enabled = bool(step_timing_cfg.enabled)

    # Create dataloaders
    train_dataset, test_dataset = create_dataset(config)
    train_loader = create_dataloader(train_dataset,
                                     shuffle=config.dataset.shuffle,
                                     batch_size=config.dataset.train_batch_size,
                                     num_workers=config.dataset.num_workers,
                                     worker_init_fn=worker_init_fn)
    
    downsample_factors = config.training.downsample_factors

    # Create batch parser
    sample_batch = next(iter(train_loader))
    _, _, _, sample_y = sample_batch
    h, w = sample_y.shape[1:3]
    aux_dim = int(sample_batch[2].shape[-1]) if sample_batch[2].ndim > 1 and sample_batch[2].shape[-1] > 0 else 0
    batch_parser = H5BatchParser(config, h, w, aux_dim=aux_dim)

    # Create checkpoint manager
    job_name = f"{config.model.model_name}_use_pde_{config.training.use_pde}"
    ckpt_path = os.path.join(os.getcwd(), job_name, "ckpt")
    if jax.process_index() == 0:
        if not os.path.isdir(ckpt_path):
            os.makedirs(ckpt_path)

        # Save config
        config_dict = config.to_dict()
        config_path = os.path.join(os.getcwd(), job_name, "config.json")
        with open(config_path, "w") as json_file:
            json.dump(config_dict, json_file, indent=4)

        # Initialize W&B
        wandb_config = config.wandb
        wandb.init(project=wandb_config.project, name=job_name, config=config)

    # Create checkpoint manager
    ckpt_mngr = create_checkpoint_manager(config.saving, ckpt_path)

    # Training loop
    rng_key = jax.random.PRNGKey(0)
    step = 0
    loss = loss_data = loss_res = jnp.array(0.0)

    global_step = int(state.step)

    for epoch in range(10000):
        start_time = time.time()
        train_iter = iter(train_loader)
        while True:
            if step_timing_enabled:
                step_timing_t0 = time.perf_counter()
            try:
                batch = next(train_iter)
            except StopIteration:
                break
            if step_timing_enabled:
                data_loader_ms = (time.perf_counter() - step_timing_t0) * 1000.0

            rng_key, subkey = jax.random.split(rng_key)
            batch = jax.tree.map(jnp.array, batch)

            if config.training.random_resolution:
                key1, key2 = jax.random.split(subkey)

                random_downsample = int(jax.device_get(jax.random.choice(key1, downsample_factors)))
                x_branch, x_probe, x_aux, y = batch
                y = jax.image.resize(
                    y[:, ::random_downsample, ::random_downsample],
                    (y.shape[0], h, w, y.shape[-1]),
                    method="bilinear",
                )
            else:
                x_branch, x_probe, x_aux, y = batch
                key2 = subkey

            query_batch = ((x_probe, x_aux), y)
            if step_timing_enabled:
                random_query_t0 = time.perf_counter()
            #coords, probe_vec, y_query = batch_parser.random_query(query_batch, rng_key=key2)
            coords, probe_vec, y_query = batch_parser.query_all(query_batch)
            if step_timing_enabled:
                jax.block_until_ready(y_query)
                random_query_ms = (time.perf_counter() - random_query_t0) * 1000.0

            batch_data = (coords, y, y_query) #(coords, input ,y)
            batch_data = (batch_data, x_branch, probe_vec)

            """batch_data = multihost_utils.host_local_array_to_global_array(
                batch_data, mesh, P("batch")
            )"""
            if step_timing_enabled:
                train_step_t0 = time.perf_counter()

            global_step += 1
            if grad_norm_cfg.enabled:
                state, loss, loss_data, loss_res, grads = train_step(state, batch_data, subkey)
                if global_step > 0 and global_step % grad_norm_cfg.log_interval == 0:
                    grad_norms = jax.tree.map(lambda g: jnp.linalg.norm(g), grads)
                    # 假设元组顺序为 (encoder_grad_norms, decoder_grad_norms)
                    grad_norms = {"encoder": grad_norms[0], "decoder": grad_norms[1]}
                    grad_log_dict = _format_grad_norm_logs(grad_norms, grad_norm_cfg.prefix)
                    if jax.process_index() == 0:
                        wandb.log(grad_log_dict, global_step)
            else:
                state, loss, loss_data, loss_res = train_step(state, batch_data, subkey)

            if step_timing_enabled:
                jax.block_until_ready(loss_res)
                train_step_ms = (time.perf_counter() - train_step_t0) * 1000.0
                if jax.process_index() == 0:
                    wandb.log(
                        {
                            "timing/data_loader_ms": data_loader_ms,
                            "timing/random_query_ms": random_query_ms,
                            "timing/train_step_ms": train_step_ms,
                        },
                        global_step,
                    )



        # Logging
        if epoch % config.logging.log_interval == 0:
            # Log metrics
            #step = int(state.step)
            loss = loss.item()
            loss_data = loss_data.item()
            loss_res = loss_res.item()

            end_time = time.time()
            log_dict = {"loss": loss,
                        "loss_data": loss_data,
                        "loss_res": loss_res,
                        "lr": lr(global_step)}

            if jax.process_index() == 0:
                wandb.log(log_dict, global_step)  # Log metrics to W&B
                print(f"step: {global_step}, loss data: {loss_data:.3e}, "
                      f"loss res: {loss_res:.3e}, time: {end_time - start_time:.3e}")
        # Save checkpoint
        if epoch % config.saving.save_interval == 0:
            save_checkpoint(ckpt_mngr, state)

        if global_step >= config.training.max_steps:
            break

    # Save final checkpoint
    print("Training finished, saving final checkpoint...")
    save_checkpoint(ckpt_mngr, state)
    ckpt_mngr.wait_until_finished()




