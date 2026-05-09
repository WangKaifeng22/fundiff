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
    state = multihost_utils.host_local_array_to_global_array(state, mesh, P())

    # Create loss and train step functions
    encoder_cfg = config.model.encoder
    train_step = create_train_step(
        encoder,
        decoder,
        mesh,
        use_pde=config.training.use_pde,
        drop_prob=encoder_cfg.get("drop_prob", 0.2),
        training=encoder_cfg.get("training", True),
        force_cond=encoder_cfg.get("force_cond", False),
    )

    # Create dataloaders
    train_dataset, test_dataset = create_dataset(config)
    train_loader = create_dataloader(train_dataset,
                                     batch_size=config.dataset.train_batch_size,
                                     num_workers=config.dataset.num_workers,
                                     worker_init_fn=worker_init_fn)

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
    for epoch in range(10000):
        start_time = time.time()
        for batch in train_loader:
            rng_key, subkey = jax.random.split(rng_key)
            batch = jax.tree.map(jnp.array, batch)

            if config.training.random_resolution:
                key1, key2 = jax.random.split(subkey)

                downsample_factors = jnp.array([1, 2, 5])
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

            query_batch = ((x_branch, x_probe, x_aux), y)
            coords, rf_branch, probe_vec, y_query = batch_parser.random_query(query_batch, rng_key=key2)
            target_probe_dim = int(getattr(encoder_cfg, "cond_num_parameter", probe_vec.shape[-1]))

            if probe_vec.shape[-1] != target_probe_dim:
                raise ValueError("Target probe dimension must be equal to the encoder dimension")

            batch_data = (coords, y, y_query)
            batch_data = (batch_data, rf_branch, probe_vec)

            batch_data = multihost_utils.host_local_array_to_global_array(
                batch_data, mesh, P("batch")
            )
            state, loss, loss_data, loss_res = train_step(state, batch_data, rng=subkey)

        # Logging
        if epoch % config.logging.log_interval == 0:
            # Log metrics
            step = int(state.step)
            loss = loss.item()
            loss_data = loss_data.item()
            loss_res = loss_res.item()

            end_time = time.time()
            log_dict = {"loss": loss,
                        "loss_data": loss_data,
                        "loss_res": loss_res,
                        "lr": lr(step)}

            if jax.process_index() == 0:
                wandb.log(log_dict, step)  # Log metrics to W&B
                print("step: {},  loss data: {:.3e}, loss res: {:.3e}, time: {:.3e}".format(step, loss_data, loss_res, end_time - start_time))

        # Save checkpoint
        if epoch % config.saving.save_interval == 0:
            save_checkpoint(ckpt_mngr, state)

        if step >= config.training.max_steps:
            break

    # Save final checkpoint
    print("Training finished, saving final checkpoint...")
    save_checkpoint(ckpt_mngr, state)
    ckpt_mngr.wait_until_finished()




