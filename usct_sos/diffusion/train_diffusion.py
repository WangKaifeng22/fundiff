import os
import json
import time
import sys

import ml_collections
import wandb

import jax
import jax.numpy as jnp
from jax import random, jit

from jax.experimental import mesh_utils, multihost_utils
from jax.sharding import Mesh, PartitionSpec as P

from function_diffusion.models import DiT, Encoder, Decoder

from function_diffusion.utils.model_utils import (
    create_optimizer,
    create_diffusion_state,
    compute_total_params,
)
from function_diffusion.utils.train_utils import create_train_diffusion_step, get_diffusion_batch
from function_diffusion.utils.checkpoint_utils import (
    create_checkpoint_manager,
    save_checkpoint,
    restore_fae_state
)
from function_diffusion.utils.h5_data_utils import create_dataloader, worker_init_fn

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from data_utils import create_dataset
from model_utils import create_encoder_step


def train_and_evaluate(config: ml_collections.ConfigDict):
    # Initialize function autoencoder
    encoder = Encoder(**config.autoencoder.encoder)
    decoder = Decoder(**config.autoencoder.decoder)
    fae_job_name = f"{config.autoencoder.model_name}_use_pde_{config.training.use_pde}"
    fae_state = restore_fae_state(config, fae_job_name, encoder, decoder)

    # Initialize diffusion model
    dit = DiT(**config.diffusion)
    # Create learning rate schedule and optimizer
    lr, tx = create_optimizer(config)

    # Create diffusion train state
    state = create_diffusion_state(config, dit, tx, use_conditioning=True)
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
    #fae_state = multihost_utils.host_local_array_to_global_array(fae_state, mesh, P())

    # Create train step function
    train_step = create_train_diffusion_step(dit, mesh, use_conditioning=True)
    encoder_step = create_encoder_step(encoder, mesh)

    # Create dataloaders
    train_dataset, _ = create_dataset(config)
    train_loader = create_dataloader(train_dataset,
                                     shuffle=config.dataset.shuffle,
                                     batch_size=config.dataset.batch_size,
                                     num_workers=config.dataset.num_workers,
                                     worker_init_fn=worker_init_fn)
    
    downsample_factors = config.training.downsample_factors

    sample_batch = next(iter(train_loader))
    _, _, _, sample_y = sample_batch
    h, w = sample_y.shape[1:3]

    # Create checkpoint manager
    job_name = f"{config.diffusion.model_name}_use_pde_{config.training.use_pde}"
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

    encode_real_step = create_encoder_step(encoder, mesh, training=True, drop_prob=1.0, force_cond=False)
    encode_cond_step = create_encoder_step(encoder, mesh, training=False, drop_prob=0.0, force_cond=True)

    rng_key = random.PRNGKey(0)
    global_step = int(state.step)
    loss = jnp.array(0.0)
    for epoch in range(10000):
        start_time = time.time()
        for batch in train_loader:
            rng_key, subkey = random.split(rng_key)

            batch = jax.tree.map(jnp.array, batch)

            x_branch, x_probe, x_aux, y = batch
            probe = x_probe.reshape(x_probe.shape[0], -1)
            if x_aux.ndim >= 2 and x_aux.shape[-1] > 0:
                probe = jnp.concatenate([probe, x_aux.reshape(x_aux.shape[0], -1)], axis=-1)
            target_probe_dim = int(config.autoencoder.encoder.cond_num_parameter)
            if probe.shape[-1] < target_probe_dim:
                probe = jnp.pad(probe, ((0, 0), (0, target_probe_dim - probe.shape[-1])))
            elif probe.shape[-1] > target_probe_dim:
                probe = probe[:, :target_probe_dim]

            if config.training.random_resolution:
                key1, key2 = random.split(subkey)
                d = int(jax.device_get(random.choice(key1, downsample_factors)))
                y = jax.image.resize(
                    y[:, ::d, ::d],
                    (y.shape[0], h, w, y.shape[-1]),
                    method="bilinear",
                )
            else:
                key2 = subkey

            z_u = encode_real_step(fae_state.params[0], (y, x_branch, probe), key2) #rng=key2
            z_c = encode_cond_step(fae_state.params[0], (y, x_branch, probe), subkey) #rng=subkey

            diffusion_batch, rng_key = get_diffusion_batch(rng_key, z1=z_u, c=z_c, use_conditioning=True)
            global_step += 1
            state, loss = train_step(state, diffusion_batch)

        # Logging
        if epoch % config.logging.log_interval == 0:
            # Log metrics
            loss = loss.item()
            end_time = time.time()
            log_dict = {"loss": loss, "lr": lr(global_step)}

            if jax.process_index() == 0:
                wandb.log(log_dict, global_step)  # Log metrics to W&B
                print("step: {}, loss: {:.3e}, time: {:.3e}".format(global_step, loss, end_time - start_time))

        # Save checkpoint
        if epoch % config.saving.save_interval == 0:
            save_checkpoint(ckpt_mngr, state)

        if global_step >= config.training.max_steps:
            break

    # Save final checkpoint
    print("Training finished, saving final checkpoint...")
    save_checkpoint(ckpt_mngr, state)
    ckpt_mngr.wait_until_finished()





