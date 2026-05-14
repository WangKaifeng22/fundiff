import ml_collections

from configs import models
import math

def get_config(model):
    """Get the hyperparameter configuration for a specific model."""
    config = get_base_config()
    get_model_config = getattr(models, f"get_{model}_config")
    config.model = get_model_config()
    return config


def get_base_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    # Random seed
    config.seed = 42

    # Input shape for initializing Flax models
    config.x_dim = [2, 200, 200, 1]
    config.coords_dim = [2,]  # Only for initializing CViT model
    config.cond_Tx = 32
    config.cond_Rx = 32
    config.cond_T_steps = 1900

    # Training or evaluation
    config.mode = "train_autoencoder"

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "fundiff_usct_sos"
    wandb.tag = None

    # Dataset
    config.dataset = dataset = ml_collections.ConfigDict()
    dataset.data_path = "/home/wkf/kwave-python/dataset/dataset_shuffle_0.140625-0.453125.h5"
    dataset.downsample_factor = 1
    dataset.num_train_samples = 45000
    dataset.train_batch_size = 32  # Per device
    dataset.test_batch_size = 4  # Per device
    dataset.num_workers = 2 #prefetch_factor=2

    # Learning rate
    config.lr = lr = ml_collections.ConfigDict()
    lr.init_value = 0.0
    lr.peak_value = 1e-3
    lr.decay_rate = 0.9
    lr.transition_steps = 2000
    lr.warmup_steps = 4000

    # Optim
    config.optim = optim = ml_collections.ConfigDict()
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.eps = 1e-8
    optim.weight_decay = 1e-5
    optim.clip_norm = 1.0

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.max_steps = math.ceil(45000 / 32) * 150
    training.num_queries = 2048 #H5BatchParser __init__
    training.random_resolution = False
    training.downsample_factors = [1,2,5]
    training.use_pde = False

    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_interval = 1 #epoch
    logging.grad_norm = grad_norm = ml_collections.ConfigDict()
    grad_norm.enabled = True
    grad_norm.log_interval = 500 #step
    grad_norm.prefix = "grad_norms"

    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.save_interval = 2 #epoch
    saving.num_keep_ckpts = 1

    return config
