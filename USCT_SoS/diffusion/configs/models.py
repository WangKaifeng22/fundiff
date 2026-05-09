import ml_collections


MODEL_CONFIGS = {}


def _register(get_config):
    """Adds reference to model config into MODEL_CONFIGS."""
    config = get_config().lock()
    name = config.get("model_name")
    MODEL_CONFIGS[name] = config
    return get_config


@_register
def get_fae_config():
    config = ml_collections.ConfigDict()
    config.model_name = "FAE"

    config.encoder = encoder = ml_collections.ConfigDict()
    encoder.patch_size = (20, 20)   # For tokenization
    encoder.emb_dim = 256
    encoder.num_latents = 100
    encoder.grid_size = (200, 200)   # USCT 声速图分辨率
    encoder.depth = 8
    encoder.num_heads = 8
    encoder.mlp_ratio = 2
    encoder.layer_norm_eps = 1e-5

    config.cond_encoder = cond_encoder = ml_collections.ConfigDict()
    cond_encoder.use_condition_encoder = True
    cond_encoder.cond_width = 64
    cond_encoder.cond_num_parameter = 64
    """cond_encoder.cond_Tx = 32
    cond_encoder.cond_Rx = 32
    cond_encoder.cond_T_steps = 1900
    cond_encoder.cond_H_out = 200
    cond_encoder.cond_W_out = 200
    cond_encoder.cond_out_channels = 1
    cond_encoder.channel_lift_first = False"""
    cond_encoder.drop_prob = 0.2

    encoder.use_condition_encoder = cond_encoder.use_condition_encoder
    encoder.cond_width = cond_encoder.cond_width
    encoder.cond_num_parameter = cond_encoder.cond_num_parameter
    """encoder.cond_Tx = cond_encoder.cond_Tx
    encoder.cond_Rx = cond_encoder.cond_Rx
    encoder.cond_T_steps = cond_encoder.cond_T_steps
    encoder.cond_H_out = cond_encoder.cond_H_out
    encoder.cond_W_out = cond_encoder.cond_W_out
    encoder.cond_out_channels = cond_encoder.cond_out_channels
    encoder.channel_lift_first = cond_encoder.channel_lift_first"""

    config.decoder = decoder = ml_collections.ConfigDict()
    decoder.period = False
    decoder.fourier_freq = 10.0
    decoder.dec_emb_dim = 256
    decoder.dec_depth = 4
    decoder.dec_num_heads = 8
    decoder.mlp_ratio = 2
    decoder.num_mlp_layers = 2
    decoder.out_dim = 1
    decoder.layer_norm_eps = 1e-5

    return config


@_register
def get_dit_config():
    config = ml_collections.ConfigDict()
    config.model_name = "DiT"

    config.emb_dim = 256
    config.depth = 8
    config.num_heads = 8
    config.mlp_ratio = 2
    config.out_dim = 256

    return config


@_register
def get_cvit_config():
    config = ml_collections.ConfigDict()
    config.model_name = "CViT"

    config.encoder = encoder = ml_collections.ConfigDict()

    encoder.patch_size = (16, 16)
    encoder.grid_size = (256, 256)
    encoder.emb_dim = 256
    encoder.depth = 8
    encoder.num_heads = 8
    encoder.mlp_ratio = 2
    encoder.layer_norm_eps = 1e-5

    config.decoder = decoder = ml_collections.ConfigDict()

    decoder.period = None
    decoder.fourier_freq = 1.0
    decoder.dec_emb_dim = 256
    decoder.dec_depth = 4
    decoder.mlp_ratio = 2
    decoder.dec_num_heads = 8
    decoder.num_mlp_layers = 2
    decoder.out_dim = 1

    return config

