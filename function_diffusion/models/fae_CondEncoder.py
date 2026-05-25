from einops import rearrange, repeat

import jax
import jax.numpy as jnp
from jax.nn.initializers import uniform, normal, xavier_uniform

import flax.linen as nn
from typing import Optional, Callable, Dict, Union, Tuple


# Positional embedding from masked autoencoder https://arxiv.org/abs/2111.06377
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = jnp.arange(embed_dim // 2, dtype=jnp.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = jnp.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = jnp.sin(out)  # (M, D/2)
    emb_cos = jnp.cos(out)  # (M, D/2)

    emb = jnp.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, length):
    pos_embed = get_1d_sincos_pos_embed_from_grid(
            embed_dim, jnp.arange(length, dtype=jnp.float32)
        )
    return jnp.expand_dims(pos_embed, 0)


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
        assert embed_dim % 2 == 0
        # use half of dimensions to encode grid_h
        emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
        emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
        emb = jnp.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
        return emb

    grid_h = jnp.arange(grid_size[0], dtype=jnp.float32)
    grid_w = jnp.arange(grid_size[1], dtype=jnp.float32)
    grid = jnp.meshgrid(grid_w, grid_h, indexing="ij")  # here w goes first
    grid = jnp.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

    return jnp.expand_dims(pos_embed, 0)


class PatchEmbed(nn.Module):
    patch_size: Tuple[int, int] = (16, 16)
    emb_dim: int = 768
    use_norm: bool = False
    kernel_init: Callable = nn.initializers.xavier_uniform()

    @nn.compact
    def __call__(self, inputs):
        b, h, w, c = inputs.shape
        x = nn.Conv(
            self.emb_dim,
            (self.patch_size[0], self.patch_size[1]),
            (self.patch_size[0], self.patch_size[1]),
            kernel_init=self.kernel_init,
            name="proj",
        )(inputs)
        x = jnp.reshape(x, (b, -1, self.emb_dim))
        if self.use_norm:
            x = nn.LayerNorm(name="norm", epsilon=1e-5)(x)
        return x


class MlpBlock(nn.Module):
    dim: int
    out_dim: int
    kernel_init: Callable = xavier_uniform()

    @nn.compact
    def __call__(self, inputs):
        x = nn.Dense(self.dim, kernel_init=self.kernel_init)(inputs)
        x = nn.gelu(x)
        x = nn.Dense(self.out_dim, kernel_init=self.kernel_init)(x)
        return x


class SelfAttnBlock(nn.Module):
    num_heads: int
    emb_dim: int
    mlp_ratio: int
    layer_norm_eps: float = 1e-5

    @nn.compact
    def __call__(self, inputs):
        x = nn.LayerNorm(epsilon=self.layer_norm_eps)(inputs)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads, qkv_features=self.emb_dim
        )(x, x)
        x = x + inputs

        y = nn.LayerNorm(epsilon=self.layer_norm_eps)(x)
        y = MlpBlock(self.emb_dim * self.mlp_ratio, self.emb_dim)(y)

        return x + y


class CrossAttnBlock(nn.Module):
    num_heads: int
    emb_dim: int
    mlp_ratio: int
    layer_norm_eps: float = 1e-5

    @nn.compact
    def __call__(self, q_inputs, kv_inputs):
        q = nn.LayerNorm(epsilon=self.layer_norm_eps)(q_inputs)
        kv = nn.LayerNorm(epsilon=self.layer_norm_eps)(kv_inputs)

        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads, qkv_features=self.emb_dim
        )(q, kv)
        x = x + q_inputs
        y = nn.LayerNorm(epsilon=self.layer_norm_eps)(x)
        y = MlpBlock(self.emb_dim * self.mlp_ratio, self.emb_dim)(y)

        return x + y


class PerceiverBlock(nn.Module):
    emb_dim: int
    depth: int
    num_heads: int = 8
    num_latents: int = 64
    mlp_ratio: int = 1
    layer_norm_eps: float = 1e-5

    @nn.compact
    def __call__(self, x):  # (B, L,  D) --> (B, L', D)
        latents = self.param('latents',
                             normal(),
                             (self.num_latents, self.emb_dim)  # (L', D)
                             )

        latents = repeat(latents, 'l d -> b l d', b=x.shape[0])  # (B, L', D)
        # Transformer
        for _ in range(self.depth):
            latents = CrossAttnBlock(self.num_heads,
                                     self.emb_dim,
                                     self.mlp_ratio,
                                     self.layer_norm_eps)(latents, x)

        latents = nn.LayerNorm(epsilon=self.layer_norm_eps)(latents)
        return latents


class Encoder(nn.Module):
    patch_size: Tuple[int, int]
    grid_size: Tuple[int, int]
    emb_dim: int
    num_latents: int
    depth: int
    num_heads: int
    mlp_ratio: int
    layer_norm_eps: float = 1e-5
    pos_emb_init: Callable = get_2d_sincos_pos_embed

    # ---------- 新增：条件编码器相关参数 ----------
    use_condition_encoder: bool = False  # 是否启用 RF + 坐标编码模块
    cond_width: int = 64                 # Branch 内部宽度
    cond_num_parameter: int = 64
    #cond_out_channels: int = 64
    drop_prob: float = 0.2
    training: bool = False
    force_cond: bool = False
    modes1: int = 12
    modes2: int = 20
    num_groups: int = 32

    def setup(self):
        if self.use_condition_encoder:
            self.cond_enc = ConditionEncoder(
                width=self.cond_width,
                num_parameter=self.cond_num_parameter,
                modes1=self.modes1,
                modes2=self.modes2,
                num_groups=self.num_groups,
                
            )

    @nn.compact
    def __call__(self, x, rf=None, probe=None, rng=None,
                 drop_prob: Optional[float] = None,
                 training: Optional[bool] = None,
                 force_cond: Optional[bool] = None):
        """
        x: 真实声速图 (B, H, W, C)
        rf: RF 信号 (B, Tx, Rx, T)
        probe: 换能器坐标 (B, num_parameter)
        drop_prob: 训练时使用真实图（无条件）的概率
        training: 是否为训练模式
        force_cond: 若为 True，则不论其它条件，都使用条件编码器输出
        """
        drop_prob = self.drop_prob if drop_prob is None else drop_prob
        training = self.training if training is None else training
        force_cond = self.force_cond if force_cond is None else force_cond

        print(f"shape:{x.shape},{rf.shape}")

        # 确定最终用作编码器输入的图像
        if self.use_condition_encoder and rf is not None and probe is not None:
            use_real = True

            if force_cond:
                use_real = False
            elif training and rng is not None and drop_prob > 0.0:
                rng_drop, _ = jax.random.split(rng)
                use_real = jax.random.bernoulli(rng_drop, p=drop_prob)   # 形状 () 布尔数组
            elif not training and not force_cond:
                use_real = False

            # 用 jax.lax.cond 根据 use_real 动态选择输入
            x = jax.lax.cond(
                use_real,
                lambda: x,                                    # 使用真实图
                lambda: self.cond_enc(rf, probe),            # 使用条件编码器输出
            )
                
            # else: 保持原始真实图像 x

        # ------------- 原始 Encoder 主体 -------------
        b, h, w, c = x.shape

        # Patch embedding
        x = PatchEmbed(self.patch_size, self.emb_dim)(x)

        pos_emb = self.variable(
            "pos_emb",
            "enc_pos_emb",
            self.pos_emb_init,
            self.emb_dim,
            (self.grid_size[0] // self.patch_size[0], self.grid_size[1] // self.patch_size[1]),
        )

        pos_emb_interp = pos_emb.value.reshape(
            1,
            self.grid_size[0] // self.patch_size[0],
            self.grid_size[1] // self.patch_size[1],
            self.emb_dim
        )
        pos_emb_interp = jax.image.resize(
            pos_emb_interp,
            (1, h // self.patch_size[0], w // self.patch_size[1], self.emb_dim),
            method='bilinear'
        )
        pos_emb_interp = rearrange(pos_emb_interp, 'b h w d -> b (h w) d')
        x = x + pos_emb_interp

        # Perceiver 压缩
        x = PerceiverBlock(
            emb_dim=self.emb_dim, depth=2,
            num_heads=self.num_heads, num_latents=self.num_latents
        )(x)
        x = nn.LayerNorm(epsilon=self.layer_norm_eps)(x)

        # 深层自注意力
        for _ in range(self.depth):
            x = SelfAttnBlock(
                self.num_heads, self.emb_dim, self.mlp_ratio, self.layer_norm_eps
            )(x)
        x = nn.LayerNorm(epsilon=self.layer_norm_eps)(x)
        return x


class PeriodEmbs(nn.Module):
    period: Tuple[float]  # Periods for different axes
    axis: Tuple[int]  # Axes where the period embeddings are to be applied

    def setup(self):
        # Initialize period parameters and store them in a flax frozen dict
        self.period_params = {f"period_{idx}": period for idx, period in enumerate(self.period)}

    @nn.compact
    def __call__(self, x):
        """
        Apply the period embeddings to the specified axes.
        """
        y = []
        for i, xi in enumerate(x):
            if i in self.axis:
                idx = self.axis.index(i)
                period = self.period_params[f"period_{idx}"]
                y.extend([jnp.cos(period * xi), jnp.sin(period * xi)])
            else:
                y.append(xi)

        return jnp.hstack(y)


class FourierEmbs(nn.Module):
    embed_scale: float
    embed_dim: int

    @nn.compact
    def __call__(self, x):
        kernel = self.param(
            "kernel", normal(self.embed_scale), (x.shape[-1], self.embed_dim // 2)
        )
        y = jnp.concatenate(
            [jnp.cos(jnp.dot(x, kernel)), jnp.sin(jnp.dot(x, kernel))], axis=-1
        )
        return y


class Mlp(nn.Module):
    num_layers: int
    hidden_dim: int
    out_dim: int
    kernel_init: Callable = xavier_uniform()
    layer_norm_eps: float = 1e-5

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        for _ in range(self.num_layers):
            x = nn.Dense(features=self.hidden_dim, kernel_init=self.kernel_init)(x)
            x = nn.gelu(x)
        x = nn.Dense(features=self.out_dim)(x)
        return x


class Decoder(nn.Module):
    fourier_freq: float = 1.0
    period: Union[None, Dict] = None
    dec_depth: int = 2
    dec_num_heads: int = 8
    dec_emb_dim: int = 256
    mlp_ratio: int = 1
    out_dim: int = 1
    num_mlp_layers: int = 1
    layer_norm_eps: float = 1e-5

    @nn.compact
    def __call__(self, x, coords):
        b, n, c = x.shape

        # # Embed periodic boundary conditions if specified
        if self.period is True:
            # Hardcode the periodicity, assuming the domain is [0, 1]x[0, 1]
            coords = PeriodEmbs(period=(2 * jnp.pi, 2 * jnp.pi), axis=(0, 1))(coords)

        coords = FourierEmbs(embed_scale=self.fourier_freq, embed_dim=self.dec_emb_dim)(coords)
        coords = repeat(coords, 'd -> b n d', n=1, b=b)

        x = nn.Dense(self.dec_emb_dim)(x)
        for _ in range(self.dec_depth):
            coords = CrossAttnBlock(num_heads=self.dec_num_heads,
                               emb_dim=self.dec_emb_dim,
                               mlp_ratio=self.mlp_ratio,
                               layer_norm_eps=self.layer_norm_eps)(coords, x)

        x = nn.LayerNorm(epsilon=self.layer_norm_eps)(coords)
        # x = nn.Dense(self.out_dim)(x)

        x = Mlp(num_layers=self.num_mlp_layers,
                hidden_dim=self.dec_emb_dim,
                out_dim=self.out_dim,
                layer_norm_eps=self.layer_norm_eps)(x)

        return x

class Branch(nn.Module):
    """input: RF data"""
    width: int
    """Tx: int
    Rx: int
    T_steps: int
    H_out: int          # 输出高度，应设为 DiT 的 patch grid H
    W_out: int          # 输出宽度，应设为 DiT 的 patch grid W
    channel_lift_first: bool = False"""
    
        
    @nn.compact
    def __call__(self, x):
        x = jnp.swapaxes(x, 1, 3)  # -1, time_steps,Rx,Tx
        x = nn.Dense(self.width)(x)  # -1, time_steps, R, width
        x = jnp.swapaxes(x, 1, 3)  # -1, width, R, time_steps
        x = nn.gelu(x)
        return x

    """def __call__(self, x):   # x: (B, Tx, Rx, T)
        if self.channel_lift_first:
            # Lift Tx axis first
            x = nn.Dense(self.width * 2)(x)                     # (B, Tx, Rx, 2*width)
            x = jnp.swapaxes(x, 2, 3)                           # (B, Tx, 2*width, Rx)
            x = nn.Dense(self.H_out)(x)                         # (B, Tx, 2*width, H_out)
            x = jnp.swapaxes(x, 1, 2)                           # (B, 2*width, H_out, Tx)
            x = nn.Dense(self.W_out)(x)                         # (B, 2*width, H_out, W_out)
            x = nn.gelu(x)
            x = jnp.swapaxes(x, 1, 3)                           # (B, W_out, H_out, 2*width)
            x = nn.Dense(self.width)(x)                         # (B, W_out, H_out, width)
            x = jnp.swapaxes(x, 1, 2)                           # (B, H_out, W_out, width)
            x = jnp.transpose(x, (0, 3, 1, 2))                  # (B, width, H_out, W_out)
        else:
            # Original PyTorch order: Project T -> W, then Rx -> H, then channel lift
            x = nn.Dense(self.W_out)(x)                         # (B, Tx, Rx, W_out)
            x = jnp.swapaxes(x, 2, 3)                           # (B, Tx, W_out, Rx)
            x = nn.Dense(self.H_out)(x)                         # (B, Tx, W_out, H_out)
            x = jnp.swapaxes(x, 2, 3)                           # (B, Tx, H_out, W_out)
            # Channel lift: Tx -> width via Conv
            x = nn.Conv(self.width, kernel_size=(1, 1), use_bias=True)(x)  # (B, width, H_out, W_out)
            x = nn.gelu(x)
        return x"""


class Trunk(nn.Module):
    """坐标 -> 全局调制 (B, width, 1, 1)"""
    width: int
    num_parameter: int

    @nn.compact
    def __call__(self, x):   # x: (B, num_parameter)
        x = nn.Dense(self.width)(x)
        x = nn.gelu(x)
        return x[:, :, None, None]   # (B, width, 1, 1)


class SpectralConv2d(nn.Module):
    """2D Fourier layer (Flax version)"""
    in_channels: int
    out_channels: int
    modes1: int   # number of Fourier modes in height direction
    modes2: int   # number of Fourier modes in width direction

    @nn.compact
    def __call__(self, x):
        batchsize, _, h, w = x.shape
        # Real FFT: shape (B, in_ch, h, w//2+1)
        x_fft = jnp.fft.rfft2(x, axes=(-2, -1))

        # Initialize complex weights (two sets for low and high modes)
        scale = 1.0 / (self.in_channels * self.out_channels)
        weight1 = self.param(
            'weight1',
            lambda rng, shape: scale * (jax.random.normal(rng, shape) + 1j * jax.random.normal(rng, shape)),
            (self.in_channels, self.out_channels, self.modes1, self.modes2)
        )
        weight2 = self.param(
            'weight2',
            lambda rng, shape: scale * (jax.random.normal(rng, shape) + 1j * jax.random.normal(rng, shape)),
            (self.in_channels, self.out_channels, self.modes1, self.modes2)
        )

        # Slice and multiply
        out_fft = jnp.zeros((batchsize, self.out_channels, h, w//2+1), dtype=jnp.complex64)
        out_fft = out_fft.at[:, :, :self.modes1, :self.modes2].set(
            jnp.einsum('bixy,ioxy->boxy', x_fft[:, :, :self.modes1, :self.modes2], weight1)
        )
        out_fft = out_fft.at[:, :, -self.modes1:, :self.modes2].set(
            jnp.einsum('bixy,ioxy->boxy', x_fft[:, :, -self.modes1:, :self.modes2], weight2)
        )

        # Inverse FFT
        x_out = jnp.fft.irfft2(out_fft, s=(h, w), axes=(-2, -1))
        return x_out


class U_net(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, dropout_rate):
        super(U_net, self).__init__()
        self.input_channels = input_channels
        self.conv1 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=2,
                               dropout_rate=dropout_rate)
        self.conv2 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=2,
                               dropout_rate=dropout_rate)
        self.conv2_1 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=1,
                                 dropout_rate=dropout_rate)
        self.conv3 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=2,
                               dropout_rate=dropout_rate)
        self.conv3_1 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=1,
                                 dropout_rate=dropout_rate)

        self.deconv2 = self.deconv(input_channels, output_channels)
        self.deconv1 = self.deconv(input_channels * 2, output_channels)
        self.deconv0 = self.deconv(input_channels * 2, output_channels)

        self.output_layer = self.output(input_channels * 2, output_channels,
                                        kernel_size=kernel_size, stride=1, dropout_rate=dropout_rate)

    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2_1(self.conv2(out_conv1))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_deconv2 = self.deconv2(out_conv3)
        concat2 = torch.cat((out_conv2, out_deconv2), 1)
        out_deconv1 = self.deconv1(concat2)
        concat1 = torch.cat((out_conv1, out_deconv1), 1)
        out_deconv0 = self.deconv0(concat1)
        concat0 = torch.cat((x, out_deconv0), 1)
        out = self.output_layer(concat0)

        return out

    def conv(self, in_planes, output_channels, kernel_size, stride, dropout_rate):
        return nn.Sequential(
            nn.Conv2d(in_planes, output_channels, kernel_size=kernel_size,
                      stride=stride, padding=(kernel_size - 1) // 2, bias=False),
            nn.GroupNorm(num_groups=32, num_channels=output_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate)
        )

    def deconv(self, input_channels, output_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(input_channels, output_channels, kernel_size=4,
                               stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def output(self, input_channels, output_channels, kernel_size, stride, dropout_rate):
        return nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size,
                         stride=stride, padding=(kernel_size - 1) // 2)

class ChannelFirstConv(nn.Module):
    features: int
    kernel_size: Tuple[int, int]

    @nn.compact
    def __call__(self, x):
        x = x.transpose((0, 2, 3, 1))   # NCHW -> NHWC
        x = nn.Conv(self.features, self.kernel_size)(x)
        x = x.transpose((0, 3, 1, 2))   # NHWC -> NCHW
        return x

class ConditionEncoder(nn.Module):
    """融合RF data 和 条件输入"""
    width: int = 64
    num_parameter: int = 64
    modes1: int = 12
    modes2: int = 20
    num_groups: int = 32
    """Tx: int = 32
    Rx: int = 32
    T_steps: int = 1900
    H_out: int = 16
    W_out: int = 16
    channel_lift_first: bool = True"""

    def _forward_block0(self, x):
        # SpectralConv2d + 1x1 Conv + 残差连接
        conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2, name='conv0')
        w0 = ChannelFirstConv(self.width, kernel_size=(1, 1), name='w0')
        
        x1 = conv0(x)          # (B, width, H, W)
        x2 = w0(x)             # (B, width, H, W)
        x = x1 + x2            # 残差求和

        x = nn.gelu(x)
        return x

    def _forward_block1(self, x):
        # SpectralConv2d + 1x1 Conv + 残差连接
        conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2, name='conv1')
        w1 = ChannelFirstConv(self.width, kernel_size=(1, 1), name='w1')
        
        x1 = conv1(x)          # (B, width, H, W)
        x2 = w1(x)             # (B, width, H, W)
        x = x1 + x2            # 残差求和

        x = nn.gelu(x)
        return x

    @nn.compact
    def __call__(self, rf, probe):
        # rf: (B, Tx, Rx, T), probe: (B, num_parameter)
        x1 = Branch(self.width)(rf)    # (B, width, H_out, W_out)
        x2 = Trunk(self.width, self.num_parameter)(probe)                    # (B, width, 1, 1)

        x = x1 * x2          # 广播相乘，得到 (B, width, H_out, W_out)
        x = x + self.param('bias', nn.initializers.zeros, (1, self.width, 1, 1))
        
        x = self._forward_block0(x)
        
        x = nn.Dense(512)(x)
        
        x = jnp.transpose(x, (0, 2, 3, 1)) # 转为 channel-last: (B, H_out, W_512, width)
        x = nn.GroupNorm(num_groups=32)(x)
        x = jnp.transpose(x, (0, 3, 1, 2)) # channel-first: (B, width, H_out, W_512)

        x = self._forward_block1(x)

        x = nn.Dense(80)(x)
        x = jnp.transpose(x, (0, 1, 3, 2)) #(B, width, W_80, H_out)
        x = nn.Dense(80)(x) #(B, width, W_80, H_80)
        x = nn.gelu(x)
        
        x = jnp.transpose(x, (0, 3, 2, 1)) # channel-last: (B, H_80, W_80, width)
        x = nn.GroupNorm(num_groups=32)(x)
        return x

