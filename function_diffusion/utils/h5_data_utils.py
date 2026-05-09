import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np

class H5Dataset(Dataset):
    def __init__(self, h5_path, indices, transform=None):
        self.h5_path = h5_path
        self.indices = np.asarray(indices)
        self.transform = transform
        self._h5 = None  # 主进程不使用

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # 为避免每次打开文件，可在 worker 初始化时打开，这里作为 fallback
        h5 = getattr(self, '_local_h5', None)
        if h5 is None:
            h5 = h5py.File(self.h5_path, 'r', swmr=True, libver='latest')
        i = self.indices[idx]
        Xb = h5['X_branch'][i].astype(np.float32)
        Xp = h5['X_trunk'][i].astype(np.float32)
        y  = h5['y'][i].astype(np.float32)
        X_aux = h5['X_aux'][i].astype(np.float32) if 'X_aux' in h5 else np.zeros(0)
        return Xb, Xp, X_aux, y

def worker_init_fn(worker_id):
    # 每个子进程打开自己的 HDF5 文件句柄，并挂载到 dataset 对象上
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset._local_h5 = h5py.File(dataset.h5_path, 'r', swmr=True, libver='latest')

import jax
import jax.numpy as jnp
from jax import random, jit
from einops import rearrange, repeat
from functools import partial


class H5BatchParser:
    def __init__(self, config, H, W, probe_dim=2, use_probe=True, aux_dim=0):
        """
        config    : 配置对象，需含有 training.num_queries
        H, W      : 声速图高宽
        probe_dim : 换能器坐标维度（默认2）
        use_probe : 是否将换能器坐标纳入条件输入
        aux_dim   : 附加条件维度（若有）
        """
        self.num_queries = config.training.num_queries
        self.H, self.W = H, W
        self.use_probe = use_probe
        self.probe_dim = probe_dim
        self.aux_dim = aux_dim

        # 归一化坐标网格 [0,1]×[0,1]
        x_star = jnp.linspace(0, 1, H)
        y_star = jnp.linspace(0, 1, W)
        x_star, y_star = jnp.meshgrid(x_star, y_star, indexing="ij")
        self.coords = jnp.column_stack([x_star.ravel(), y_star.ravel()])  # (H*W, 2)

    def _build_conditions(self, Xp, X_aux=None):
        """将换能器坐标与辅助条件拼接为单一条件向量。返回 (bs, total_cond_dim) 或 None。"""
        parts = []
        if self.use_probe and Xp is not None:
            parts.append(Xp.reshape(Xp.shape[0], -1))
        if self.aux_dim > 0 and X_aux is not None:
            parts.append(X_aux.reshape(X_aux.shape[0], -1))
        if not parts:
            return None
        return jnp.concatenate(parts, axis=1)

    def random_query(self, batch, rng_key):
        """
        batch: ((Xb, Xp, X_aux), y)
        返回:
            batch_coords : (n_devices, num_queries, 2)
            rf_branch    : (bs, D_rf)   RF 数据（已展平）
            conditions   : (bs, D_cond) 其他条件拼接，若无则为 None
            batch_outputs: (bs, num_queries, 1)
        """
        (Xb, Xp, X_aux), y = batch
        bs = Xb.shape[0]

        # 展平输出场
        y_flat = rearrange(y, "b h w -> b (h w) 1") if y.ndim == 3 else \
                 rearrange(y, "b h w c -> b (h w) c")

        # 采样坐标
        query_idx = random.choice(rng_key, self.H * self.W,
                                  (self.num_queries,), replace=False)
        batch_coords = self.coords[query_idx]             # (num_queries, 2)
        batch_outputs = y_flat[:, query_idx, :]           # (bs, num_queries, 1)

        # 分离 RF 与条件
        rf_branch = Xb
        conditions = self._build_conditions(Xp, X_aux)    # 可能为 None

        # 多卡广播坐标
        batch_coords = repeat(batch_coords, "b d -> n b d", n=jax.device_count())

        return batch_coords, rf_branch, conditions, batch_outputs

    @partial(jit, static_argnums=(0,))
    def query_all(self, batch):
        """全量评估返回"""
        (Xb, Xp, X_aux), y = batch
        bs = Xb.shape[0]

        y_flat = rearrange(y, "b h w -> b (h w) 1") if y.ndim == 3 else \
                 rearrange(y, "b h w c -> b (h w) c")

        batch_coords = repeat(self.coords, "b d -> n b d", n=jax.device_count())
        rf_branch = Xb.reshape(bs, -1)
        conditions = self._build_conditions(Xp, X_aux)
        return batch_coords, rf_branch, conditions, y_flat   # 全量输出

def create_dataloader(dataset, batch_size, num_workers, shuffle=True, drop_last=True, worker_init_fn=None):
    num_devices = jax.device_count()

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size * num_devices,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=drop_last,
        worker_init_fn=worker_init_fn,
    )
    return data_loader