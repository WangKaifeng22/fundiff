from function_diffusion.utils.h5_data_utils import H5Dataset
import h5py
import numpy as np

def create_dataset(config):
    """
    从 HDF5 文件中创建训练集和测试集。
    config 需包含：
        config.dataset.data_path          # HDF5 文件路径
        config.dataset.num_train_samples  # 训练样本数（测试集自动为剩余部分）
        config.seed                       # 随机种子（用于打乱索引）
    """
    path = config.dataset.data_path
    num_train = config.dataset.num_train_samples

    # 获取总样本数
    with h5py.File(path, 'r') as f:
        total_num = f['y'].shape[0]

    # 生成乱序索引
    indices = np.arange(total_num)
    """rng = np.random.default_rng(config.seed)
    rng.shuffle(indices)"""

    train_indices = indices[:num_train]
    test_indices = indices[num_train:]

    train_dataset = H5Dataset(path, train_indices)
    test_dataset = H5Dataset(path, test_indices)

    return train_dataset, test_dataset