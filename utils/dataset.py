"""
PyTorch数据集类定义
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class NIRDataset(Dataset):
    """NIR光谱数据集类（用于Autoencoder）"""
    
    def __init__(self, spectra, labels=None, origins=None, transform=None):
        """
        初始化数据集
        
        Args:
            spectra: NIR光谱数据 (N, input_dim)
            labels: 农产品类别标签 (N,)，可选
            origins: 产地标签 (N,)，可选
            transform: 数据变换函数
        """
        self.spectra = torch.FloatTensor(spectra)
        self.labels = torch.LongTensor(labels) if labels is not None else None
        self.origins = origins  # 保持为numpy数组或列表
        self.transform = transform
        
    def __len__(self):
        return len(self.spectra)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        Returns:
            spectrum: 光谱数据
            label: 类别标签（如果有）
            origin: 产地标签（如果有）
        """
        spectrum = self.spectra[idx]
        
        if self.transform:
            spectrum = self.transform(spectrum)
        
        if self.labels is not None:
            label = self.labels[idx]
            if self.origins is not None:
                return spectrum, label, self.origins[idx]
            return spectrum, label
        
        return spectrum


class NIRDataModule:
    """数据模块，管理数据加载和预处理"""
    
    def __init__(self, data_dir, batch_size=32, num_workers=0):
        """
        初始化数据模块
        
        Args:
            data_dir: 数据目录
            batch_size: 批次大小
            num_workers: 数据加载工作进程数
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.scaler = StandardScaler()
        
        # 加载数据
        self.load_data()
        
    def load_data(self):
        """加载数据文件"""
        from data.data_generator import load_or_generate_data
        
        self.spectra, self.labels, self.origins = load_or_generate_data(
            self.data_dir,
            samples_per_product=config.DATA_CONFIG['samples_per_product'],
            regenerate=False
        )
        
        print(f"\n数据加载完成:")
        print(f"  光谱形状: {self.spectra.shape}")
        print(f"  标签形状: {self.labels.shape}")
        print(f"  农产品种类: {config.NUM_PRODUCTS}")
    
    def normalize_spectra(self, train_spectra, val_spectra=None, test_spectra=None):
        """
        标准化光谱数据
        
        Args:
            train_spectra: 训练集光谱
            val_spectra: 验证集光谱（可选）
            test_spectra: 测试集光谱（可选）
        
        Returns:
            标准化后的数据
        """
        # 在训练集上拟合scaler
        train_spectra_normalized = self.scaler.fit_transform(train_spectra)
        
        results = [train_spectra_normalized]
        
        if val_spectra is not None:
            val_spectra_normalized = self.scaler.transform(val_spectra)
            results.append(val_spectra_normalized)
        
        if test_spectra is not None:
            test_spectra_normalized = self.scaler.transform(test_spectra)
            results.append(test_spectra_normalized)
        
        return results if len(results) > 1 else results[0]
    
    def get_train_val_test_dataloaders(self, train_ratio=0.7, val_ratio=0.15, random_state=42):
        """
        获取训练集、验证集和测试集的数据加载器
        
        Args:
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            random_state: 随机种子
        
        Returns:
            train_loader, val_loader, test_loader
        """
        # 首先划分训练集和临时集（验证+测试）
        temp_ratio = val_ratio + (1 - train_ratio - val_ratio)
        train_spectra, temp_spectra, train_labels, temp_labels, train_origins, temp_origins = train_test_split(
            self.spectra, self.labels, self.origins,
            test_size=temp_ratio,
            random_state=random_state,
            stratify=self.labels  # 按类别分层采样
        )
        
        # 然后将临时集划分为验证集和测试集
        val_size = val_ratio / temp_ratio
        val_spectra, test_spectra, val_labels, test_labels, val_origins, test_origins = train_test_split(
            temp_spectra, temp_labels, temp_origins,
            test_size=(1 - val_size),
            random_state=random_state,
            stratify=temp_labels
        )
        
        # 标准化数据
        train_spectra_norm, val_spectra_norm, test_spectra_norm = self.normalize_spectra(
            train_spectra, val_spectra, test_spectra
        )
        
        # 创建数据集
        train_dataset = NIRDataset(train_spectra_norm, train_labels, train_origins)
        val_dataset = NIRDataset(val_spectra_norm, val_labels, val_origins)
        test_dataset = NIRDataset(test_spectra_norm, test_labels, test_origins)
        
        # 创建数据加载器
        # 只有在实际使用GPU时才启用pin_memory
        use_pin_memory = (config.DEVICE.type == 'cuda')
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=use_pin_memory
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=use_pin_memory
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=use_pin_memory
        )
        
        print(f"\n数据集划分:")
        print(f"  训练集: {len(train_dataset)} 样本")
        print(f"  验证集: {len(val_dataset)} 样本")
        print(f"  测试集: {len(test_dataset)} 样本")
        
        return train_loader, val_loader, test_loader
    
    def get_full_dataset_loader(self):
        """
        获取完整数据集的加载器（用于降维/可视化）
        
        Returns:
            full_loader: 完整数据集加载器
        """
        # 标准化数据
        spectra_normalized = self.scaler.fit_transform(self.spectra)
        
        # 创建数据集
        full_dataset = NIRDataset(spectra_normalized, self.labels, self.origins)
        
        # 创建数据加载器
        full_loader = DataLoader(
            full_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        
        return full_loader


if __name__ == '__main__':
    # 测试数据集类
    data_module = NIRDataModule(
        data_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE
    )
    
    # 测试数据划分
    train_loader, val_loader, test_loader = data_module.get_train_val_test_dataloaders()
    
    # 测试一个batch
    for spectra, labels, origins in train_loader:
        print(f"\nBatch测试:")
        print(f"  光谱形状: {spectra.shape}")
        print(f"  标签形状: {labels.shape}")
        print(f"  产地数量: {len(origins)}")
        print(f"  标签示例: {labels[:5].numpy()}")
        print(f"  产地示例: {origins[:5]}")
        break

