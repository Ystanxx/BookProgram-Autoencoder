"""
降维脚本
使用训练好的Autoencoder将737维光谱压缩到100维潜在特征
"""
import os
import sys
import torch
import numpy as np

# 导入项目模块
import config
from models import create_model
from utils import NIRDataModule


class DimensionReducer:
    """光谱降维器"""
    
    def __init__(self, model_path, device=None):
        """
        初始化降维器
        
        Args:
            model_path: 模型路径
            device: 设备（None则使用配置中的设备）
        """
        self.device = device if device is not None else config.DEVICE
        self.model = None
        self.model_path = model_path
        
        # 加载模型
        self.load_model()
    
    def load_model(self):
        """加载训练好的模型"""
        print(f"正在加载模型: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        # 创建模型
        self.model = create_model()
        self.model = self.model.to(self.device)
        
        # 加载权重
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"模型加载成功！")
        if 'val_metrics' in checkpoint:
            print(f"  验证PSNR: {checkpoint['val_metrics']['PSNR']:.2f} dB")
            print(f"  验证相关: {checkpoint['val_metrics']['Correlation']:.4f}")
    
    def reduce_dimension(self, spectra):
        """
        将光谱降维到潜在空间
        
        Args:
            spectra: 光谱数据 (N, 737) 或 (737,)
        
        Returns:
            latent_features: 潜在特征 (N, 100) 或 (100,)
        """
        # 确保维度正确
        is_single = False
        if spectra.ndim == 1:
            spectra = spectra.reshape(1, -1)
            is_single = True
        
        # 转换为tensor
        spectra_tensor = torch.FloatTensor(spectra).to(self.device)
        
        # 编码
        with torch.no_grad():
            latent_features = self.model.encode(spectra_tensor)
        
        # 转换回numpy
        latent_features = latent_features.cpu().numpy()
        
        if is_single:
            latent_features = latent_features.squeeze(0)
        
        return latent_features
    
    def batch_reduce(self, dataloader):
        """
        批量降维
        
        Args:
            dataloader: 数据加载器
        
        Returns:
            all_latent_features: 所有潜在特征
            all_labels: 所有标签（如果有）
            all_origins: 所有产地（如果有）
        """
        all_latent_features = []
        all_labels = []
        all_origins = []
        
        print("正在进行批量降维...")
        
        with torch.no_grad():
            for batch_data in dataloader:
                # 处理不同的数据格式
                if isinstance(batch_data, (tuple, list)):
                    spectra = batch_data[0].to(self.device)
                    if len(batch_data) > 1:
                        labels = batch_data[1]
                        all_labels.extend(labels.cpu().numpy())
                    if len(batch_data) > 2:
                        origins = batch_data[2]
                        all_origins.extend(origins)
                else:
                    spectra = batch_data.to(self.device)
                
                # 编码
                latent = self.model.encode(spectra)
                all_latent_features.append(latent.cpu().numpy())
        
        all_latent_features = np.concatenate(all_latent_features, axis=0)
        
        print(f"降维完成！")
        print(f"  原始维度: {config.INPUT_DIM}")
        print(f"  降维后维度: {config.LATENT_DIM}")
        print(f"  样本数量: {len(all_latent_features)}")
        print(f"  压缩率: {config.INPUT_DIM / config.LATENT_DIM:.2f}x")
        print(f"  空间节省: {(1 - config.LATENT_DIM / config.INPUT_DIM) * 100:.1f}%")
        
        if all_labels:
            all_labels = np.array(all_labels)
        else:
            all_labels = None
        
        if all_origins:
            all_origins = np.array(all_origins)
        else:
            all_origins = None
        
        return all_latent_features, all_labels, all_origins


def demo_single_spectrum_reduction():
    """单个光谱降维示例"""
    print("=" * 80)
    print("单个光谱降维示例")
    print("=" * 80)
    
    # 加载数据
    from data import load_or_generate_data
    spectra, labels, origins = load_or_generate_data(config.DATA_DIR)
    
    # 创建降维器
    reducer = DimensionReducer(config.MODEL_SAVE_PATH)
    
    # 选择几个样本进行降维
    num_samples = 5
    selected_indices = np.random.choice(len(spectra), num_samples, replace=False)
    
    print(f"\n降维结果:")
    print("-" * 80)
    
    for i, idx in enumerate(selected_indices, 1):
        spectrum = spectra[idx]
        label = labels[idx]
        origin = origins[idx]
        product_name = config.PRODUCT_NAMES[label]
        
        # 降维
        latent_feature = reducer.reduce_dimension(spectrum)
        
        # 显示信息
        print(f"\n样本 {i}: {product_name} ({origin})")
        print(f"  原始维度: {len(spectrum)}")
        print(f"  降维后维度: {len(latent_feature)}")
        print(f"  潜在特征范围: [{latent_feature.min():.3f}, {latent_feature.max():.3f}]")
        print(f"  潜在特征均值: {latent_feature.mean():.3f}")
        print(f"  潜在特征标准差: {latent_feature.std():.3f}")


def demo_batch_reduction():
    """批量降维示例"""
    print("\n" + "=" * 80)
    print("批量降维示例")
    print("=" * 80)
    
    # 创建数据模块
    data_module = NIRDataModule(config.DATA_DIR, batch_size=config.BATCH_SIZE)
    
    # 获取完整数据集加载器
    full_loader = data_module.get_full_dataset_loader()
    
    # 创建降维器
    reducer = DimensionReducer(config.MODEL_SAVE_PATH)
    
    # 批量降维
    latent_features, labels, origins = reducer.batch_reduce(full_loader)
    
    # 保存降维后的特征
    save_path = os.path.join(config.RESULT_DIR, 'latent_features.npy')
    np.save(save_path, latent_features)
    print(f"\n潜在特征已保存到: {save_path}")
    
    if labels is not None:
        labels_path = os.path.join(config.RESULT_DIR, 'latent_labels.npy')
        np.save(labels_path, labels)
        print(f"标签已保存到: {labels_path}")
    
    # 显示统计信息
    print(f"\n潜在特征统计:")
    print(f"  形状: {latent_features.shape}")
    print(f"  均值: {latent_features.mean():.3f}")
    print(f"  标准差: {latent_features.std():.3f}")
    print(f"  最小值: {latent_features.min():.3f}")
    print(f"  最大值: {latent_features.max():.3f}")
    
    # 显示每种农产品的潜在特征统计
    if labels is not None:
        print(f"\n各农产品潜在特征统计:")
        for product_idx, product_name in enumerate(config.PRODUCT_NAMES):
            mask = labels == product_idx
            product_features = latent_features[mask]
            if len(product_features) > 0:
                print(f"  {product_name}:")
                print(f"    样本数: {len(product_features)}")
                print(f"    均值: {product_features.mean():.3f}")
                print(f"    标准差: {product_features.std():.3f}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='光谱降维')
    parser.add_argument('--mode', type=str, default='batch',
                        choices=['single', 'batch'],
                        help='降维模式')
    
    args = parser.parse_args()
    
    # 设置随机种子
    np.random.seed(42)
    
    if args.mode == 'single':
        demo_single_spectrum_reduction()
    elif args.mode == 'batch':
        demo_batch_reduction()

