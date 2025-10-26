"""
重构脚本
从100维潜在特征重构737维光谱
"""
import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 导入项目模块
import config
from models import create_model
from utils import NIRDataModule, ReconstructionMetrics


class SpectrumReconstructor:
    """光谱重构器"""
    
    def __init__(self, model_path, device=None):
        """
        初始化重构器
        
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
    
    def reconstruct_from_spectrum(self, spectra):
        """
        从原始光谱重构（完整的编码-解码过程）
        
        Args:
            spectra: 原始光谱 (N, 737) 或 (737,)
        
        Returns:
            reconstructed: 重构的光谱
            latent: 潜在特征
        """
        # 确保维度正确
        is_single = False
        if spectra.ndim == 1:
            spectra = spectra.reshape(1, -1)
            is_single = True
        
        # 转换为tensor
        spectra_tensor = torch.FloatTensor(spectra).to(self.device)
        
        # 重构
        with torch.no_grad():
            reconstructed, latent = self.model(spectra_tensor)
        
        # 转换回numpy
        reconstructed = reconstructed.cpu().numpy()
        latent = latent.cpu().numpy()
        
        if is_single:
            reconstructed = reconstructed.squeeze(0)
            latent = latent.squeeze(0)
        
        return reconstructed, latent
    
    def reconstruct_from_latent(self, latent_features):
        """
        从潜在特征重构光谱
        
        Args:
            latent_features: 潜在特征 (N, 100) 或 (100,)
        
        Returns:
            reconstructed: 重构的光谱 (N, 737) 或 (737,)
        """
        # 确保维度正确
        is_single = False
        if latent_features.ndim == 1:
            latent_features = latent_features.reshape(1, -1)
            is_single = True
        
        # 转换为tensor
        latent_tensor = torch.FloatTensor(latent_features).to(self.device)
        
        # 解码
        with torch.no_grad():
            reconstructed = self.model.decode(latent_tensor)
        
        # 转换回numpy
        reconstructed = reconstructed.cpu().numpy()
        
        if is_single:
            reconstructed = reconstructed.squeeze(0)
        
        return reconstructed


def plot_reconstruction_comparison(original, reconstructed, product_name, origin, save_path):
    """
    绘制原始光谱与重构光谱对比图
    
    Args:
        original: 原始光谱
        reconstructed: 重构光谱
        product_name: 农产品名称
        origin: 产地
        save_path: 保存路径
    """
    wavenumbers = np.linspace(config.WAVENUMBER_RANGE[0], config.WAVENUMBER_RANGE[1], config.INPUT_DIM)
    
    plt.figure(figsize=(12, 5))
    
    # 子图1: 对比图
    plt.subplot(1, 2, 1)
    plt.plot(wavenumbers, original, 'b-', linewidth=1.5, label='原始光谱', alpha=0.7)
    plt.plot(wavenumbers, reconstructed, 'r--', linewidth=1.5, label='重构光谱', alpha=0.7)
    plt.xlabel('波数 (cm⁻¹)')
    plt.ylabel('吸光度 (归一化)')
    plt.title(f'{product_name} ({origin}) - 光谱对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图2: 误差图
    plt.subplot(1, 2, 2)
    error = reconstructed - original
    plt.plot(wavenumbers, error, 'g-', linewidth=1)
    plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    plt.xlabel('波数 (cm⁻¹)')
    plt.ylabel('重构误差')
    plt.title(f'重构误差 (MAE: {np.abs(error).mean():.4f})')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def demo_single_reconstruction():
    """单个光谱重构示例"""
    print("=" * 80)
    print("单个光谱重构示例")
    print("=" * 80)
    
    # 加载数据
    from data import load_or_generate_data
    spectra, labels, origins = load_or_generate_data(config.DATA_DIR)
    
    # 创建重构器
    reconstructor = SpectrumReconstructor(config.MODEL_SAVE_PATH)
    
    # 选择几个样本进行重构
    num_samples = 5
    selected_indices = np.random.choice(len(spectra), num_samples, replace=False)
    
    print(f"\n重构结果:")
    print("-" * 80)
    
    for i, idx in enumerate(selected_indices, 1):
        original = spectra[idx]
        label = labels[idx]
        origin = origins[idx]
        product_name = config.PRODUCT_NAMES[label]
        
        # 重构
        reconstructed, latent = reconstructor.reconstruct_from_spectrum(original)
        
        # 计算误差
        mse = np.mean((original - reconstructed) ** 2)
        mae = np.mean(np.abs(original - reconstructed))
        correlation = np.corrcoef(original, reconstructed)[0, 1]
        
        print(f"\n样本 {i}: {product_name} ({origin})")
        print(f"  MSE: {mse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  相关系数: {correlation:.4f}")
        
        # 保存对比图
        save_path = os.path.join(config.RESULT_DIR, f'reconstruction_sample_{i}.png')
        plot_reconstruction_comparison(original, reconstructed, product_name, origin, save_path)
        print(f"  对比图已保存: {save_path}")


def demo_batch_reconstruction():
    """批量重构示例"""
    print("\n" + "=" * 80)
    print("批量重构示例")
    print("=" * 80)
    
    # 创建数据模块
    data_module = NIRDataModule(config.DATA_DIR, batch_size=config.BATCH_SIZE)
    
    # 获取测试集
    _, _, test_loader = data_module.get_train_val_test_dataloaders()
    
    # 创建重构器
    reconstructor = SpectrumReconstructor(config.MODEL_SAVE_PATH)
    
    # 批量重构并评估
    metrics_calc = ReconstructionMetrics()
    
    print("\n正在进行批量重构...")
    
    with torch.no_grad():
        for batch_data in test_loader:
            # 处理数据
            if isinstance(batch_data, (tuple, list)):
                spectra = batch_data[0]
            else:
                spectra = batch_data
            
            spectra = spectra.to(reconstructor.device)
            
            # 重构
            reconstructed, _ = reconstructor.model(spectra)
            
            # 更新指标
            metrics_calc.update(reconstructed, spectra)
    
    # 计算指标
    metrics = metrics_calc.compute()
    
    print(f"\n批量重构性能:")
    metrics_calc.print_metrics(metrics, prefix="  ")


def demo_latent_to_spectrum():
    """从潜在特征重构光谱示例"""
    print("\n" + "=" * 80)
    print("从潜在特征重构光谱示例")
    print("=" * 80)
    
    # 加载降维后的潜在特征
    latent_path = os.path.join(config.RESULT_DIR, 'latent_features.npy')
    labels_path = os.path.join(config.RESULT_DIR, 'latent_labels.npy')
    
    if not os.path.exists(latent_path):
        print("⚠️  警告: 未找到潜在特征文件")
        print("   请先运行 'python dimension_reduction.py --mode batch' 生成潜在特征")
        return
    
    print(f"加载潜在特征: {latent_path}")
    latent_features = np.load(latent_path)
    labels = np.load(labels_path) if os.path.exists(labels_path) else None
    
    # 创建重构器
    reconstructor = SpectrumReconstructor(config.MODEL_SAVE_PATH)
    
    # 选择几个潜在特征进行重构
    num_samples = 3
    selected_indices = np.random.choice(len(latent_features), num_samples, replace=False)
    
    print(f"\n从潜在特征重构光谱:")
    print("-" * 80)
    
    for i, idx in enumerate(selected_indices, 1):
        latent = latent_features[idx]
        
        if labels is not None:
            label = labels[idx]
            product_name = config.PRODUCT_NAMES[label]
            print(f"\n样本 {i}: {product_name}")
        else:
            print(f"\n样本 {i}:")
        
        # 从潜在特征重构
        reconstructed = reconstructor.reconstruct_from_latent(latent)
        
        print(f"  潜在特征维度: {len(latent)}")
        print(f"  重构光谱维度: {len(reconstructed)}")
        print(f"  重构光谱范围: [{reconstructed.min():.3f}, {reconstructed.max():.3f}]")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='光谱重构')
    parser.add_argument('--mode', type=str, default='single',
                        choices=['single', 'batch', 'latent'],
                        help='重构模式')
    
    args = parser.parse_args()
    
    # 设置随机种子
    np.random.seed(42)
    
    if args.mode == 'single':
        demo_single_reconstruction()
    elif args.mode == 'batch':
        demo_batch_reconstruction()
    elif args.mode == 'latent':
        demo_latent_to_spectrum()

