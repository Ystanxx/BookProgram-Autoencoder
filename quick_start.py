"""
快速开始脚本
演示数据生成、模型创建、编码解码和简单训练
"""
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
from data import NIRProductDataGenerator


def demo_data_generation():
    """演示数据生成"""
    print("=" * 80)
    print("1. 数据生成演示")
    print("=" * 80)
    
    # 创建数据生成器
    generator = NIRProductDataGenerator(samples_per_product=5)
    
    # 生成数据
    spectra, labels, origins = generator.generate_dataset()
    
    print(f"\n生成的数据:")
    print(f"  光谱形状: {spectra.shape}")
    print(f"  标签形状: {labels.shape}")
    print(f"  产地形状: {origins.shape}")
    print(f"  光谱范围: [{spectra.min():.3f}, {spectra.max():.3f}]")
    print(f"  农产品类别: {config.NUM_PRODUCTS}种")
    
    # 可视化不同农产品的光谱样本
    print(f"\n可视化8种农产品的代表性光谱...")
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    wavenumbers = np.linspace(config.WAVENUMBER_RANGE[0], config.WAVENUMBER_RANGE[1], config.INPUT_DIM)
    
    for product_idx, product_name in enumerate(config.PRODUCT_NAMES):
        ax = axes[product_idx]
        
        # 找到该农产品的样本
        mask = labels == product_idx
        product_spectra = spectra[mask]
        
        if len(product_spectra) > 0:
            # 显示该农产品的所有样本
            for i, spectrum in enumerate(product_spectra):
                origin = origins[mask][i]
                linestyle = '-' if origin == '国产' else '--'
                alpha = 0.6
                ax.plot(wavenumbers, spectrum, linestyle=linestyle, linewidth=1, alpha=alpha)
            
            ax.set_xlabel('波数 (cm⁻¹)', fontsize=9)
            ax.set_ylabel('吸光度', fontsize=9)
            ax.set_title(f'{product_name}', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(['国产', '进口'], fontsize=7)
    
    plt.tight_layout()
    save_path = 'results/sample_spectra.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"光谱图已保存到: {save_path}")
    plt.close()
    
    return spectra, labels, origins


def demo_model_creation():
    """演示模型创建"""
    print("\n" + "=" * 80)
    print("2. 模型创建演示")
    print("=" * 80)
    
    # 创建模型
    model = create_model()
    
    # 打印模型信息
    model_info = model.get_model_info()
    print(f"\n模型信息:")
    print(f"  模型名称: {model_info['model_name']}")
    print(f"  输入维度: {model_info['input_dim']}")
    print(f"  潜在维度: {model_info['latent_dim']}")
    print(f"  激活函数: {model_info['activation']}")
    print(f"  总参数量: {model_info['total_parameters']:,}")
    print(f"  编码器参数: {model_info['encoder_parameters']:,}")
    print(f"  解码器参数: {model_info['decoder_parameters']:,}")
    print(f"  压缩率: {model_info['compression_ratio']:.2f}x")
    
    # 打印模型结构
    print(f"\n模型结构:")
    print(model)
    
    return model


def demo_encode_decode(model, spectra):
    """演示编码和解码"""
    print("\n" + "=" * 80)
    print("3. 编码-解码演示")
    print("=" * 80)
    
    # 将模型设置为评估模式
    model.eval()
    model = model.to(config.DEVICE)
    
    # 选择几个样本
    batch_size = 4
    batch_spectra = spectra[:batch_size]
    
    # 转换为tensor
    batch_tensor = torch.FloatTensor(batch_spectra).to(config.DEVICE)
    
    print(f"\n输入光谱形状: {batch_tensor.shape}")
    
    with torch.no_grad():
        # 编码
        latent = model.encode(batch_tensor)
        print(f"潜在特征形状: {latent.shape}")
        print(f"潜在特征范围: [{latent.min().item():.3f}, {latent.max().item():.3f}]")
        
        # 解码
        reconstructed = model.decode(latent)
        print(f"重构光谱形状: {reconstructed.shape}")
        
        # 计算重构误差
        mse = torch.mean((batch_tensor - reconstructed) ** 2).item()
        mae = torch.mean(torch.abs(batch_tensor - reconstructed)).item()
        
        print(f"\n重构误差（未训练）:")
        print(f"  MSE: {mse:.6f}")
        print(f"  MAE: {mae:.6f}")
    
    # 可视化一个样本的编码-解码过程
    print(f"\n可视化编码-解码过程...")
    
    original = batch_spectra[0]
    reconstructed_np = reconstructed[0].cpu().numpy()
    latent_np = latent[0].cpu().numpy()
    
    wavenumbers = np.linspace(config.WAVENUMBER_RANGE[0], config.WAVENUMBER_RANGE[1], config.INPUT_DIM)
    
    fig = plt.figure(figsize=(14, 5))
    
    # 子图1: 原始光谱
    ax1 = plt.subplot(1, 3, 1)
    ax1.plot(wavenumbers, original, 'b-', linewidth=1.5)
    ax1.set_xlabel('波数 (cm⁻¹)')
    ax1.set_ylabel('吸光度')
    ax1.set_title(f'原始光谱 ({config.INPUT_DIM}维)')
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 潜在特征
    ax2 = plt.subplot(1, 3, 2)
    ax2.bar(range(config.LATENT_DIM), latent_np, color='green', alpha=0.7)
    ax2.set_xlabel('潜在维度')
    ax2.set_ylabel('特征值')
    ax2.set_title(f'潜在特征 ({config.LATENT_DIM}维)')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 子图3: 重构光谱
    ax3 = plt.subplot(1, 3, 3)
    ax3.plot(wavenumbers, original, 'b-', linewidth=1.5, label='原始', alpha=0.7)
    ax3.plot(wavenumbers, reconstructed_np, 'r--', linewidth=1.5, label='重构', alpha=0.7)
    ax3.set_xlabel('波数 (cm⁻¹)')
    ax3.set_ylabel('吸光度')
    ax3.set_title(f'重构光谱（MSE={mse:.4f}）')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = 'results/encode_decode_demo.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"编码-解码过程图已保存到: {save_path}")
    plt.close()


def demo_simple_training():
    """演示简单的训练循环"""
    print("\n" + "=" * 80)
    print("4. 简单训练演示 (10个epoch)")
    print("=" * 80)
    
    # 生成小量数据
    generator = NIRProductDataGenerator(samples_per_product=10)
    spectra, labels, origins = generator.generate_dataset()
    
    # 创建模型
    model = create_model()
    model = model.to(config.DEVICE)
    
    # 准备数据
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    spectra_norm = scaler.fit_transform(spectra)
    
    # 转换为tensor
    X = torch.FloatTensor(spectra_norm).to(config.DEVICE)
    
    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    
    # 训练几个epoch
    model.train()
    losses = []
    
    print("\n开始训练...")
    for epoch in range(1, 11):
        optimizer.zero_grad()
        
        # 前向传播
        reconstructed, latent = model(X)
        
        # 计算损失
        loss = criterion(reconstructed, X)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        print(f"Epoch {epoch:2d}/10 - Loss: {loss.item():.6f}")
    
    print(f"\n训练完成！损失从 {losses[0]:.6f} 降到 {losses[-1]:.6f}")
    print(f"损失减少: {(1 - losses[-1]/losses[0]) * 100:.1f}%")
    
    # 评估
    model.eval()
    with torch.no_grad():
        reconstructed, latent = model(X)
        final_loss = criterion(reconstructed, X)
        
        # 计算更多指标
        mse = final_loss.item()
        mae = torch.mean(torch.abs(reconstructed - X)).item()
        
        # 计算相关系数
        X_np = X.cpu().numpy()
        reconstructed_np = reconstructed.cpu().numpy()
        correlations = []
        for i in range(len(X_np)):
            corr = np.corrcoef(X_np[i], reconstructed_np[i])[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
        avg_corr = np.mean(correlations)
        
        print(f"\n最终评估:")
        print(f"  MSE:  {mse:.6f}")
        print(f"  MAE:  {mae:.6f}")
        print(f"  平均相关系数: {avg_corr:.4f}")
    
    # 绘制训练曲线
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(losses) + 1), losses, 'b-o', linewidth=2, markersize=6)
    plt.xlabel('Epoch')
    plt.ylabel('重构损失 (MSE)')
    plt.title('简单训练演示 - 损失曲线')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = 'results/simple_training_curve.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"训练曲线已保存到: {save_path}")
    plt.close()


def main():
    """主函数"""
    print("欢迎使用Autoencoder NIR光谱降维系统")
    print(f"使用设备: {config.DEVICE}\n")
    
    # 1. 数据生成演示
    spectra, labels, origins = demo_data_generation()
    
    # 2. 模型创建演示
    model = demo_model_creation()
    
    # 3. 编码-解码演示
    demo_encode_decode(model, spectra)
    
    # 4. 简单训练演示
    demo_simple_training()
    
    print("\n" + "=" * 80)
    print("快速开始演示完成！")
    print("=" * 80)
    print("\n下一步:")
    print("  1. 运行 'python train.py' 开始完整的50轮训练")
    print("  2. 运行 'python dimension_reduction.py --mode batch' 进行批量降维")
    print("  3. 运行 'python reconstruct.py --mode single' 查看重构效果")
    print("  4. 运行 'python visualize.py' 可视化潜在空间")
    print("\n详细说明请参考 使用指南.md")


if __name__ == '__main__':
    main()

