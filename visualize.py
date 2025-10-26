"""
可视化脚本
可视化潜在空间和重构效果
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 导入项目模块
import config
from dimension_reduction import DimensionReducer
from utils import NIRDataModule


def visualize_latent_space_tsne(latent_features, labels, save_path):
    """
    使用t-SNE可视化潜在空间
    
    Args:
        latent_features: 潜在特征 (N, 100)
        labels: 类别标签 (N,)
        save_path: 保存路径
    """
    print("正在进行t-SNE降维...")
    
    # t-SNE降维到2维
    tsne = TSNE(
        n_components=config.TSNE_CONFIG['n_components'],
        perplexity=config.TSNE_CONFIG['perplexity'],
        n_iter=config.TSNE_CONFIG['n_iter'],
        random_state=config.TSNE_CONFIG['random_state']
    )
    latent_2d = tsne.fit_transform(latent_features)
    
    # 绘制散点图
    plt.figure(figsize=(12, 10))
    
    # 为每种农产品使用不同的颜色和标记
    colors = plt.cm.tab10(np.linspace(0, 1, config.NUM_PRODUCTS))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
    
    for product_idx, product_name in enumerate(config.PRODUCT_NAMES):
        mask = labels == product_idx
        plt.scatter(
            latent_2d[mask, 0],
            latent_2d[mask, 1],
            c=[colors[product_idx]],
            marker=markers[product_idx % len(markers)],
            label=product_name,
            s=100,
            alpha=0.7,
            edgecolors='black',
            linewidths=0.5
        )
    
    plt.xlabel('t-SNE 维度 1', fontsize=12)
    plt.ylabel('t-SNE 维度 2', fontsize=12)
    plt.title('潜在空间可视化 (t-SNE)', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"t-SNE可视化已保存到: {save_path}")
    plt.close()


def visualize_latent_space_pca(latent_features, labels, save_path):
    """
    使用PCA可视化潜在空间
    
    Args:
        latent_features: 潜在特征 (N, 100)
        labels: 类别标签 (N,)
        save_path: 保存路径
    """
    print("正在进行PCA降维...")
    
    # PCA降维到2维
    pca = PCA(n_components=config.PCA_CONFIG['n_components'])
    latent_2d = pca.fit_transform(latent_features)
    
    print(f"  解释方差比: {pca.explained_variance_ratio_}")
    print(f"  累计解释方差: {pca.explained_variance_ratio_.sum():.2%}")
    
    # 绘制散点图
    plt.figure(figsize=(12, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, config.NUM_PRODUCTS))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
    
    for product_idx, product_name in enumerate(config.PRODUCT_NAMES):
        mask = labels == product_idx
        plt.scatter(
            latent_2d[mask, 0],
            latent_2d[mask, 1],
            c=[colors[product_idx]],
            marker=markers[product_idx % len(markers)],
            label=product_name,
            s=100,
            alpha=0.7,
            edgecolors='black',
            linewidths=0.5
        )
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12)
    plt.title('潜在空间可视化 (PCA)', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"PCA可视化已保存到: {save_path}")
    plt.close()


def visualize_reconstruction_samples(data_module, reconstructor, num_samples=8):
    """
    可视化多个样本的重构效果
    
    Args:
        data_module: 数据模块
        reconstructor: 重构器
        num_samples: 显示的样本数量
    """
    print(f"正在生成{num_samples}个样本的重构对比图...")
    
    # 加载数据
    spectra = data_module.spectra
    labels = data_module.labels
    origins = data_module.origins
    
    # 从每种农产品中选择一个样本
    selected_indices = []
    for product_idx in range(min(num_samples, config.NUM_PRODUCTS)):
        mask = labels == product_idx
        indices = np.where(mask)[0]
        if len(indices) > 0:
            selected_idx = np.random.choice(indices)
            selected_indices.append(selected_idx)
    
    # 创建子图
    rows = (len(selected_indices) + 1) // 2
    cols = 2
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4*rows))
    axes = axes.flatten()
    
    wavenumbers = np.linspace(config.WAVENUMBER_RANGE[0], config.WAVENUMBER_RANGE[1], config.INPUT_DIM)
    
    for i, idx in enumerate(selected_indices):
        original = spectra[idx]
        label = labels[idx]
        origin = origins[idx]
        product_name = config.PRODUCT_NAMES[label]
        
        # 重构
        reconstructed, _ = reconstructor.reconstruct_from_spectrum(original)
        
        # 计算误差
        mae = np.mean(np.abs(original - reconstructed))
        correlation = np.corrcoef(original, reconstructed)[0, 1]
        
        # 绘制
        ax = axes[i]
        ax.plot(wavenumbers, original, 'b-', linewidth=1.5, label='原始', alpha=0.7)
        ax.plot(wavenumbers, reconstructed, 'r--', linewidth=1.5, label='重构', alpha=0.7)
        ax.set_xlabel('波数 (cm⁻¹)', fontsize=10)
        ax.set_ylabel('吸光度', fontsize=10)
        ax.set_title(f'{product_name} ({origin})\nMAE={mae:.4f}, Corr={correlation:.4f}', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # 删除多余的子图
    for i in range(len(selected_indices), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    
    save_path = os.path.join(config.RESULT_DIR, 'reconstruction_samples.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"重构样本对比图已保存到: {save_path}")
    plt.close()


def visualize_latent_distribution(latent_features, labels, save_path):
    """
    可视化潜在特征的分布
    
    Args:
        latent_features: 潜在特征 (N, 100)
        labels: 类别标签 (N,)
        save_path: 保存路径
    """
    print("正在生成潜在特征分布图...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 整体分布直方图
    ax = axes[0, 0]
    ax.hist(latent_features.flatten(), bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel('潜在特征值', fontsize=10)
    ax.set_ylabel('频数', fontsize=10)
    ax.set_title('潜在特征整体分布', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 2. 各维度的均值和标准差
    ax = axes[0, 1]
    dims = np.arange(config.LATENT_DIM)
    means = latent_features.mean(axis=0)
    stds = latent_features.std(axis=0)
    ax.errorbar(dims, means, yerr=stds, fmt='o', markersize=3, alpha=0.5, capsize=2)
    ax.set_xlabel('潜在维度', fontsize=10)
    ax.set_ylabel('特征值', fontsize=10)
    ax.set_title('各维度的均值和标准差', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 3. 不同农产品的潜在特征均值
    ax = axes[1, 0]
    for product_idx, product_name in enumerate(config.PRODUCT_NAMES):
        mask = labels == product_idx
        product_means = latent_features[mask].mean(axis=0)
        ax.plot(dims, product_means, label=product_name, alpha=0.7, linewidth=1.5)
    ax.set_xlabel('潜在维度', fontsize=10)
    ax.set_ylabel('平均特征值', fontsize=10)
    ax.set_title('各农产品的潜在特征均值', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # 4. 前10个维度的箱线图
    ax = axes[1, 1]
    data_to_plot = [latent_features[:, i] for i in range(min(10, config.LATENT_DIM))]
    bp = ax.boxplot(data_to_plot, labels=[f'D{i+1}' for i in range(len(data_to_plot))], patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax.set_xlabel('潜在维度', fontsize=10)
    ax.set_ylabel('特征值', fontsize=10)
    ax.set_title('前10个维度的分布（箱线图）', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"潜在特征分布图已保存到: {save_path}")
    plt.close()


def main():
    """主函数"""
    print("=" * 80)
    print("Autoencoder潜在空间和重构效果可视化")
    print("=" * 80)
    
    # 创建数据模块
    data_module = NIRDataModule(config.DATA_DIR, batch_size=config.BATCH_SIZE)
    
    # 创建降维器
    print("\n正在加载模型...")
    reducer = DimensionReducer(config.MODEL_SAVE_PATH)
    
    # 获取完整数据集加载器
    full_loader = data_module.get_full_dataset_loader()
    
    # 批量降维
    print("\n正在提取潜在特征...")
    latent_features, labels, origins = reducer.batch_reduce(full_loader)
    
    # 1. t-SNE可视化
    print("\n" + "=" * 80)
    print("1. t-SNE可视化")
    print("=" * 80)
    tsne_path = os.path.join(config.RESULT_DIR, 'latent_space_tsne.png')
    visualize_latent_space_tsne(latent_features, labels, tsne_path)
    
    # 2. PCA可视化
    print("\n" + "=" * 80)
    print("2. PCA可视化")
    print("=" * 80)
    pca_path = os.path.join(config.RESULT_DIR, 'latent_space_pca.png')
    visualize_latent_space_pca(latent_features, labels, pca_path)
    
    # 3. 潜在特征分布
    print("\n" + "=" * 80)
    print("3. 潜在特征分布")
    print("=" * 80)
    dist_path = os.path.join(config.RESULT_DIR, 'latent_distribution.png')
    visualize_latent_distribution(latent_features, labels, dist_path)
    
    # 4. 重构样本对比
    print("\n" + "=" * 80)
    print("4. 重构样本对比")
    print("=" * 80)
    from reconstruct import SpectrumReconstructor
    reconstructor = SpectrumReconstructor(config.MODEL_SAVE_PATH)
    visualize_reconstruction_samples(data_module, reconstructor, num_samples=8)
    
    print("\n" + "=" * 80)
    print("可视化完成！")
    print("=" * 80)
    print(f"\n所有图片已保存到: {config.RESULT_DIR}")


if __name__ == '__main__':
    # 设置随机种子
    np.random.seed(42)
    
    main()

