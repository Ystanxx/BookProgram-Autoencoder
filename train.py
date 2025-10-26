"""
训练脚本
实现Autoencoder模型的训练
"""
import os
import sys
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import json
import time

# 配置matplotlib
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 导入项目模块
import config
from models import create_model
from utils import NIRDataModule, ReconstructionMetrics, calculate_loss, evaluate_model


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience=10, min_delta=0.0, mode='min'):
        """
        初始化早停
        
        Args:
            patience: 耐心值（多少个epoch没改善就停止）
            min_delta: 最小改善幅度
            mode: 'min'表示越小越好，'max'表示越大越好
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        """
        检查是否应该早停
        
        Args:
            score: 当前分数（例如验证损失）
        
        Returns:
            is_best: 是否是最佳模型
        """
        if self.best_score is None:
            self.best_score = score
            return True
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


def train_one_epoch(model, train_loader, optimizer, device, epoch):
    """
    训练一个epoch
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        device: 设备
        epoch: 当前epoch
    
    Returns:
        avg_loss: 平均损失
        metrics: 训练指标
    """
    model.train()
    total_loss = 0.0
    metrics_calculator = ReconstructionMetrics()
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, batch_data in enumerate(pbar):
        # 处理不同的数据格式
        if isinstance(batch_data, (tuple, list)):
            spectra = batch_data[0]  # 第一个元素是光谱数据
        else:
            spectra = batch_data
        
        spectra = spectra.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        reconstructed, latent = model(spectra)
        
        # 计算损失
        loss = calculate_loss(reconstructed, spectra, config.LOSS_FUNCTION)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 记录
        total_loss += loss.item()
        metrics_calculator.update(reconstructed, spectra)
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.6f}',
        })
    
    avg_loss = total_loss / len(train_loader)
    metrics = metrics_calculator.compute()
    
    return avg_loss, metrics


def validate(model, val_loader, device):
    """
    验证模型
    
    Args:
        model: 模型
        val_loader: 验证数据加载器
        device: 设备
    
    Returns:
        avg_loss: 平均损失
        metrics: 验证指标
    """
    return evaluate_model(model, val_loader, device)


def train_model(save_results=True):
    """
    训练Autoencoder模型
    
    Args:
        save_results: 是否保存结果
    
    Returns:
        training_history: 训练历史
    """
    print("=" * 80)
    print("开始训练Autoencoder模型")
    print("=" * 80)
    print(f"\n配置信息:")
    print(f"  设备: {config.DEVICE}")
    print(f"  批次大小: {config.BATCH_SIZE}")
    print(f"  学习率: {config.LEARNING_RATE}")
    print(f"  训练轮数: {config.EPOCHS}")
    print(f"  潜在维度: {config.LATENT_DIM}")
    print(f"  激活函数: {config.ACTIVATION}")
    
    # 创建数据模块
    data_module = NIRDataModule(
        data_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=0
    )
    
    # 获取数据加载器
    train_loader, val_loader, test_loader = data_module.get_train_val_test_dataloaders(
        train_ratio=config.TRAIN_RATIO,
        val_ratio=config.VAL_RATIO,
        random_state=42
    )
    
    # 创建模型
    model = create_model()
    model = model.to(config.DEVICE)
    
    # 打印模型信息
    model_info = model.get_model_info()
    print(f"\n模型信息:")
    print(f"  模型名称: {model_info['model_name']}")
    print(f"  输入维度: {model_info['input_dim']}")
    print(f"  潜在维度: {model_info['latent_dim']}")
    print(f"  总参数量: {model_info['total_parameters']:,}")
    print(f"  压缩率: {model_info['compression_ratio']:.2f}x")
    
    # 优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config.LR_SCHEDULER['factor'],
        patience=config.LR_SCHEDULER['patience'],
        min_lr=config.LR_SCHEDULER['min_lr'],
        verbose=True
    )
    
    # 早停
    early_stopping = EarlyStopping(
        patience=config.EARLY_STOPPING_PATIENCE,
        mode='min'
    )
    
    # 训练历史
    train_losses = []
    val_losses = []
    train_metrics_history = []
    val_metrics_history = []
    best_val_loss = float('inf')
    
    # 训练循环
    print(f"\n开始训练...")
    for epoch in range(1, config.EPOCHS + 1):
        # 训练
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, config.DEVICE, epoch
        )
        
        # 验证
        val_metrics, val_loss = validate(model, val_loader, config.DEVICE)
        
        # 记录
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_metrics_history.append(train_metrics)
        val_metrics_history.append(val_metrics)
        
        # 打印信息
        print(f"\nEpoch {epoch}/{config.EPOCHS}:")
        print(f"  训练损失: {train_loss:.6f}")
        print(f"  验证损失: {val_loss:.6f}")
        print(f"  验证PSNR: {val_metrics['PSNR']:.2f} dB")
        print(f"  验证相关: {val_metrics['Correlation']:.4f}")
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 保存最佳模型
        is_best = early_stopping(val_loss)
        if is_best:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'model_info': model_info,
            }, config.MODEL_SAVE_PATH)
            print(f"  ✓ 保存最佳模型 (验证损失: {val_loss:.6f})")
        
        # 早停检查
        if early_stopping.early_stop:
            print(f"\n早停触发！在第 {epoch} 轮停止训练")
            break
    
    # 加载最佳模型并在测试集上评估
    print(f"\n加载最佳模型进行最终评估...")
    checkpoint = torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics, test_loss = validate(model, test_loader, config.DEVICE)
    
    print(f"\n最终测试集结果:")
    print(f"  MSE:         {test_metrics['MSE']:.6f}")
    print(f"  RMSE:        {test_metrics['RMSE']:.6f}")
    print(f"  MAE:         {test_metrics['MAE']:.6f}")
    print(f"  PSNR:        {test_metrics['PSNR']:.2f} dB")
    print(f"  Correlation: {test_metrics['Correlation']:.4f}")
    
    # 保存训练历史
    training_history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_metrics': train_metrics_history,
        'val_metrics': val_metrics_history,
        'test_metrics': test_metrics,
        'test_loss': test_loss,
        'best_epoch': checkpoint['epoch'],
        'model_info': model_info,
    }
    
    # 保存结果
    if save_results:
        # 保存JSON结果
        results_path = os.path.join(config.RESULT_DIR, 'training_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            # 转换numpy类型为Python原生类型
            def convert_to_serializable(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(item) for item in obj]
                return obj
            
            serializable_history = convert_to_serializable(training_history)
            json.dump(serializable_history, f, indent=4, ensure_ascii=False)
        print(f"\n结果已保存到: {results_path}")
        
        # 绘制训练曲线
        plot_training_curves(train_losses, val_losses)
    
    return training_history


def plot_training_curves(train_losses, val_losses):
    """
    绘制训练曲线
    
    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
    """
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='训练损失', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='验证损失', linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('重构损失 (MSE)')
    plt.title('Autoencoder训练曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    plot_path = os.path.join(config.RESULT_DIR, 'training_curves.png')
    plt.savefig(plot_path, dpi=config.PLOT_CONFIG['dpi'], bbox_inches='tight')
    print(f"训练曲线已保存到: {plot_path}")
    plt.close()


if __name__ == '__main__':
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 开始训练
    start_time = time.time()
    training_history = train_model(save_results=True)
    end_time = time.time()
    
    print(f"\n总训练时间: {(end_time - start_time) / 60:.2f} 分钟")
    print("\n训练完成！")

