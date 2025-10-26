"""
评估指标
包含Autoencoder重构任务的评估指标计算
"""
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def calculate_loss(reconstructed, original, loss_type='MSE'):
    """
    计算重构损失
    
    Args:
        reconstructed: 重构的光谱 (batch_size, input_dim)
        original: 原始光谱 (batch_size, input_dim)
        loss_type: 损失函数类型
    
    Returns:
        loss: 损失值
    """
    if loss_type == 'MSE':
        criterion = nn.MSELoss()
    elif loss_type == 'MAE':
        criterion = nn.L1Loss()
    elif loss_type == 'SmoothL1':
        criterion = nn.SmoothL1Loss()
    else:
        raise ValueError(f"不支持的损失函数类型: {loss_type}")
    
    loss = criterion(reconstructed, original)
    return loss


def calculate_psnr(reconstructed, original, max_val=1.0):
    """
    计算峰值信噪比 (PSNR)
    
    Args:
        reconstructed: 重构的光谱
        original: 原始光谱
        max_val: 数据的最大值（归一化后通常为1.0）
    
    Returns:
        psnr: 峰值信噪比（dB）
    """
    mse = np.mean((reconstructed - original) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * np.log10(max_val / np.sqrt(mse))
    return psnr


class ReconstructionMetrics:
    """重构评估指标计算器"""
    
    def __init__(self):
        """初始化指标计算器"""
        self.reset()
    
    def reset(self):
        """重置所有累积的数据"""
        self.reconstructed_list = []
        self.original_list = []
    
    def update(self, reconstructed, original):
        """
        更新指标
        
        Args:
            reconstructed: 重构的光谱，可以是Tensor或ndarray
            original: 原始光谱，可以是Tensor或ndarray
        """
        # 转换为numpy数组
        if torch.is_tensor(reconstructed):
            reconstructed = reconstructed.detach().cpu().numpy()
        if torch.is_tensor(original):
            original = original.detach().cpu().numpy()
        
        self.reconstructed_list.append(reconstructed)
        self.original_list.append(original)
    
    def compute(self):
        """
        计算所有指标
        
        Returns:
            metrics: 包含所有指标的字典
        """
        if len(self.reconstructed_list) == 0:
            return {
                'MSE': 0.0,
                'RMSE': 0.0,
                'MAE': 0.0,
                'PSNR': 0.0,
            }
        
        # 合并所有batch的数据
        reconstructed = np.concatenate(self.reconstructed_list, axis=0)
        original = np.concatenate(self.original_list, axis=0)
        
        # 均方误差
        mse = mean_squared_error(original, reconstructed)
        
        # 均方根误差
        rmse = np.sqrt(mse)
        
        # 平均绝对误差
        mae = mean_absolute_error(original, reconstructed)
        
        # 峰值信噪比
        psnr = calculate_psnr(reconstructed, original)
        
        # 相关系数（每个样本的原始vs重构）
        correlations = []
        for i in range(len(reconstructed)):
            corr = np.corrcoef(original[i], reconstructed[i])[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
        avg_correlation = np.mean(correlations) if correlations else 0.0
        
        metrics = {
            'MSE': float(mse),
            'RMSE': float(rmse),
            'MAE': float(mae),
            'PSNR': float(psnr),
            'Correlation': float(avg_correlation),
        }
        
        return metrics
    
    def compute_and_reset(self):
        """计算指标并重置"""
        metrics = self.compute()
        self.reset()
        return metrics
    
    def print_metrics(self, metrics, prefix=''):
        """
        打印指标
        
        Args:
            metrics: 指标字典
            prefix: 打印前缀
        """
        print(f"{prefix}MSE:         {metrics['MSE']:.6f}")
        print(f"{prefix}RMSE:        {metrics['RMSE']:.6f}")
        print(f"{prefix}MAE:         {metrics['MAE']:.6f}")
        print(f"{prefix}PSNR:        {metrics['PSNR']:.2f} dB")
        print(f"{prefix}Correlation: {metrics['Correlation']:.4f}")


def evaluate_model(model, dataloader, device):
    """
    评估模型性能
    
    Args:
        model: Autoencoder模型
        dataloader: 数据加载器
        device: 设备
    
    Returns:
        metrics: 评估指标字典
        loss: 平均损失
    """
    model.eval()
    metrics_calculator = ReconstructionMetrics()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_data in dataloader:
            # 处理不同的数据格式
            if isinstance(batch_data, (tuple, list)):
                spectra = batch_data[0]  # 第一个元素是光谱数据
            else:
                spectra = batch_data
            
            spectra = spectra.to(device)
            
            # 前向传播
            reconstructed, _ = model(spectra)
            
            # 计算损失
            loss = calculate_loss(reconstructed, spectra, config.LOSS_FUNCTION)
            total_loss += loss.item()
            num_batches += 1
            
            # 更新指标
            metrics_calculator.update(reconstructed, spectra)
    
    # 计算平均损失
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    # 计算指标
    metrics = metrics_calculator.compute()
    
    return metrics, avg_loss


if __name__ == '__main__':
    # 测试指标计算
    print("测试重构指标计算...")
    
    # 创建模拟数据
    np.random.seed(42)
    original = np.random.rand(100, 737)
    # 添加一些重构误差
    reconstructed = original + np.random.normal(0, 0.05, original.shape)
    
    # 计算指标
    metrics_calc = ReconstructionMetrics()
    metrics_calc.update(reconstructed, original)
    metrics = metrics_calc.compute()
    
    print("\n指标结果:")
    metrics_calc.print_metrics(metrics, prefix="  ")
    
    # 测试Tensor输入
    print("\n\n测试Tensor输入...")
    tensor_reconstructed = torch.FloatTensor(reconstructed)
    tensor_original = torch.FloatTensor(original)
    
    metrics_calc.reset()
    metrics_calc.update(tensor_reconstructed, tensor_original)
    metrics2 = metrics_calc.compute()
    
    print("指标结果:")
    metrics_calc.print_metrics(metrics2, prefix="  ")

