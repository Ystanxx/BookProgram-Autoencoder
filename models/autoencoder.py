"""
Autoencoder模型定义
用于NIR光谱降维和特征重构
"""
import torch
import torch.nn as nn
import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class Autoencoder(nn.Module):
    """单隐层Autoencoder模型"""
    
    def __init__(self, input_dim=737, latent_dim=100, activation='SELU'):
        """
        初始化Autoencoder
        
        Args:
            input_dim: 输入维度（光谱波段点数）
            latent_dim: 潜在空间维度（压缩后的维度）
            activation: 激活函数类型（'SELU', 'ReLU', 'Tanh'）
        """
        super(Autoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.activation_type = activation
        
        # 选择激活函数
        if activation == 'SELU':
            self.activation = nn.SELU()
        elif activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'Tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"不支持的激活函数: {activation}")
        
        # ==================== 编码器 ====================
        # 将737维光谱压缩到100维潜在特征
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            self.activation
        )
        
        # ==================== 解码器 ====================
        # 将100维潜在特征重构回737维光谱
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim),
            self.activation
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # SELU激活函数推荐使用lecun_normal初始化
                if self.activation_type == 'SELU':
                    nn.init.normal_(m.weight, 0, 1 / (m.in_features ** 0.5))
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def encode(self, x):
        """
        编码：将输入光谱压缩到潜在空间
        
        Args:
            x: 输入光谱 (batch_size, input_dim)
        
        Returns:
            latent: 潜在特征 (batch_size, latent_dim)
        """
        latent = self.encoder(x)
        return latent
    
    def decode(self, latent):
        """
        解码：从潜在特征重构光谱
        
        Args:
            latent: 潜在特征 (batch_size, latent_dim)
        
        Returns:
            reconstructed: 重构的光谱 (batch_size, input_dim)
        """
        reconstructed = self.decoder(latent)
        return reconstructed
    
    def forward(self, x):
        """
        前向传播：编码 -> 解码
        
        Args:
            x: 输入光谱 (batch_size, input_dim)
        
        Returns:
            reconstructed: 重构的光谱 (batch_size, input_dim)
            latent: 潜在特征 (batch_size, latent_dim)
        """
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent
    
    def get_latent_features(self, x):
        """
        获取潜在特征（用于降维）
        
        Args:
            x: 输入光谱 (batch_size, input_dim)
        
        Returns:
            latent: 潜在特征 (batch_size, latent_dim)
        """
        with torch.no_grad():
            latent = self.encode(x)
        return latent
    
    def reconstruct(self, x):
        """
        重构光谱（不返回潜在特征）
        
        Args:
            x: 输入光谱 (batch_size, input_dim)
        
        Returns:
            reconstructed: 重构的光谱 (batch_size, input_dim)
        """
        with torch.no_grad():
            reconstructed, _ = self.forward(x)
        return reconstructed
    
    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        
        compression_ratio = self.input_dim / self.latent_dim
        
        info = {
            'model_name': 'Autoencoder',
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
            'activation': self.activation_type,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'encoder_parameters': encoder_params,
            'decoder_parameters': decoder_params,
            'compression_ratio': compression_ratio,
        }
        return info


def create_model(config_dict=None):
    """
    创建Autoencoder模型的工厂函数
    
    Args:
        config_dict: 配置字典，如果为None则使用默认配置
    
    Returns:
        model: Autoencoder模型实例
    """
    if config_dict is None:
        config_dict = config.MODEL_CONFIG
    
    model = Autoencoder(
        input_dim=config_dict['input_dim'],
        latent_dim=config_dict['latent_dim'],
        activation=config_dict['activation']
    )
    
    return model


if __name__ == '__main__':
    # 测试模型
    print("测试Autoencoder模型...")
    
    # 创建模型
    model = create_model()
    model.eval()
    
    # 打印模型信息
    info = model.get_model_info()
    print("\n模型信息:")
    for key, value in info.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # 测试前向传播
    batch_size = 4
    test_input = torch.randn(batch_size, config.INPUT_DIM)
    
    with torch.no_grad():
        reconstructed, latent = model(test_input)
    
    print(f"\n前向传播测试:")
    print(f"  输入形状: {test_input.shape}")
    print(f"  潜在特征形状: {latent.shape}")
    print(f"  重构输出形状: {reconstructed.shape}")
    
    # 测试压缩率
    input_size = test_input.numel() * 4 / 1024  # KB (假设float32)
    latent_size = latent.numel() * 4 / 1024
    print(f"\n数据压缩:")
    print(f"  原始数据大小: {input_size:.2f} KB")
    print(f"  压缩后大小: {latent_size:.2f} KB")
    print(f"  压缩率: {info['compression_ratio']:.2f}x")
    print(f"  空间节省: {(1 - latent_size/input_size) * 100:.1f}%")
    
    print("\n模型结构:")
    print(model)

