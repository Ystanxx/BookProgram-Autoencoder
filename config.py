"""
配置文件
定义Autoencoder谱图降维系统的所有超参数和配置
"""
import os
import torch

# ==================== 数据配置 ====================
# NIR光谱数据配置
INPUT_DIM = 737  # NIR光谱波段点数量
WAVENUMBER_RANGE = (9100, 3900)  # 波数范围 (cm⁻¹)

# 农产品类别配置
PRODUCT_NAMES = [
    '红豆', '当归', '胡萝卜', '大蒜', 
    '生姜', '人参', '大豆', '小麦'
]
NUM_PRODUCTS = len(PRODUCT_NAMES)  # 8种农产品

# 样本配置
SAMPLES_PER_PRODUCT = 50  # 每种农产品的样本数量
TOTAL_SAMPLES = NUM_PRODUCTS * SAMPLES_PER_PRODUCT  # 总样本数：400
ORIGINS = ['国产', '进口']  # 产地类型

# ==================== 模型配置 ====================
# Autoencoder配置
LATENT_DIM = 100  # 潜在空间维度（压缩后的维度）
ACTIVATION = 'SELU'  # 激活函数类型

# 模型架构
MODEL_CONFIG = {
    'input_dim': INPUT_DIM,      # 输入维度 737
    'latent_dim': LATENT_DIM,    # 潜在维度 100
    'activation': ACTIVATION,     # SELU激活函数
}

# ==================== 训练配置 ====================
# 基本训练参数
BATCH_SIZE = 32
EPOCHS = 50  # 50轮训练
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5

# 早停策略（可选，因为只训练50轮）
EARLY_STOPPING_PATIENCE = 10

# 学习率调度
LR_SCHEDULER = {
    'type': 'ReduceLROnPlateau',
    'factor': 0.5,
    'patience': 5,
    'min_lr': 1e-6,
}

# 损失函数
LOSS_FUNCTION = 'MSE'  # 均方误差重构损失

# ==================== 数据集配置 ====================
# 数据划分比例
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# 模拟数据生成参数
DATA_CONFIG = {
    'samples_per_product': SAMPLES_PER_PRODUCT,
    'noise_level': 0.02,           # 噪声水平
    'random_seed': 42,             # 随机种子
    'origin_variation': 0.1,       # 产地差异程度
}

# ==================== 路径配置 ====================
# 基础路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'saved_models')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
RESULT_DIR = os.path.join(BASE_DIR, 'results')

# 创建必要的目录
for directory in [DATA_DIR, MODEL_DIR, LOG_DIR, RESULT_DIR]:
    os.makedirs(directory, exist_ok=True)

# 模型保存路径
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, 'autoencoder_best.pth')

# ==================== 设备配置 ====================
# 设备选择配置
# 选项: 'auto' - 自动检测, 'cuda' - 强制GPU, 'cpu' - 强制CPU
DEVICE_MODE = 'cpu'  # 默认使用CPU以确保兼容性

def get_device():
    """智能检测并返回可用设备"""
    global DEVICE_MODE
    
    # 检查环境变量
    env_force_cpu = os.environ.get('FORCE_CPU', '').lower() in ('1', 'true', 'yes')
    if env_force_cpu:
        print("💡 环境变量FORCE_CPU已设置，使用CPU模式")
        return torch.device('cpu')
    
    # 强制CPU模式
    if DEVICE_MODE == 'cpu':
        print("💡 配置设定为CPU模式")
        return torch.device('cpu')
    
    # 强制CUDA模式
    if DEVICE_MODE == 'cuda':
        if torch.cuda.is_available():
            print("💡 配置设定为CUDA模式")
            return torch.device('cuda')
        else:
            print("⚠️  警告: CUDA不可用，切换到CPU模式")
            return torch.device('cpu')
    
    # 自动检测模式
    if torch.cuda.is_available():
        try:
            # 尝试在CUDA上创建一个小张量来测试兼容性
            test_tensor = torch.zeros(1).cuda()
            del test_tensor
            torch.cuda.empty_cache()
            print("✅ CUDA兼容性测试通过，使用GPU加速")
            return torch.device('cuda')
        except Exception as e:
            print(f"\n⚠️  警告: CUDA设备不兼容")
            print(f"   原因: {str(e)[:80]}...")
            print("   自动切换到CPU模式")
            print("   提示: 如需使用GPU，请确保PyTorch版本支持您的显卡")
            print(f"   或在config.py中设置 DEVICE_MODE = 'cpu'")
            return torch.device('cpu')
    else:
        print("💡 CUDA不可用，使用CPU模式")
        return torch.device('cpu')

DEVICE = get_device()

# ==================== 可视化配置 ====================
PLOT_CONFIG = {
    'dpi': 150,
    'figsize': (12, 8),
    'save_format': 'png',
}

# t-SNE可视化配置
TSNE_CONFIG = {
    'n_components': 2,      # 降维到2维
    'perplexity': 30,       # 困惑度
    'n_iter': 1000,         # 迭代次数
    'random_state': 42,
}

# PCA可视化配置
PCA_CONFIG = {
    'n_components': 2,      # 降维到2维
}

# ==================== 日志配置 ====================
LOG_CONFIG = {
    'log_interval': 5,      # 每隔多少个batch打印一次日志
    'save_interval': 5,     # 每隔多少个epoch保存一次检查点
}

print(f"配置加载完成，使用设备: {DEVICE}")
if DEVICE.type == 'cpu':
    print("💡 提示: 如果您有兼容的GPU，可以在config.py中修改 DEVICE_MODE = 'auto' 来启用GPU加速")

