"""
NIR光谱数据生成器
生成8种农产品的NIR光谱数据用于Autoencoder训练
"""
import numpy as np
import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class NIRProductDataGenerator:
    """农产品NIR光谱数据生成器"""
    
    def __init__(self, samples_per_product=50, random_seed=42):
        """
        初始化数据生成器
        
        Args:
            samples_per_product: 每种农产品的样本数量
            random_seed: 随机种子
        """
        self.samples_per_product = samples_per_product
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # 从配置文件获取参数
        self.input_dim = config.INPUT_DIM
        self.wavenumber_range = config.WAVENUMBER_RANGE
        self.product_names = config.PRODUCT_NAMES
        self.num_products = config.NUM_PRODUCTS
        self.origins = config.ORIGINS
        self.noise_level = config.DATA_CONFIG['noise_level']
        self.origin_variation = config.DATA_CONFIG['origin_variation']
        
        # 生成波数数组
        self.wavenumbers = np.linspace(
            self.wavenumber_range[0],
            self.wavenumber_range[1],
            self.input_dim
        )
        
        # 定义每种农产品的特征峰位置（索引）
        self.product_peak_profiles = {
            '红豆': [100, 200, 350, 500, 600],
            '当归': [80, 180, 300, 480, 620],
            '胡萝卜': [120, 220, 380, 520, 640],
            '大蒜': [90, 190, 320, 490, 610],
            '生姜': [110, 210, 360, 510, 630],
            '人参': [95, 195, 330, 495, 615],
            '大豆': [105, 205, 340, 505, 625],
            '小麦': [115, 215, 370, 515, 635],
        }
    
    def _generate_baseline(self, product_type):
        """
        生成基线吸收
        
        Args:
            product_type: 农产品类型
        
        Returns:
            baseline: 基线吸收数组
        """
        x = np.linspace(0, 1, self.input_dim)
        # 不同农产品有不同的基线形状
        offset = self.product_names.index(product_type) * 0.05
        baseline = 0.3 + offset + 0.15 * x - 0.08 * x**2 + 0.03 * np.sin(4 * np.pi * x)
        return baseline
    
    def _generate_peak(self, position, intensity, width=20):
        """
        生成高斯吸收峰
        
        Args:
            position: 峰位置（波段索引）
            intensity: 峰强度
            width: 峰宽度
        
        Returns:
            peak: 吸收峰数组
        """
        x = np.arange(self.input_dim)
        peak = intensity * np.exp(-((x - position) ** 2) / (2 * width ** 2))
        return peak
    
    def generate_spectrum(self, product_type, origin='国产'):
        """
        生成单个NIR光谱
        
        Args:
            product_type: 农产品类型
            origin: 产地类型（'国产' 或 '进口'）
        
        Returns:
            spectrum: 生成的NIR光谱
        """
        # 初始化谱图为基线
        spectrum = self._generate_baseline(product_type)
        
        # 获取该农产品的特征峰位置
        peak_positions = self.product_peak_profiles[product_type]
        
        # 添加特征峰
        for i, pos in enumerate(peak_positions):
            # 峰强度随位置变化
            base_intensity = 0.15 + 0.1 * (i / len(peak_positions))
            intensity = base_intensity * np.random.uniform(0.8, 1.2)
            
            # 产地差异：进口产品峰强度略有不同
            if origin == '进口':
                intensity *= (1 + np.random.uniform(-self.origin_variation, self.origin_variation))
            
            width = np.random.uniform(15, 25)
            spectrum += self._generate_peak(pos, intensity, width)
        
        # 添加一些次要峰（模拟其他化学成分）
        num_minor_peaks = np.random.randint(3, 6)
        for _ in range(num_minor_peaks):
            pos = np.random.randint(50, self.input_dim - 50)
            intensity = np.random.uniform(0.02, 0.06)
            width = np.random.uniform(15, 25)
            spectrum += self._generate_peak(pos, intensity, width)
        
        # 添加产地特异性特征
        if origin == '进口':
            # 进口产品在某些区域有特定变化
            spectrum[200:300] *= np.random.uniform(0.95, 1.05)
        
        # 添加噪声
        noise = np.random.normal(0, self.noise_level, self.input_dim)
        spectrum += noise
        
        # 确保光谱值为正
        spectrum = np.maximum(spectrum, 0)
        
        # 归一化到0-1范围
        spec_min = spectrum.min()
        spec_max = spectrum.max()
        spec_range = spec_max - spec_min
        
        if spec_range > 1e-10:
            spectrum = (spectrum - spec_min) / spec_range
        else:
            spectrum = np.full_like(spectrum, 0.5)
        
        return spectrum
    
    def generate_dataset(self):
        """
        生成完整数据集
        
        Returns:
            spectra: NIR光谱数组 (total_samples, input_dim)
            labels: 农产品类别标签 (total_samples,)
            origins: 产地标签 (total_samples,)
        """
        spectra = []
        labels = []
        origins = []
        
        for product_idx, product_name in enumerate(self.product_names):
            for i in range(self.samples_per_product):
                # 随机选择产地（国产或进口）
                origin = np.random.choice(self.origins)
                
                # 生成光谱
                spectrum = self.generate_spectrum(product_name, origin)
                
                spectra.append(spectrum)
                labels.append(product_idx)
                origins.append(origin)
        
        spectra = np.array(spectra, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)
        origins = np.array(origins)
        
        # 打乱顺序
        indices = np.random.permutation(len(spectra))
        spectra = spectra[indices]
        labels = labels[indices]
        origins = origins[indices]
        
        return spectra, labels, origins
    
    def save_dataset(self, save_dir):
        """
        生成并保存数据集
        
        Args:
            save_dir: 保存目录
        """
        total_samples = self.num_products * self.samples_per_product
        print(f"正在生成 {total_samples} 个NIR光谱样本...")
        print(f"  农产品种类: {self.num_products}")
        print(f"  每种样本数: {self.samples_per_product}")
        
        spectra, labels, origins = self.generate_dataset()
        
        # 保存为numpy文件
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, 'spectra.npy'), spectra)
        np.save(os.path.join(save_dir, 'labels.npy'), labels)
        np.save(os.path.join(save_dir, 'origins.npy'), origins)
        
        print(f"数据集已保存到 {save_dir}")
        print(f"光谱形状: {spectra.shape}")
        print(f"标签形状: {labels.shape}")
        print(f"产地形状: {origins.shape}")
        
        # 打印每种农产品的样本数
        print(f"\n各农产品样本分布:")
        for idx, product_name in enumerate(self.product_names):
            count = np.sum(labels == idx)
            print(f"  {product_name}: {count} 个样本")
        
        return spectra, labels, origins


def load_or_generate_data(data_dir, samples_per_product=50, regenerate=False):
    """
    加载或生成数据集
    
    Args:
        data_dir: 数据目录
        samples_per_product: 每种农产品的样本数量
        regenerate: 是否重新生成数据
    
    Returns:
        spectra, labels, origins
    """
    spectra_path = os.path.join(data_dir, 'spectra.npy')
    labels_path = os.path.join(data_dir, 'labels.npy')
    origins_path = os.path.join(data_dir, 'origins.npy')
    
    # 检查数据是否已存在
    if not regenerate and os.path.exists(spectra_path) and os.path.exists(labels_path):
        print("加载已有数据...")
        spectra = np.load(spectra_path)
        labels = np.load(labels_path)
        origins = np.load(origins_path)
        print(f"加载完成: {len(spectra)} 个样本")
    else:
        print("生成新数据...")
        generator = NIRProductDataGenerator(samples_per_product=samples_per_product)
        spectra, labels, origins = generator.save_dataset(data_dir)
    
    return spectra, labels, origins


if __name__ == '__main__':
    # 测试数据生成器
    generator = NIRProductDataGenerator(samples_per_product=10)
    spectra, labels, origins = generator.generate_dataset()
    
    print(f"\n数据统计:")
    print(f"光谱形状: {spectra.shape}")
    print(f"光谱范围: [{spectra.min():.3f}, {spectra.max():.3f}]")
    print(f"标签形状: {labels.shape}")
    print(f"标签范围: [{labels.min()}, {labels.max()}]")
    
    # 显示前5个样本的信息
    print(f"\n前5个样本的信息:")
    for i in range(5):
        product_name = config.PRODUCT_NAMES[labels[i]]
        origin = origins[i]
        print(f"  样本{i+1}: {product_name} ({origin})")

