"""
配置文件示例
用于快速调整系统参数
"""

import json
from pathlib import Path


# ============================================
# 模型配置
# ============================================
MODEL_CONFIG = {
    "name": "DenoisingNetwork",
    "in_channels": 1,  # 灰度图
    "num_features": 64,  # CNN特征维度
    "num_params": 2,  # 滤波器参数个数 [alpha, beta]
}

# ============================================
# 双调滤波器配置
# ============================================
FILTER_CONFIG = {
    "kernel_size": 3,  # 滤波核大小
    "alpha_range": [0.0, 1.0],  # alpha参数范围
    "beta_range": [0.0, 1.0],  # beta参数范围
    "alpha_default": 0.5,  # 默认alpha值
    "beta_default": 0.5,  # 默认beta值
}

# ============================================
# 训练配置
# ============================================
TRAINING_CONFIG = {
    # 基本参数
    "epochs": 100,
    "batch_size": 1,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    
    # 学习率调度
    "lr_scheduler": {
        "type": "StepLR",
        "step_size": 50,
        "gamma": 0.5,  # 每50个epoch乘以0.5
    },
    
    # 损失函数权重
    "loss_weights": {
        "reconstruction": 1.0,  # MSE损失权重
        "regularization": 0.1,  # 参数正则化权重
    },
    
    # 数据相关
    "noise_sigma": 0.15,  # 高斯噪声标准差
    "num_noise_levels": 2,  # 噪声等级数量
}

# ============================================
# 数据配置
# ============================================
DATA_CONFIG = {
    # 预处理
    "preprocessing": {
        "normalize": True,  # 是否归一化到[0,1]
        "grayscale": True,  # 转为灰度图
        "resize": None,  # 是否调整大小，None表示不调整
    },
    
    # 分块
    "blocking": {
        "block_size": 32,  # 块大小
        "overlap": 0,  # 块之间的重叠像素
    },
    
    # 数据集
    "dataset": {
        "num_images": 10,  # 合成数据集中的图像数
        "image_size": 256,  # 图像大小
        "train_val_split": 0.8,  # 训练集比例
        "patterns": ["gradient", "checkerboard", "circles"],  # 图案类型
    },
}

# ============================================
# 推理配置
# ============================================
INFERENCE_CONFIG = {
    "device": "cuda",  # "cuda" 或 "cpu"
    "batch_inference": False,  # 是否批量推理
    "pad_image": True,  # 是否填充图像
    "blend_overlaps": True,  # 重叠区域是否混合
}

# ============================================
# 输出和保存配置
# ============================================
OUTPUT_CONFIG = {
    "save_dir": "checkpoints",
    "model_name_format": "{prefix}_{epoch:03d}.pth",
    "log_dir": "logs",
    "save_frequency": 10,  # 每N个epoch保存一次
    "keep_best_only": True,  # 只保存最佳模型
    "save_training_state": True,  # 保存优化器等状态
}

# ============================================
# 评估配置
# ============================================
EVALUATION_CONFIG = {
    "metrics": ["psnr", "ssim", "mae", "mse"],
    "psnr_threshold": 30,  # PSNR优秀阈值
    "ssim_threshold": 0.9,  # SSIM优秀阈值
    "verbose": True,  # 是否详细输出
}


def load_config(config_path: str = None) -> dict:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径，如果为None则返回默认配置
        
    Returns:
        配置字典
    """
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    
    return get_default_config()


def get_default_config() -> dict:
    """
    获取默认配置
    
    Returns:
        完整的默认配置字典
    """
    return {
        "model": MODEL_CONFIG,
        "filter": FILTER_CONFIG,
        "training": TRAINING_CONFIG,
        "data": DATA_CONFIG,
        "inference": INFERENCE_CONFIG,
        "output": OUTPUT_CONFIG,
        "evaluation": EVALUATION_CONFIG,
    }


def save_config(config: dict, config_path: str):
    """
    保存配置文件
    
    Args:
        config: 配置字典
        config_path: 保存路径
    """
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def print_config(config: dict = None):
    """
    打印配置
    
    Args:
        config: 配置字典，如果为None则打印默认配置
    """
    if config is None:
        config = get_default_config()
    
    print("="*60)
    print("系统配置")
    print("="*60)
    
    for section, values in config.items():
        print(f"\n[{section.upper()}]")
        if isinstance(values, dict):
            for key, value in values.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in value.items():
                        print(f"    {k}: {v}")
                else:
                    print(f"  {key}: {value}")
        else:
            print(f"  {values}")


# ============================================
# 使用示例
# ============================================
if __name__ == "__main__":
    # 获取默认配置
    config = get_default_config()
    
    # 打印配置
    print_config(config)
    
    # 保存配置
    save_config(config, "config.json")
    print("\n配置已保存到 config.json")
    
    # 加载配置
    loaded_config = load_config("config.json")
    print("\n配置已从 config.json 加载")
