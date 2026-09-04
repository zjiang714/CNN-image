"""
训练脚本
完整的训练流程
支持合成数据集和真实数据集（DIV2K、BSD68、SET12）
"""

import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple
import argparse

from models import create_model
from preprocessor import ImagePreprocessor, create_noisy_clean_pairs
from trainer import DenoisingTrainer
from inference import create_sample_image
from dataset_manager import prepare_real_dataset


def create_synthetic_dataset(num_images: int = 10, image_size: Tuple[int, int] = (256, 256),
                            noise_sigma: float = 0.15) -> Tuple[List, List]:
    """
    创建合成训练数据集
    
    Args:
        num_images: 图像数量
        image_size: 图像大小
        noise_sigma: 噪声标准差
        
    Returns:
        (训练数据, 验证数据)
    """
    print("创建合成数据集...")
    preprocessor = ImagePreprocessor()
    
    # 创建示例图像
    patterns = ['gradient', 'checkerboard', 'circles']
    clean_images = []
    
    for i in range(num_images):
        pattern = patterns[i % len(patterns)]
        clean_image = create_sample_image(size=image_size, pattern=pattern)
        clean_images.append(clean_image)
    
    # 创建噪声对
    dataset = create_noisy_clean_pairs(clean_images, noise_sigma=noise_sigma,
                                      num_noise_levels=2)
    
    # 分割训练和验证集
    split = int(len(dataset) * 0.8)
    train_data = dataset[:split]
    val_data = dataset[split:]
    
    print(f"训练集: {len(train_data)} 个样本")
    print(f"验证集: {len(val_data)} 个样本")
    
    return train_data, val_data


def main(args):
    """
    主训练函数
    
    Args:
        args: 命令行参数
    """
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu'
    print(f"使用设备: {device}")
    
    # 创建输出目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建数据集
    if args.dataset == 'synthetic':
        print("\n创建合成数据集...")
        train_data, val_data = create_synthetic_dataset(
            num_images=args.num_images,
            image_size=(args.image_size, args.image_size),
            noise_sigma=args.noise_sigma
        )
    else:
        # 使用真实数据集
        print(f"\n加载真实数据集: {args.dataset}")
        image_paths, images = prepare_real_dataset(
            dataset_name=args.dataset,
            max_images=args.num_images,
            data_dir=args.data_dir
        )
        
        if not images:
            print("✗ 数据集加载失败，请检查网络连接或手动下载")
            return
        
        # 创建带噪数据对
        dataset = create_noisy_clean_pairs(images, noise_sigma=args.noise_sigma,
                                          num_noise_levels=args.num_noise_levels)
        
        # 分割训练和验证集
        split = int(len(dataset) * 0.8)
        train_data = dataset[:split]
        val_data = dataset[split:]
        
        print(f"训练集: {len(train_data)} 个样本")
        print(f"验证集: {len(val_data)} 个样本")
    
    # 创建模型
    print("\n创建模型...")
    model = create_model(in_channels=1, num_features=args.num_features,
                        num_params=8, device=device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数总数: {total_params:,}")
    
    # 创建训练器
    trainer = DenoisingTrainer(
        model=model,
        device=device,
        learning_rate=args.learning_rate,
        block_size=args.block_size,
        num_params=8
    )
    
    # 训练
    print("\n开始训练...\n")
    history = trainer.fit(
        train_data=train_data,
        val_data=val_data,
        epochs=args.epochs,
        save_dir=str(save_dir)
    )
    
    # 保存模型和历史记录
    print("\n保存模型...")
    trainer.save_model(str(save_dir / "final_model.pth"))
    
    # 保存训练历史（转换 numpy 类型为 Python 原生类型）
    import json
    
    # 递归转换 numpy/torch 类型为 Python 原生类型
    def convert_to_serializable(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, torch.Tensor):
            return float(obj.item())
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    history_serializable = convert_to_serializable(history)
    
    with open(save_dir / "history.json", 'w') as f:
        json.dump(history_serializable, f, indent=2)
    
    print(f"模型已保存到 {save_dir}")
    
    # 打印最终结果
    print("\n" + "="*50)
    print("训练完成！")
    print("="*50)
    print(f"最终 PSNR: {history['psnr'][-1]:.2f}")
    print(f"最终 SSIM: {history['ssim'][-1]:.4f}")
    print(f"最终验证损失: {history['val_loss'][-1]:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练去噪模型')
    
    # 数据集相关参数
    parser.add_argument('--dataset', type=str, default='synthetic',
                       choices=['synthetic', 'div2k_train', 'div2k_valid', 'bsd68', 'set12'],
                       help='数据集选择。默认: synthetic (生成合成数据)\n'
                            '  div2k_train: DIV2K 训练集 (800张)\n'
                            '  div2k_valid: DIV2K 验证集 (100张)\n'
                            '  bsd68: BSD68 测试集 (68张)\n'
                            '  set12: SET12 标准集 (12张)')
    parser.add_argument('--data-dir', type=str, default='datasets',
                       help='数据集目录。默认: datasets。\n'
                            '可直接指向已下载的 DIV2K_train_HR 或 DIV2K_valid_HR 目录')
    parser.add_argument('--num-images', type=int, default=10,
                       help='使用的图像数量。默认: 10\n'
                            '  (对于合成数据) 生成的合成图像数\n'
                            '  (对于真实数据) 最多加载的图像数')
    parser.add_argument('--num-noise-levels', type=int, default=2,
                       help='每张干净图像生成多少个噪声版本。默认: 2')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮次。默认: 100')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='批次大小。默认: 1')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                       help='学习率。默认: 1e-3')
    parser.add_argument('--block-size', type=int, default=32,
                       help='图像块大小。默认: 32')
    parser.add_argument('--num-features', type=int, default=64,
                       help='特征维度。默认: 64')
    
    # 图像处理参数（仅合成数据）
    parser.add_argument('--image-size', type=int, default=256,
                       help='合成图像大小。默认: 256')
    parser.add_argument('--noise-sigma', type=float, default=0.15,
                       help='高斯噪声标准差。默认: 0.15')
    
    # 输出参数
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                       help='模型保存目录。默认: checkpoints')
    parser.add_argument('--use-cuda', action='store_true',
                       help='使用GPU。默认: 使用CPU')
    
    args = parser.parse_args()
    
    main(args)
