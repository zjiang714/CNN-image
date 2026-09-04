"""
完整的演示脚本
展示整个图像去噪系统的功能
"""

import numpy as np
import torch
import argparse
from pathlib import Path
import json

from models import create_model
from preprocessor import ImagePreprocessor, create_noisy_clean_pairs
from trainer import DenoisingTrainer
from inference import DenoisingPipeline, create_sample_image, visualize_results


def demo_bitonic_filter():
    """
    演示双调滤波器
    """
    print("\n" + "="*60)
    print("演示 1: 双调滤波器基本功能")
    print("="*60)
    
    from bitonic_filter import BitonicFilter
    
    # 创建示例图像
    image = create_sample_image((128, 128), pattern='gradient')
    preprocessor = ImagePreprocessor()
    noisy = preprocessor.add_gaussian_noise(image, sigma=0.2)
    
    # 应用不同参数的双调滤波
    bf = BitonicFilter(kernel_size=3)
    
    print("\n应用双调滤波，参数组合：")
    
    params_list = [
        (0.3, 0.5, "低平滑，中等边界保留"),
        (0.5, 0.5, "中等平滑，中等边界保留"),
        (0.7, 0.7, "高平滑，高边界保留"),
    ]
    
    results = {}
    for alpha, beta, desc in params_list:
        filtered = bf.apply(noisy, np.array([alpha, beta]))
        
        # 计算MSE
        mse = np.mean((filtered - image) ** 2)
        psnr = 10 * np.log10(1.0 / mse) if mse > 0 else 100
        
        results[f"α={alpha}, β={beta}"] = {"psnr": psnr, "desc": desc}
        print(f"  α={alpha}, β={beta}: PSNR={psnr:.2f} - {desc}")
    
    return results


def demo_model_architecture():
    """
    演示模型架构
    """
    print("\n" + "="*60)
    print("演示 2: 模型架构和参数统计")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_model(in_channels=1, num_features=64, num_params=2, device=device)
    
    print(f"\n使用设备: {device}")
    print(f"\n完整模型结构:")
    print(model)
    
    # 计算参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n参数统计:")
    print(f"  总参数数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  模型大小: {total_params * 4 / (1024**2):.2f} MB (假设float32)")
    
    # 测试前向传播
    print(f"\n前向传播测试:")
    test_input = torch.randn(1, 1, 32, 32).to(device)
    with torch.no_grad():
        features, params = model(test_input)
    
    print(f"  输入形状: {test_input.shape}")
    print(f"  特征向量形状: {features.shape}")
    print(f"  预测参数: {params.cpu().numpy()}")
    print(f"  参数范围: [{params.min():.4f}, {params.max():.4f}]")


def demo_preprocessing():
    """
    演示预处理和分块
    """
    print("\n" + "="*60)
    print("演示 3: 图像预处理和分块")
    print("="*60)
    
    preprocessor = ImagePreprocessor(block_size=32, overlap=0)
    
    # 创建示例图像
    image = create_sample_image((128, 128), pattern='checkerboard')
    
    print(f"\n原始图像信息:")
    print(f"  大小: {image.shape}")
    print(f"  范围: [{image.min():.3f}, {image.max():.3f}]")
    
    # 分块
    blocks, positions = preprocessor.split_into_blocks(image)
    
    print(f"\n分块信息:")
    print(f"  块大小: 32×32")
    print(f"  块数量: {len(blocks)}")
    print(f"  覆盖范围: {len(blocks)} × 32² = {len(blocks) * 32 * 32} 像素")
    
    # 合并
    merged = preprocessor.merge_blocks(blocks, positions, image.shape[:2])
    
    print(f"\n合并后:")
    print(f"  大小: {merged.shape}")
    print(f"  重构误差 (MSE): {np.mean((merged - image) ** 2):.2e}")
    
    # 添加噪声
    noisy = preprocessor.add_gaussian_noise(image, sigma=0.15)
    
    print(f"\n噪声添加:")
    print(f"  噪声类型: 高斯噪声")
    print(f"  标准差 (σ): 0.15")
    print(f"  含噪图像范围: [{noisy.min():.3f}, {noisy.max():.3f}]")
    print(f"  噪声强度 (MSE): {np.mean((noisy - image) ** 2):.4f}")


def demo_training_pipeline():
    """
    演示训练流程（完整）
    """
    print("\n" + "="*60)
    print("演示 4: 完整训练流程")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建小规模数据集
    print("\n创建训练数据集...")
    preprocessor = ImagePreprocessor()
    
    # 3个小图像
    clean_images = [
        create_sample_image((128, 128), pattern='gradient'),
        create_sample_image((128, 128), pattern='checkerboard'),
        create_sample_image((128, 128), pattern='circles'),
    ]
    
    # 创建噪声对
    dataset = create_noisy_clean_pairs(clean_images, noise_sigma=0.15, num_noise_levels=2)
    
    # 分割
    split = int(len(dataset) * 0.8)
    train_data = dataset[:split]
    val_data = dataset[split:]
    
    print(f"  训练集: {len(train_data)} 个样本")
    print(f"  验证集: {len(val_data)} 个样本")
    
    # 创建模型和训练器
    print("\n创建模型...")
    model = create_model(in_channels=1, num_features=64, num_params=2, device=device)
    
    print("创建训练器...")
    trainer = DenoisingTrainer(model=model, device=device, learning_rate=1e-3, block_size=32)
    
    # 训练少量轮次展示
    print("\n开始训练 (10个epoch 展示)...\n")
    history = trainer.fit(
        train_data=train_data,
        val_data=val_data,
        epochs=10,
        save_dir=None  # 演示不保存
    )
    
    print("\n" + "-"*60)
    print("训练结果:")
    print("-"*60)
    print(f"初始 PSNR: {history['psnr'][0]:.2f} dB")
    print(f"最终 PSNR: {history['psnr'][-1]:.2f} dB")
    print(f"初始 SSIM: {history['ssim'][0]:.4f}")
    print(f"最终 SSIM: {history['ssim'][-1]:.4f}")
    print(f"PSNR改进: {history['psnr'][-1] - history['psnr'][0]:.2f} dB")
    
    return trainer, train_data, val_data


def demo_inference():
    """
    演示推理
    """
    print("\n" + "="*60)
    print("演示 5: 推理和去噪")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建管道
    print("\n初始化推理管道...")
    pipeline = DenoisingPipeline(device=device, block_size=32)
    
    # 创建测试图像
    print("创建测试图像...")
    clean = create_sample_image((256, 256), pattern='circles')
    preprocessor = ImagePreprocessor()
    noisy = preprocessor.add_gaussian_noise(clean, sigma=0.15)
    
    # 去噪
    print("进行去噪...")
    denoised = pipeline.denoise(noisy, normalize=False)
    
    # 计算指标
    print("\n去噪结果:")
    
    # 含噪 vs 干净
    mse_noisy = np.mean((noisy - clean) ** 2)
    psnr_noisy = 10 * np.log10(1.0 / mse_noisy) if mse_noisy > 0 else 100
    
    # 去噪 vs 干净
    mse_denoised = np.mean((denoised - clean) ** 2)
    psnr_denoised = 10 * np.log10(1.0 / mse_denoised) if mse_denoised > 0 else 100
    
    ssim_denoised = np.mean([
        np.corrcoef(denoised.flatten(), clean.flatten())[0, 1]
        for _ in range(1)
    ])
    
    print(f"  含噪图像 PSNR: {psnr_noisy:.2f} dB")
    print(f"  去噪图像 PSNR: {psnr_denoised:.2f} dB")
    print(f"  PSNR 改进: {psnr_denoised - psnr_noisy:.2f} dB")
    print(f"  去噪图像 SSIM: {ssim_denoised:.4f}")


def demo_parameter_analysis():
    """
    演示参数敏感性分析
    """
    print("\n" + "="*60)
    print("演示 6: 参数敏感性分析")
    print("="*60)
    
    from bitonic_filter import BitonicFilter
    
    # 创建图像
    image = create_sample_image((128, 128), pattern='gradient')
    preprocessor = ImagePreprocessor()
    noisy = preprocessor.add_gaussian_noise(image, sigma=0.2)
    
    print("\nalpha 参数影响 (beta=0.5):")
    print("-" * 40)
    
    bf = BitonicFilter()
    for alpha in [0.2, 0.4, 0.6, 0.8]:
        filtered = bf.apply(noisy, np.array([alpha, 0.5]))
        mse = np.mean((filtered - image) ** 2)
        psnr = 10 * np.log10(1.0 / mse) if mse > 0 else 100
        print(f"  α={alpha}: PSNR={psnr:.2f} dB")
    
    print("\nbeta 参数影响 (alpha=0.5):")
    print("-" * 40)
    
    for beta in [0.3, 0.5, 0.7, 0.9]:
        filtered = bf.apply(noisy, np.array([0.5, beta]))
        mse = np.mean((filtered - image) ** 2)
        psnr = 10 * np.log10(1.0 / mse) if mse > 0 else 100
        print(f"  β={beta}: PSNR={psnr:.2f} dB")


def main(args):
    """
    运行所有演示
    """
    print("\n" + "="*60)
    print("深度学习图像去噪系统 - 完整演示")
    print("="*60)
    
    # 创建输出目录
    Path("demo_output").mkdir(exist_ok=True)
    
    try:
        # 演示1：双调滤波器
        if args.demo == 'all' or args.demo == '1':
            demo_bitonic_filter()
        
        # 演示2：模型架构
        if args.demo == 'all' or args.demo == '2':
            demo_model_architecture()
        
        # 演示3：预处理
        if args.demo == 'all' or args.demo == '3':
            demo_preprocessing()
        
        # 演示4：训练
        if args.demo == 'all' or args.demo == '4':
            trainer, train_data, val_data = demo_training_pipeline()
        
        # 演示5：推理
        if args.demo == 'all' or args.demo == '5':
            demo_inference()
        
        # 演示6：参数分析
        if args.demo == 'all' or args.demo == '6':
            demo_parameter_analysis()
        
        print("\n" + "="*60)
        print("所有演示完成！")
        print("="*60)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='图像去噪系统演示')
    
    parser.add_argument('--demo', type=str, default='all',
                       choices=['all', '1', '2', '3', '4', '5', '6'],
                       help='运行指定的演示 (all=全部)')
    
    args = parser.parse_args()
    
    main(args)
