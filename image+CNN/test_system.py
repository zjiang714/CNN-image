"""
完整的系统测试脚本
验证所有模块的功能
"""

import sys
import traceback
import numpy as np
from pathlib import Path


def test_imports():
    """测试所有导入"""
    print("\n" + "="*60)
    print("测试1: 模块导入")
    print("="*60)
    
    tests = [
        ("numpy", lambda: __import__('numpy')),
        ("torch", lambda: __import__('torch')),
        ("cv2", lambda: __import__('cv2')),
        ("matplotlib", lambda: __import__('matplotlib')),
        ("bitonic_filter", lambda: __import__('bitonic_filter')),
        ("models", lambda: __import__('models')),
        ("preprocessor", lambda: __import__('preprocessor')),
        ("trainer", lambda: __import__('trainer')),
        ("inference", lambda: __import__('inference')),
    ]
    
    passed = 0
    for name, import_func in tests:
        try:
            import_func()
            print(f"  ✓ {name}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {name}: {e}")
    
    return passed, len(tests)


def test_bitonic_filter():
    """测试双调滤波器"""
    print("\n" + "="*60)
    print("测试2: 双调滤波器")
    print("="*60)
    
    from bitonic_filter import BitonicFilter
    
    try:
        # 创建滤波器
        bf = BitonicFilter(kernel_size=3, alpha=0.5, beta=0.5)
        
        # 创建测试图像
        image = np.random.rand(64, 64).astype(np.float32)
        
        # 应用滤波
        filtered = bf.apply(image)
        
        assert filtered.shape == image.shape, "输出形状不匹配"
        assert filtered.min() >= 0 and filtered.max() <= 1, "输出范围超出[0,1]"
        
        print("  ✓ BitonicFilter初始化")
        print("  ✓ 单通道去噪")
        
        # 测试彩色图像
        image_rgb = np.random.rand(64, 64, 3).astype(np.float32)
        filtered_rgb = bf.apply(image_rgb)
        assert filtered_rgb.shape == image_rgb.shape, "RGB输出形状不匹配"
        
        print("  ✓ 彩色图像处理")
        
        # 测试参数预测
        params = np.array([0.7, 0.6])
        filtered_with_params = bf.apply(image, params)
        
        print("  ✓ 参数化滤波")
        
        return 4, 4
        
    except Exception as e:
        print(f"  ✗ 错误: {e}")
        traceback.print_exc()
        return 0, 4


def test_preprocessor():
    """测试预处理模块"""
    print("\n" + "="*60)
    print("测试3: 图像预处理")
    print("="*60)
    
    from preprocessor import ImagePreprocessor
    
    try:
        preprocessor = ImagePreprocessor(block_size=32)
        
        # 归一化测试
        image_uint8 = (np.random.rand(128, 128) * 255).astype(np.uint8)
        normalized = preprocessor.normalize(image_uint8)
        assert normalized.dtype == np.float32, "归一化类型错误"
        assert normalized.min() >= 0 and normalized.max() <= 1, "归一化范围错误"
        print("  ✓ 图像归一化")
        
        # 反归一化测试
        denormalized = preprocessor.denormalize(normalized)
        assert denormalized.dtype == np.uint8, "反归一化类型错误"
        assert denormalized.min() >= 0 and denormalized.max() <= 255, "反归一化范围错误"
        print("  ✓ 图像反归一化")
        
        # 分块测试
        image = np.random.rand(256, 256).astype(np.float32)
        blocks, positions = preprocessor.split_into_blocks(image)
        assert len(blocks) > 0, "分块数量为0"
        assert all(b.shape == (32, 32) for b in blocks), "块大小不正确"
        print(f"  ✓ 图像分块 ({len(blocks)} 个块)")
        
        # 合并测试
        merged = preprocessor.merge_blocks(blocks, positions, image.shape[:2])
        assert merged.shape == image.shape, "合并后形状不匹配"
        mse = np.mean((merged - image) ** 2)
        print(f"  ✓ 块合并 (MSE: {mse:.2e})")
        
        # 噪声添加测试
        clean = np.ones((128, 128), dtype=np.float32) * 0.5
        noisy = preprocessor.add_gaussian_noise(clean, sigma=0.1)
        noise_level = np.std(noisy - clean)
        assert 0.08 < noise_level < 0.12, "噪声水平不符合预期"
        print(f"  ✓ 高斯噪声添加 (σ={noise_level:.4f})")
        
        # 填充测试
        image = np.random.rand(100, 100).astype(np.float32)
        padded, (pad_h, pad_w) = preprocessor.pad_image(image)
        unpadded = preprocessor.unpad_image(padded, pad_h, pad_w)
        assert unpadded.shape == image.shape, "填充/去填充失败"
        print("  ✓ 图像填充处理")
        
        return 6, 6
        
    except Exception as e:
        print(f"  ✗ 错误: {e}")
        traceback.print_exc()
        return 0, 6


def test_models():
    """测试模型"""
    print("\n" + "="*60)
    print("测试4: CNN模型")
    print("="*60)
    
    try:
        import torch
        from models import create_model, FeatureExtractor, ParameterPredictor
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"  使用设备: {device}")
        
        # 创建模型
        model = create_model(in_channels=1, num_features=64, num_params=2, device=device)
        print("  ✓ 模型创建")
        
        # 前向传播测试
        input_tensor = torch.randn(1, 1, 32, 32, device=device)
        with torch.no_grad():
            features, params = model(input_tensor)
        
        assert features.shape == (1, 64), f"特征形状错误: {features.shape}"
        assert params.shape == (1, 2), f"参数形状错误: {params.shape}"
        assert params.min() >= 0 and params.max() <= 1, "参数范围错误"
        print("  ✓ 前向传播")
        
        # 参数统计
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  ✓ 模型参数: {total_params:,}")
        
        # 梯度测试
        input_tensor.requires_grad = True
        features, params = model(input_tensor)
        loss = features.mean() + params.mean()
        loss.backward()
        print("  ✓ 梯度计算")
        
        return 4, 4
        
    except Exception as e:
        print(f"  ✗ 错误: {e}")
        traceback.print_exc()
        return 0, 4


def test_trainer():
    """测试训练器"""
    print("\n" + "="*60)
    print("测试5: 训练器")
    print("="*60)
    
    try:
        import torch
        from models import create_model
        from trainer import DenoisingTrainer
        from preprocessor import ImagePreprocessor
        from inference import create_sample_image
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 创建小规模数据
        clean = create_sample_image((128, 128), 'gradient')
        preprocessor = ImagePreprocessor()
        noisy = preprocessor.add_gaussian_noise(clean, sigma=0.15)
        
        # 创建训练器
        model = create_model(device=device)
        trainer = DenoisingTrainer(model, device=device)
        print("  ✓ 训练器创建")
        
        # 训练步骤
        result = trainer.train_step(noisy, clean)
        assert 'loss' in result, "缺少loss"
        assert 'psnr' in result, "缺少psnr"
        assert 'ssim' in result, "缺少ssim"
        print(f"  ✓ 训练步骤 (Loss: {result['loss']:.4f})")
        
        # 验证步骤
        result = trainer.validate(noisy, clean)
        assert 'loss' in result, "缺少验证loss"
        print(f"  ✓ 验证步骤 (PSNR: {result['psnr']:.2f})")
        
        # 模型保存/加载
        trainer.save_model('/tmp/test_model.pth')
        trainer.load_model('/tmp/test_model.pth')
        print("  ✓ 模型保存和加载")
        
        return 4, 4
        
    except Exception as e:
        print(f"  ✗ 错误: {e}")
        traceback.print_exc()
        return 0, 4


def test_inference():
    """测试推理管道"""
    print("\n" + "="*60)
    print("测试6: 推理管道")
    print("="*60)
    
    try:
        import numpy as np
        from inference import DenoisingPipeline, create_sample_image
        from preprocessor import ImagePreprocessor
        
        # 创建测试数据
        clean = create_sample_image((128, 128), 'checkerboard')
        preprocessor = ImagePreprocessor()
        noisy = preprocessor.add_gaussian_noise(clean, sigma=0.15)
        
        # 创建管道
        pipeline = DenoisingPipeline(device='cpu')
        print("  ✓ 推理管道创建")
        
        # 去噪
        denoised = pipeline.denoise(noisy, normalize=False)
        assert denoised.shape == noisy.shape, "输出形状不匹配"
        assert denoised.min() >= 0 and denoised.max() <= 1, "输出范围错误"
        print("  ✓ 图像去噪")
        
        # 质量评估
        mse = np.mean((denoised - clean) ** 2)
        psnr = 10 * np.log10(1.0 / mse) if mse > 0 else 100
        print(f"  ✓ 性能指标 (PSNR: {psnr:.2f} dB)")
        
        return 3, 3
        
    except Exception as e:
        print(f"  ✗ 错误: {e}")
        traceback.print_exc()
        return 0, 3


def test_integration():
    """集成测试"""
    print("\n" + "="*60)
    print("测试7: 端到端集成")
    print("="*60)
    
    try:
        import torch
        from models import create_model
        from trainer import DenoisingTrainer
        from preprocessor import ImagePreprocessor, create_noisy_clean_pairs
        from inference import create_sample_image
        
        # 创建小数据集
        clean_images = [
            create_sample_image((64, 64), 'gradient'),
            create_sample_image((64, 64), 'circles'),
        ]
        
        dataset = create_noisy_clean_pairs(clean_images, noise_sigma=0.15, num_noise_levels=1)
        print(f"  ✓ 数据集创建 ({len(dataset)} 个样本)")
        
        # 训练多个步骤
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = create_model(device=device)
        trainer = DenoisingTrainer(model, device=device)
        
        losses = []
        for noisy, clean in dataset:
            result = trainer.train_step(noisy, clean)
            losses.append(result['loss'])
        
        print(f"  ✓ 多步骤训练 (平均损失: {np.mean(losses):.4f})")
        
        # 验证整个数据集
        val_losses = []
        for noisy, clean in dataset:
            result = trainer.validate(noisy, clean)
            val_losses.append(result['psnr'])
        
        avg_psnr = np.mean(val_losses)
        print(f"  ✓ 数据集验证 (平均PSNR: {avg_psnr:.2f})")
        
        return 3, 3
        
    except Exception as e:
        print(f"  ✗ 错误: {e}")
        traceback.print_exc()
        return 0, 3


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("深度学习图像去噪系统 - 完整测试套件")
    print("="*60)
    
    tests = [
        ("导入", test_imports),
        ("双调滤波", test_bitonic_filter),
        ("预处理", test_preprocessor),
        ("CNN模型", test_models),
        ("训练器", test_trainer),
        ("推理", test_inference),
        ("集成", test_integration),
    ]
    
    results = []
    total_passed = 0
    total_tests = 0
    
    for name, test_func in tests:
        try:
            passed, total = test_func()
            results.append((name, passed, total))
            total_passed += passed
            total_tests += total
        except Exception as e:
            print(f"\n✗ {name}测试崩溃: {e}")
            traceback.print_exc()
            results.append((name, 0, 1))
            total_tests += 1
    
    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    
    for name, passed, total in results:
        status = "✓" if passed == total else "✗"
        print(f"{status} {name}: {passed}/{total}")
    
    print("\n" + "-"*60)
    print(f"总计: {total_passed}/{total_tests} 个测试通过")
    print("-"*60)
    
    if total_passed == total_tests:
        print("\n🎉 所有测试通过！系统准备就绪。")
        return 0
    else:
        print(f"\n⚠️  {total_tests - total_passed} 个测试失败，请检查上面的错误信息。")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
