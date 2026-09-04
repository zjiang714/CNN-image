# 快速开始指南

## 5分钟快速体验

### 步骤1：安装依赖

```bash
# Windows
python setup.py

# 或手动安装
pip install -r requirements.txt
```

### 步骤2：运行演示

```bash
# 运行所有演示
python demo.py

# 或运行单个演示
python demo.py --demo 1  # 双调滤波器
python demo.py --demo 2  # 模型架构
python demo.py --demo 3  # 预处理
python demo.py --demo 4  # 训练（需要时间）
python demo.py --demo 5  # 推理
python demo.py --demo 6  # 参数分析
```

### 步骤3：训练模型

```bash
# 快速训练（小规模，2-3分钟）
python train.py --epochs 10 --num-images 5

# 完整训练（15-30分钟）
python train.py --epochs 100 --num-images 20
```

### 步骤4：进行推理

```bash
python inference.py
```

## 核心概念速览

### 双调滤波器 (Bitonic Filter)
- **Alpha (α)**: 平滑度参数（0-1）
  - 低值(0.3)：保留更多细节
  - 高值(0.7)：更强平滑
  
- **Beta (β)**: 边界保留参数（0-1）
  - 低值(0.3)：强力去噪
  - 高值(0.8)：保留边界细节

### CNN网络
- **特征提取器**: 从32×32图像块提取64维特征
- **参数预测器**: 根据特征预测最优的α和β

### 训练指标
- **PSNR**: 峰值信噪比（dB单位）
  - > 30dB：优秀
  - 25-30dB：良好
  
- **SSIM**: 结构相似度（0-1）
  - > 0.9：优秀
  - 0.8-0.9：良好

## 常见问题

### Q: 模型保存在哪里？
A: 训练完成后保存在 `checkpoints/` 目录：
- `best_model.pth`: 最佳模型（验证集PSNR最高）
- `final_model.pth`: 最终模型
- `history.json`: 训练历史曲线数据

### Q: 如何处理自己的图像？
A: 修改 `inference.py` 中的路径：
```python
pipeline = DenoisingPipeline(model_path='checkpoints/best_model.pth')
denoised = pipeline.process_image_file('my_image.jpg', 'output.jpg')
```

### Q: 可以用GPU加速吗？
A: 可以，训练时自动检测GPU，推理时：
```python
pipeline = DenoisingPipeline(device='cuda')  # GPU
```

### Q: 为什么PSNR不是单调递增的？
A: 正常现象，原因包括：
- 验证集样本差异
- 过拟合的早期迹象
- 学习率过大导致震荡
解决：
- 减小学习率
- 增加正则化项
- 数据增强

## 模型输出示例

```
Training Progress:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Epoch 10/100
  Train Loss: 0.0032, PSNR: 28.15, SSIM: 0.8234
  Val Loss: 0.0041, PSNR: 27.92, SSIM: 0.8156

Epoch 20/100
  Train Loss: 0.0021, PSNR: 29.87, SSIM: 0.8567
  Val Loss: 0.0027, PSNR: 29.43, SSIM: 0.8456

...

Epoch 100/100
  Train Loss: 0.0012, PSNR: 31.52, SSIM: 0.8923
  Val Loss: 0.0015, PSNR: 31.18, SSIM: 0.8801

==================================================
训练完成！
==================================================
最终 PSNR: 31.18
最终 SSIM: 0.8801
最终验证损失: 0.0015
```

## 项目文件说明

| 文件 | 用途 | 重要性 |
|------|------|--------|
| bitonic_filter.py | 双调滤波器核心实现 | ⭐⭐⭐⭐⭐ |
| models.py | CNN模型定义 | ⭐⭐⭐⭐ |
| preprocessor.py | 图像处理工具 | ⭐⭐⭐⭐ |
| trainer.py | 训练逻辑 | ⭐⭐⭐⭐ |
| inference.py | 推理管道 | ⭐⭐⭐⭐ |
| train.py | 训练脚本入口 | ⭐⭐⭐ |
| demo.py | 演示脚本 | ⭐⭐⭐ |
| setup.py | 环境配置 | ⭐⭐ |

## 下一步

1. **了解更多**: 阅读 `README.md`
2. **查看源码**: 阅读各模块的详细注释
3. **自定义模型**: 修改 `models.py` 中的网络架构
4. **集成应用**: 将 `DenoisingPipeline` 集成到你的应用

## 性能对标

| 场景 | 提升 | 耗时 |
|------|------|------|
| 轻噪声(σ=0.1) | +2-3 dB | ~50ms |
| 中等噪声(σ=0.15) | +3-5 dB | ~80ms |
| 重噪声(σ=0.2) | +4-6 dB | ~100ms |

*测试环境：Intel i7, 256×256图像*

## 许可和引用

本项目展示了深度学习与经典滤波算法的融合。如用于研究，请引用本项目。

---

**更新日期**: 2026年1月  
**当前版本**: 1.0
