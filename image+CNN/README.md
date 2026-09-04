# 深度学习图像去噪系统 - 双调滤波器与CNN融合

## 概述

本项目实现了一个完整的深度学习图像去噪系统，结合了**CNN参数预测**和**双调滤波器(Bitonic Filter)**技术。核心思想是：

1. **图像预处理与分块**：对输入图像进行预处理和分块处理
2. **特征提取**：使用CNN从每个图像块提取特征
3. **参数预测**：根据提取的特征预测双调滤波器参数
4. **自适应滤波**：对每个图像块应用参数化的双调滤波器进行去噪
5. **优化训练**：通过对比含噪和干净图像的差异来优化网络参数

## 系统架构

```
输入图像
   ↓
预处理 & 分块
   ↓
┌─────────────────────┐
│ 对每个图像块:         │
│  1. CNN特征提取       │
│  2. 参数预测         │
│  3. 双调滤波        │
└─────────────────────┘
   ↓
块合并
   ↓
去噪后的图像
```

## 核心模块说明

### 1. `bitonic_filter.py` - 双调滤波器

**BitonicFilter 类**：
- 实现双调排序网络算法
- 参数：
  - `alpha` (0-1)：控制平滑度，值越大越倾向于使用中值
  - `beta` (0-1)：控制边界保留，值越大越保留原始像素
- `apply()`：对图像块应用双调滤波
- `_bitonic_sort()`：双调排序实现

**关键特点**：
- 自适应滤波参数
- 基于排序网络的中值滤波变体
- 平衡去噪强度和边界保留

### 2. `models.py` - 深度学习模型

**FeatureExtractor 类**：
```
输入 (1, 1, H, W)
  ↓
Conv2d(1→32) + BatchNorm + ReLU + MaxPool
  ↓
Conv2d(32→64) + BatchNorm + ReLU + MaxPool
  ↓
Conv2d(64→128) + BatchNorm + ReLU
  ↓
全局平均池化
  ↓
线性层 → 特征向量 (1, num_features)
```

**ParameterPredictor 类**：
```
特征向量 (1, 64)
  ↓
FC(64→128) + BatchNorm + ReLU
  ↓
FC(128→64) + BatchNorm + ReLU
  ↓
FC(64→32) + BatchNorm + ReLU
  ↓
FC(32→2) + Sigmoid → [alpha, beta]
```

**DenoisingNetwork**：完整网络，结合特征提取和参数预测

### 3. `preprocessor.py` - 预处理模块

**ImagePreprocessor 类**：
- `normalize()`：将图像归一化到 [0, 1]
- `denormalize()`：反归一化到 [0, 255]
- `to_grayscale()`：RGB转灰度
- `split_into_blocks()`：将图像分块
- `merge_blocks()`：合并图像块（支持重叠混合）
- `pad_image()`：填充图像
- `add_gaussian_noise()`：添加高斯噪声

### 4. `trainer.py` - 训练模块

**DenoisingTrainer 类**：
- 端到端的训练流程
- 损失函数：
  - 重构损失 (MSE)：去噪图像 vs 干净图像
  - 参数正则化：鼓励参数在合理范围
  
- 性能指标：
  - **PSNR (Peak Signal-to-Noise Ratio)**：越高越好，>30dB为良好
  - **SSIM (Structural Similarity Index)**：考虑结构相似性，范围[0,1]

- 训练特性：
  - Adam优化器
  - 学习率调度
  - 自动保存最佳模型

### 5. `inference.py` - 推理和演示

**DenoisingPipeline 类**：
- 加载预训练模型
- `denoise()`：对图像进行去噪
- `process_image_file()`：处理本地图像文件

## 使用说明

### 第一步：安装依赖

```bash
# 方法1：使用setup脚本
python setup.py

# 方法2：手动安装
pip install -r requirements.txt
```

### 第二步：训练模型

```bash
# 基本训练
python train.py

# 自定义参数
python train.py \
  --epochs 200 \
  --learning-rate 0.001 \
  --num-images 20 \
  --image-size 256 \
  --noise-sigma 0.15 \
  --save-dir checkpoints
```

**在 Kaggle 使用已下载的 DIV2K 数据集**：

```bash
python train.py \
  --dataset div2k_train \
  --data-dir /kaggle/input/div2k-dataset/DIV2K_train_HR \
  --num-images 50 \
  --epochs 50 \
  --use-cuda
```

> 说明：`--data-dir` 可以直接指向已下载的 `DIV2K_train_HR` 或 `DIV2K_valid_HR` 目录，
> 这样不会尝试下载或创建目录，适用于 Kaggle 的只读数据路径。

**训练参数说明**：
- `--epochs`：训练轮次（推荐100-200）
- `--learning-rate`：学习率（推荐1e-3至1e-4）
- `--block-size`：图像块大小（推荐32）
- `--num-features`：CNN特征维度（推荐64）
- `--num-images`：合成训练图像数（推荐10-20）
- `--image-size`：图像大小（推荐256）
- `--noise-sigma`：噪声标准差（推荐0.1-0.2）

### 第三步：进行推理

```bash
# 使用演示脚本
python inference.py

# 在代码中使用
from inference import DenoisingPipeline
import cv2

# 初始化管道
pipeline = DenoisingPipeline(
    model_path='checkpoints/best_model.pth',
    device='cuda'  # 或 'cpu'
)

# 读取和处理图像
image = cv2.imread('input.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
denoised = pipeline.denoise(image, normalize=True)

# 保存结果
cv2.imwrite('output.jpg', cv2.cvtColor(denoised, cv2.COLOR_RGB2BGR))
```

## 数据流详解

### 训练流程

```
含噪图像
  ↓
预处理 + 分块
  ↓ [32x32块]
对每个块：
  ├→ 张量转换
  ├→ CNN特征提取
  ├→ 参数预测 [α, β]
  ├→ 双调滤波
  └→ 计算损失
  
损失函数：
  L = MSE(滤波后, 干净) + λ·正则化(参数)
  
反向传播 + 优化器更新
  ↓
块合并
  ↓
计算PSNR和SSIM
  ↓ 重复直到收敛
最佳模型保存
```

### 推理流程

```
输入图像
  ↓
灰度转换 + 归一化
  ↓
填充处理
  ↓
分块
  ↓ [对每个块]
  ├→ CNN前向传播
  ├→ 预测α和β
  ├→ 双调滤波
  └→ 收集结果
  ↓
块合并（重叠混合）
  ↓
移除填充
  ↓
反归一化
  ↓
输出图像
```

## 参数调优建议

### CNN参数
- **num_features**: 特征维度，64-128较好平衡性能和速度
- **kernel_size**: 卷积核大小，默认3效果佳

### 滤波器参数
- **alpha**: 0.3-0.7，值越大越平滑（中值倾向）
- **beta**: 0.4-0.8，值越小越强力去噪，越大越保留细节
- **kernel_size**: 3-5，影响邻域范围

### 训练参数
- **学习率**: 1e-3到1e-4之间逐步调整
- **batch_size**: 现在支持在线学习（批次大小1），可扩展
- **epochs**: 初期100-150观察收敛，可根据损失曲线调整
- **noise_sigma**: 0.1-0.2模拟真实噪声

## 性能指标解读

### PSNR (分贝)
- < 20 dB：严重失真
- 20-25 dB：明显失真
- 25-35 dB：良好质量
- > 35 dB：优秀质量

### SSIM (0-1)
- < 0.6：质量差
- 0.6-0.8：中等质量
- > 0.8：好质量
- > 0.95：优秀质量

## 文件结构

```
image-denoising-bitonic/
├── bitonic_filter.py    # 双调滤波器实现
├── models.py            # CNN模型定义
├── preprocessor.py      # 图像预处理和分块
├── trainer.py           # 训练器
├── inference.py         # 推理管道和演示
├── train.py             # 训练脚本
├── setup.py             # 系统初始化
├── requirements.txt     # 依赖列表
├── README.md            # 本文件
├── checkpoints/         # 模型保存目录
│   ├── best_model.pth
│   ├── final_model.pth
│   └── history.json
├── output/              # 推理结果目录
├── data/images/         # 输入数据目录
└── logs/                # 日志目录
```

## 扩展建议

### 1. 多通道支持
修改 `in_channels=3` 来支持彩色图像处理

### 2. 批量处理
在 `trainer.py` 中实现 DataLoader 支持批量训练

### 3. 不同噪声类型
扩展 `add_gaussian_noise()` 支持泊松噪声、椒盐噪声等

### 4. 实时去噪
使用预先计算的参数表或量化模型部署到移动设备

### 5. 视频去噪
利用时间相关性处理视频帧序列

## 已知限制

1. **单通道处理**：当前主要面向灰度图，可扩展为RGB
2. **块大小固定**：32x32块可根据内存调整
3. **内存占用**：大图像需要GPU加速
4. **合成数据**：演示使用合成数据，实际应用需要真实数据

## 性能基准

在GPU上的运行时间估计（256×256图像）：
- 推理：~50-100ms
- 训练一轮（10张图像）：~5-10秒
- 完整训练（100轮）：~10-15分钟

## 许可证

开源项目，自由使用和修改

## 联系和反馈

如有问题或改进建议，欢迎反馈！

---

**最后更新**: 2026年1月
**版本**: 1.0
