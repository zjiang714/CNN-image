# 使用指南和API参考

## 目录
1. [快速入门](#快速入门)
2. [API参考](#api参考)
3. [完整示例](#完整示例)
4. [常见用法](#常见用法)
5. [故障排除](#故障排除)

---

## 快速入门

### 最简单的方式 - 3行代码去噪

```python
from inference import DenoisingPipeline
import cv2

# 初始化
pipeline = DenoisingPipeline(device='cpu')  # 自动创建模型

# 处理图像
denoised = pipeline.denoise(cv2.imread('noisy.jpg'))

# 保存
cv2.imwrite('output.jpg', denoised)
```

### 训练自己的模型

```python
# 仅需5行命令
python setup.py              # 安装依赖
python train.py              # 训练模型
python inference.py          # 使用模型
```

---

## API参考

### 1. BitonicFilter - 双调滤波器

```python
from bitonic_filter import BitonicFilter
import numpy as np

# 初始化（参数范围都是0-1）
filter = BitonicFilter(
    kernel_size=3,      # 滤波核大小
    alpha=0.5,          # 平滑度参数
    beta=0.5            # 边界保留参数
)

# 应用滤波
image = np.random.rand(256, 256)  # 输入图像 [0, 1]
filtered = filter.apply(image)     # 去噪

# 动态参数
params = np.array([0.6, 0.7])      # [alpha, beta]
filtered = filter.apply(image, params)
```

**参数说明**：
- `alpha` (0.0-1.0)：
  - 0.0：完全使用均值平滑
  - 0.5：均值和中值混合
  - 1.0：完全使用中值
  
- `beta` (0.0-1.0)：
  - 0.0：完全替换为滤波值
  - 0.5：50%保留原值
  - 1.0：完全保留原值

### 2. ImagePreprocessor - 图像预处理

```python
from preprocessor import ImagePreprocessor
import cv2

preprocessor = ImagePreprocessor(
    block_size=32,      # 块大小
    overlap=0           # 块重叠像素
)

# 读取图像
image = cv2.imread('image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 归一化
normalized = preprocessor.normalize(image)

# 转灰度
gray = preprocessor.to_grayscale(image)

# 分块
blocks, positions = preprocessor.split_into_blocks(image)
# blocks: [block1, block2, ...] 每个是32x32
# positions: [(0,0), (0,32), ...] 块的位置

# 合并
merged = preprocessor.merge_blocks(blocks, positions, image.shape[:2])

# 添加噪声
noisy = preprocessor.add_gaussian_noise(image, sigma=0.15)

# 填充处理（用于分块）
padded, (pad_h, pad_w) = preprocessor.pad_image(image)
unpadded = preprocessor.unpad_image(padded, pad_h, pad_w)
```

**常用组合**：

```python
# 完整预处理管道
image = cv2.imread('image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

preprocessor = ImagePreprocessor(block_size=32)

# 流程
image = preprocessor.normalize(image)      # RGB to [0,1]
image = preprocessor.to_grayscale(image)   # RGB to Gray
image, pad = preprocessor.pad_image(image) # 填充
blocks, pos = preprocessor.split_into_blocks(image)  # 分块

# ... 处理块 ...

merged = preprocessor.merge_blocks(blocks, pos, image.shape[:2])
image = preprocessor.unpad_image(merged, pad[0], pad[1])
image = preprocessor.denormalize(image)    # [0,1] to [0,255]
```

### 3. DenoisingNetwork - CNN模型

```python
from models import create_model
import torch

# 创建模型
model = create_model(
    in_channels=1,      # 灰度图
    num_features=64,    # 特征维度
    num_params=2,       # 输出参数个数
    device='cuda'       # 计算设备
)

# 前向传播
input_tensor = torch.randn(1, 1, 32, 32)  # (B, C, H, W)
features, params = model(input_tensor)

# features: (1, 64) - 特征向量
# params: (1, 2) - [alpha, beta]，范围[0,1]

# 保存和加载
torch.save(model.state_dict(), 'model.pth')
model.load_state_dict(torch.load('model.pth'))
```

### 4. DenoisingTrainer - 训练器

```python
from trainer import DenoisingTrainer
from models import create_model

model = create_model(device='cuda')
trainer = DenoisingTrainer(
    model=model,
    device='cuda',
    learning_rate=1e-3,
    block_size=32
)

# 训练一个epoch
result = trainer.train_step(noisy_image, clean_image)
# result['loss']: 损失值
# result['psnr']: 峰值信噪比
# result['ssim']: 结构相似度
# result['denoised_image']: 去噪后的图像

# 验证
result = trainer.validate(noisy_image, clean_image)

# 完整训练
history = trainer.fit(
    train_data=[...],   # 列表of (noisy, clean) 对
    val_data=[...],
    epochs=100,
    save_dir='checkpoints'
)

# 保存/加载
trainer.save_model('model.pth')
trainer.load_model('model.pth')
```

### 5. DenoisingPipeline - 推理管道

```python
from inference import DenoisingPipeline
import numpy as np

# 初始化（最简单）
pipeline = DenoisingPipeline()

# 或加载预训练模型
pipeline = DenoisingPipeline(
    model_path='checkpoints/best_model.pth',
    device='cuda'
)

# 推理
noisy_image = np.random.rand(256, 256)  # [0,1]
denoised = pipeline.denoise(noisy_image, normalize=False)

# 或处理图像文件
denoised = pipeline.process_image_file('input.jpg', 'output.jpg')
```

---

## 完整示例

### 示例1：基础去噪

```python
import cv2
import numpy as np
from inference import DenoisingPipeline
from preprocessor import ImagePreprocessor

# 读取图像
image = cv2.imread('clean.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image.astype(np.float32) / 255.0

# 添加噪声
preprocessor = ImagePreprocessor()
noisy = preprocessor.add_gaussian_noise(image, sigma=0.1)

# 去噪
pipeline = DenoisingPipeline(device='cuda')
denoised = pipeline.denoise(noisy, normalize=False)

# 计算PSNR
mse = np.mean((denoised - image) ** 2)
psnr = 10 * np.log10(1.0 / mse)
print(f"PSNR: {psnr:.2f} dB")

# 保存
output = cv2.cvtColor((denoised * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
cv2.imwrite('denoised.jpg', output)
```

### 示例2：自定义参数的滤波

```python
from bitonic_filter import BitonicFilter
import numpy as np

# 创建测试图像
image = np.random.rand(256, 256) + np.linspace(0, 1, 256)[:, np.newaxis]
image = np.clip(image, 0, 1)

# 创建滤波器
bf = BitonicFilter()

# 尝试不同的参数组合
best_psnr = 0
best_params = None

for alpha in np.linspace(0.2, 0.8, 7):
    for beta in np.linspace(0.3, 0.9, 7):
        filtered = bf.apply(image, np.array([alpha, beta]))
        # 计算质量指标...
        if psnr > best_psnr:
            best_psnr = psnr
            best_params = (alpha, beta)

print(f"最优参数: α={best_params[0]:.2f}, β={best_params[1]:.2f}")
```

### 示例3：完整的训练流程

```python
import torch
from models import create_model
from trainer import DenoisingTrainer
from preprocessor import ImagePreprocessor, create_noisy_clean_pairs
from inference import create_sample_image

# 1. 创建数据集
print("创建数据集...")
clean_images = [
    create_sample_image((256, 256), pattern='gradient'),
    create_sample_image((256, 256), pattern='checkerboard'),
]

train_data = create_noisy_clean_pairs(clean_images, noise_sigma=0.15)
val_data = train_data[-2:]  # 最后2个作为验证集
train_data = train_data[:-2]

# 2. 创建模型
print("创建模型...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = create_model(device=device)

# 3. 创建训练器
print("创建训练器...")
trainer = DenoisingTrainer(model, device=device, learning_rate=1e-3)

# 4. 训练
print("训练...")
history = trainer.fit(
    train_data=train_data,
    val_data=val_data,
    epochs=100,
    save_dir='checkpoints'
)

# 5. 查看结果
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(history['train_loss'], label='Train')
plt.plot(history['val_loss'], label='Val')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.title('Loss Curve')

plt.subplot(1, 3, 2)
plt.plot(history['psnr'])
plt.ylabel('PSNR (dB)')
plt.xlabel('Epoch')
plt.title('PSNR')

plt.subplot(1, 3, 3)
plt.plot(history['ssim'])
plt.ylabel('SSIM')
plt.xlabel('Epoch')
plt.title('SSIM')

plt.tight_layout()
plt.show()
```

### 示例4：批量处理图像

```python
from pathlib import Path
from inference import DenoisingPipeline
import cv2

# 初始化管道
pipeline = DenoisingPipeline(model_path='checkpoints/best_model.pth')

# 处理目录中的所有图像
input_dir = Path('input_images')
output_dir = Path('output_images')
output_dir.mkdir(exist_ok=True)

for image_path in input_dir.glob('*.jpg'):
    print(f"处理: {image_path.name}")
    
    output_path = output_dir / f"denoised_{image_path.name}"
    pipeline.process_image_file(str(image_path), str(output_path))

print("完成！")
```

---

## 常见用法

### 用法1：调整去噪强度

```python
from bitonic_filter import BitonicFilter

# 轻度去噪（保留细节）
bf = BitonicFilter(alpha=0.4, beta=0.8)

# 中度去噪（平衡）
bf = BitonicFilter(alpha=0.5, beta=0.5)

# 强度去噪（平滑）
bf = BitonicFilter(alpha=0.7, beta=0.3)
```

### 用法2：自定义模型架构

```python
import torch
import torch.nn as nn
from models import DenoisingNetwork

class CustomDenoisingNetwork(DenoisingNetwork):
    def __init__(self):
        super().__init__(in_channels=1)
        # 自定义修改
        self.feature_extractor = MyFeatureExtractor()
        self.parameter_predictor = MyPredictor()

model = CustomDenoisingNetwork()
```

### 用法3：监控训练进度

```python
from trainer import DenoisingTrainer

class MonitoredTrainer(DenoisingTrainer):
    def train_step(self, noisy, clean):
        result = super().train_step(noisy, clean)
        
        # 自定义监控
        print(f"Loss: {result['loss']:.4f}")
        print(f"PSNR: {result['psnr']:.2f}")
        
        return result

trainer = MonitoredTrainer(model)
```

### 用法4：在CPU和GPU间切换

```python
import torch
from inference import DenoisingPipeline

# 自动检测
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pipeline = DenoisingPipeline(device=device)

# 手动指定
pipeline = DenoisingPipeline(device='cpu')    # 只用CPU
pipeline = DenoisingPipeline(device='cuda')   # 只用GPU
```

---

## 故障排除

### 问题1: "ModuleNotFoundError: No module named 'torch'"

**解决**:
```bash
pip install torch torchvision
# 或使用GPU版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 问题2: CUDA内存不足

**解决**:
```python
# 方案1：使用CPU
pipeline = DenoisingPipeline(device='cpu')

# 方案2：减小块大小
preprocessor = ImagePreprocessor(block_size=16)

# 方案3：减小模型
model = create_model(num_features=32)
```

### 问题3: 去噪效果不好

**诊断**:
```python
# 检查模型是否有权重
model = create_model()
params = sum(p.numel() for p in model.parameters())
print(f"模型参数: {params}")

# 检查数据范围
print(f"图像范围: [{image.min()}, {image.max()}]")
# 应该是 [0, 1] 或 [0, 255]

# 检查噪声水平
print(f"噪声等级: {np.std(noisy - clean)}")
```

**改进**:
- 训练更多epoch: `python train.py --epochs 200`
- 调整学习率: `python train.py --learning-rate 5e-4`
- 增加数据: `python train.py --num-images 30`

### 问题4: 训练不收敛

**调试**:
```python
# 打印第一个batch的损失变化
trainer = DenoisingTrainer(model)
for i in range(10):
    result = trainer.train_step(train_data[0][0], train_data[0][1])
    print(f"Step {i}: Loss={result['loss']:.6f}")
```

**解决**:
- 减小学习率 (1e-3 → 1e-4)
- 增加正则化权重
- 检查数据质量

---

**更新日期**: 2026年1月  
**版本**: 1.0  
**维护者**: AI Assistant
