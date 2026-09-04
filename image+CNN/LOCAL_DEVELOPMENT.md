"""
本地CPU开发指南
在没有GPU的电脑上进行开发和测试
"""

# ============================================
# 本地开发完整流程
# ============================================

## 第一步：安装依赖（仅需一次）

# 打开命令行，进入项目目录
cd c:\Users\22530\Downloads\image-denoising-bitonic

# 安装依赖
pip install -r requirements.txt

# 或手动安装
pip install torch torchvision numpy opencv-python matplotlib scikit-image


## 第二步：验证系统

# 测试所有模块
python test_system.py

# 输出应该显示所有测试都通过


## 第三步：快速演示（了解系统）

# 运行演示脚本
python demo.py

# 可以查看各个演示
python demo.py --demo 1    # 双调滤波器
python demo.py --demo 2    # 模型架构
python demo.py --demo 3    # 预处理
python demo.py --demo 5    # 推理


## 第四步：本地快速训练（验证代码正确性）

# 快速训练 - 仅用来验证流程，不用等很久
python train.py \
    --epochs 5 \
    --num-images 3 \
    --image-size 128 \
    --block-size 16

# 这会在2-5分钟内完成，帮助你：
# ✓ 验证代码没有错误
# ✓ 理解训练流程
# ✓ 检查数据加载是否正常
# ✓ 确保模型可以保存和加载


## 第五步：本地推理测试

python inference.py

# 这会：
# ✓ 创建示例图像
# ✓ 添加噪声
# ✓ 进行去噪
# ✓ 保存结果到 output/ 文件夹


# ============================================
# 本地修改代码的建议
# ============================================

## 推荐的编辑顺序

### 1. 理解代码结构（不需要改）
- 阅读 README.md 和 API_REFERENCE.md
- 查看 PROJECT_STRUCTURE.py 了解架构

### 2. 尝试修改参数（推荐首先从这里开始）

# 在 train.py 中修改：
--block-size 16      # 改为16如果内存不足
--learning-rate 1e-4  # 调整学习率
--noise-sigma 0.2     # 改变噪声强度

### 3. 修改模型架构（可选）

# 编辑 models.py 中的 FeatureExtractor 或 ParameterPredictor
# 例如：改变卷积层数、激活函数等

### 4. 自定义数据（进阶）

# 在 train.py 中，修改 create_synthetic_dataset()
# 或添加自己的图像数据

### 5. 实验新的滤波参数（推荐）

# 编辑 bitonic_filter.py 中的默认参数
# 或在 demo.py 中测试不同的参数组合


# ============================================
# 本地开发的最佳实践
# ============================================

## ✅ DO - 应该做的事

1. 小规模测试
   python train.py --epochs 5 --num-images 3
   
2. 频繁保存代码
   Ctrl+S 保存文件

3. 用print调试
   print(f"Loss: {loss:.4f}")
   
4. 在Jupyter笔记本中测试小段代码
   创建本地.ipynb文件进行实验

5. 使用版本控制
   git init && git add . && git commit -m "checkpoint"


## ❌ DON'T - 不应该做的事

1. 不要训练很多个epoch
   CPU训练100个epoch会很慢（1-2小时）
   
2. 不要修改太多地方
   一次只改一个模块，确保功能正常
   
3. 不要用大图像
   本地测试用128×128或256×256就够了
   
4. 不要忘记保存检查点
   每次修改后保存到不同文件名


## 📊 CPU训练性能预期

对于本地CPU（Intel i7/Ryzen 7）：

| 任务 | 时间 | 建议 |
|------|------|------|
| 测试（5 epoch） | 2-5分钟 | ✅ 常做 |
| 快速训练（20 epoch） | 10-15分钟 | ✅ 可接受 |
| 完整训练（100 epoch） | 1-2小时 | ❌ 用Colab |


# ============================================
# 从本地迁移到Colab的检查清单
# ============================================

在上传到Colab前，确保：

□ 代码在本地能正常运行
  python test_system.py 通过

□ 训练流程可以工作
  python train.py --epochs 5 通过

□ 推理可以工作
  python inference.py 成功

□ 没有硬编码的绝对路径
  避免 C:\Users\... 这样的路径

□ 所有依赖都在 requirements.txt 中
  pip list 检查

□ 代码有足够的注释


# ============================================
# 本地和Colab的代码同步
# ============================================

## 推荐工作流

### 本地工作
1. 编辑代码文件
2. 测试修改：python train.py --epochs 5
3. 保存到版本控制

### 同步到Colab
方案A：使用Google Drive（推荐）
- 将文件夹放在Google Drive中
- Colab挂载Drive并运行

方案B：上传文件
- 手动上传到Colab
- 或使用 `files.upload()`

方案C：使用Git
- 推送到GitHub
- Colab中 git clone

## 示例：Drive同步工作流

# 本地
cd "c:\Users\22530\Downloads\image-denoising-bitonic"
# 编辑文件
python train.py --epochs 5

# 同步到Google Drive（手动拖拽或使用工具）
# Google Drive: My Drive/image-denoising-bitonic/

# Colab
from google.colab import drive
drive.mount('/content/drive')

import sys
sys.path.insert(0, '/content/drive/My Drive/image-denoising-bitonic')

from train import main
# 现在可以使用本地的代码


# ============================================
# 常见问题
# ============================================

## Q: 本地训练太慢怎么办？
A: 这是正常的。CPU训练确实慢：
   - 快速验证代码：--epochs 5
   - Colab完整训练：--epochs 100

## Q: 如何选择好的参数？
A: 在Colab训练前：
   - 本地尝试不同的--block-size
   - 本地尝试不同的--noise-sigma
   - 然后在Colab用GPU训练

## Q: Colab训练完了怎么办？
A: 模型保存到Google Drive：
   - 本地下载模型文件
   - 用本地 inference.py 进行推理

## Q: 能在本地用GPU训练吗？
A: 如果你有NVIDIA GPU：
   # 安装CUDA工具包
   # 安装CUDA版本的PyTorch
   # 然后 python train.py 会自动用GPU

   如果没有GPU，Colab是最好的选择。


# ============================================
# 快速参考命令
# ============================================

# 安装
pip install -r requirements.txt

# 测试
python test_system.py

# 演示
python demo.py --demo 5

# 快速训练（本地）
python train.py --epochs 5 --num-images 3

# 推理
python inference.py

# 查看帮助
python train.py --help
python demo.py --help

# 清理缓存（可选）
rm -r __pycache__
rm -r .pytest_cache


# ============================================
# 总结：最实用的本地工作流
# ============================================

# 早上：快速测试新代码
python train.py --epochs 5 --num-images 3

# 中午：完整的Colab训练
# （上传代码到Colab，运行100个epoch）

# 下午：下载模型，本地推理测试
python inference.py

# 晚上：分析结果，修改参数准备第二天的训练

# 重复这个流程，不断优化！


"""

提示：
1. 本地用来开发和测试逻辑
2. Colab用来完整训练
3. 将结果下载到本地使用
"""
