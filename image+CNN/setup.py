"""
系统初始化脚本
检查依赖并创建必要的目录
"""

import subprocess
import sys
from pathlib import Path


def install_requirements():
    """安装所需的Python包"""
    
    requirements = [
        'torch',
        'torchvision',
        'numpy',
        'opencv-python',
        'matplotlib',
        'pillow',
        'scikit-image'
    ]
    
    print("安装依赖包...")
    
    for package in requirements:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package} 已安装")
        except ImportError:
            print(f"✗ {package} 未安装，正在安装...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} 安装完成")
    
    print("\n所有依赖已检查和安装！")


def create_directories():
    """创建必要的目录"""
    directories = [
        'checkpoints',
        'output',
        'data/images',
        'logs'
    ]
    
    print("\n创建必要的目录...")
    
    for dir_name in directories:
        path = Path(dir_name)
        path.mkdir(parents=True, exist_ok=True)
        print(f"✓ {dir_name}")
    
    print("\n目录结构已创建！")


def main():
    print("="*50)
    print("图像去噪系统初始化")
    print("="*50)
    
    install_requirements()
    create_directories()
    
    print("\n" + "="*50)
    print("初始化完成！")
    print("="*50)
    print("\n接下来可以运行:")
    print("  1. 训练: python train.py")
    print("  2. 推理: python inference.py")
    print("  3. 演示: python demo.py")


if __name__ == "__main__":
    main()
