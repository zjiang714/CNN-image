"""
完整的图像去噪系统演示和推理
"""

import numpy as np
import torch
import cv2
from pathlib import Path
from typing import Tuple, Optional
import matplotlib.pyplot as plt

from models import create_model
from bitonic_filter import BitonicFilter
from preprocessor import ImagePreprocessor
from trainer import DenoisingTrainer


class DenoisingPipeline:
    """
    完整的去噪流程管道
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu',
                 block_size: int = 32):
        """
        初始化去噪管道
        
        Args:
            model_path: 预训练模型路径
            device: 计算设备
            block_size: 图像块大小
        """
        self.device = device
        self.block_size = block_size
        self.preprocessor = ImagePreprocessor(block_size=block_size)
        self.bitonic_filter = BitonicFilter(kernel_size=3)
        
        # 创建模型
        self.model = create_model(in_channels=1, num_features=64, num_params=8, device=device)
        
        # 加载预训练权重
        if model_path and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state'])
            else:
                self.model.load_state_dict(checkpoint)
        
        self.model.eval()
    
    @torch.no_grad()
    def denoise(self, image: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        对图像进行去噪
        
        Args:
            image: 输入图像
            normalize: 是否进行归一化
            
        Returns:
            去噪后的图像
        """
        # 预处理
        if normalize:
            image = self.preprocessor.normalize(image)
        
        # 转为灰度图（如果需要）
        if len(image.shape) == 3:
            image = self.preprocessor.to_grayscale(image)
        
        # 填充图像
        padded_image, (pad_h, pad_w) = self.preprocessor.pad_image(image)
        
        # 分块
        blocks, positions = self.preprocessor.split_into_blocks(padded_image)
        denoised_blocks = []
        
        for block in blocks:
            # 转换为张量
            tensor = torch.FloatTensor(block[np.newaxis, np.newaxis, :, :]).to(self.device)
            
            # 预测参数
            features, params = self.model(tensor)
            params_np = params.cpu().numpy()[0]
            
            # 应用双调滤波
            denoised_block = self.bitonic_filter.apply(block, params_np)
            denoised_blocks.append(denoised_block)
        
        # 合并块
        denoised_image = self.preprocessor.merge_blocks(denoised_blocks, positions,
                                                        padded_image.shape[:2])
        
        # 移除填充
        denoised_image = self.preprocessor.unpad_image(denoised_image, pad_h, pad_w)
        
        # 反归一化
        if normalize:
            denoised_image = self.preprocessor.denormalize(denoised_image)
        
        return denoised_image
    
    def process_image_file(self, input_path: str, output_path: Optional[str] = None) -> np.ndarray:
        """
        处理图像文件
        
        Args:
            input_path: 输入图像路径
            output_path: 输出图像路径（可选）
            
        Returns:
            去噪后的图像
        """
        # 读取图像
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"无法读取图像: {input_path}")
        
        # 转为RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 去噪
        denoised = self.denoise(image, normalize=True)
        
        # 保存输出
        if output_path:
            denoised_bgr = cv2.cvtColor(denoised, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, denoised_bgr)
            print(f"去噪图像已保存到: {output_path}")
        
        return denoised


def create_sample_image(size: Tuple[int, int] = (256, 256), 
                       pattern: str = 'gradient') -> np.ndarray:
    """
    创建示例图像
    
    Args:
        size: 图像大小 (height, width)
        pattern: 图案类型 ('gradient', 'checkerboard', 'circles')
        
    Returns:
        示例图像
    """
    h, w = size
    
    if pattern == 'gradient':
        # 渐变图案
        image = np.zeros((h, w), dtype=np.float32)
        for i in range(h):
            image[i, :] = i / h
    
    elif pattern == 'checkerboard':
        # 棋盘图案
        square_size = 16
        image = np.zeros((h, w), dtype=np.float32)
        for i in range(0, h, square_size):
            for j in range(0, w, square_size):
                if ((i // square_size) + (j // square_size)) % 2 == 0:
                    image[i:i+square_size, j:j+square_size] = 1.0
    
    elif pattern == 'circles':
        # 圆形图案
        image = np.zeros((h, w), dtype=np.float32)
        center = (w // 2, h // 2)
        for i in range(h):
            for j in range(w):
                dist = np.sqrt((i - center[1])**2 + (j - center[0])**2)
                image[i, j] = np.exp(-dist / 50)
    
    else:
        image = np.random.rand(h, w).astype(np.float32)
    
    return np.clip(image, 0, 1)


def visualize_results(noisy: np.ndarray, denoised: np.ndarray, clean: np.ndarray,
                     title: str = "去噪结果") -> None:
    """
    可视化去噪结果
    
    Args:
        noisy: 含噪图像
        denoised: 去噪后的图像
        clean: 干净图像
        title: 标题
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(noisy, cmap='gray')
    axes[0].set_title('含噪图像')
    axes[0].axis('off')
    
    axes[1].imshow(denoised, cmap='gray')
    axes[1].set_title('去噪后')
    axes[1].axis('off')
    
    axes[2].imshow(clean, cmap='gray')
    axes[2].set_title('干净图像')
    axes[2].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 创建输出目录
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # 初始化管道
    pipeline = DenoisingPipeline(device='cpu', block_size=32)
    
    # 创建示例图像
    print("创建示例图像...")
    clean_image = create_sample_image(size=(256, 256), pattern='checkerboard')
    
    # 添加高斯噪声
    preprocessor = ImagePreprocessor()
    noisy_image = preprocessor.add_gaussian_noise(clean_image, sigma=0.15)
    
    # 去噪
    print("进行去噪...")
    denoised_image = pipeline.denoise(noisy_image, normalize=False)
    
    # 计算指标
    psnr_noisy = np.mean((noisy_image - clean_image) ** 2)
    psnr_noisy = 10 * np.log10(1.0 / psnr_noisy) if psnr_noisy > 0 else 100
    
    psnr_denoised = np.mean((denoised_image - clean_image) ** 2)
    psnr_denoised = 10 * np.log10(1.0 / psnr_denoised) if psnr_denoised > 0 else 100
    
    print(f"含噪图像 PSNR: {psnr_noisy:.2f}")
    print(f"去噪图像 PSNR: {psnr_denoised:.2f}")
    print(f"改进: {psnr_denoised - psnr_noisy:.2f} dB")
    
    # 保存结果
    cv2.imwrite(str(output_dir / "clean.png"), (clean_image * 255).astype(np.uint8))
    cv2.imwrite(str(output_dir / "noisy.png"), (noisy_image * 255).astype(np.uint8))
    cv2.imwrite(str(output_dir / "denoised.png"), (denoised_image * 255).astype(np.uint8))
    
    print(f"结果已保存到 {output_dir}/")
