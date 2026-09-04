"""
图像预处理和分块模块
"""

import numpy as np
from typing import Tuple, List
import cv2


class ImagePreprocessor:
    """
    图像预处理和分块
    """
    
    def __init__(self, block_size: int = 32, overlap: int = 0):
        """
        初始化预处理器
        
        Args:
            block_size: 分块大小
            overlap: 块之间的重叠像素数
        """
        self.block_size = block_size
        self.overlap = overlap
        self.stride = block_size - overlap
        
    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        图像归一化到 [0, 1]
        
        Args:
            image: 输入图像
            
        Returns:
            归一化后的图像
        """
        if image.dtype == np.uint8:
            return image.astype(np.float32) / 255.0
        elif image.dtype in [np.float32, np.float64]:
            if image.max() > 1.0:
                return image / 255.0
            return image.astype(np.float32)
        return image
    
    def denormalize(self, image: np.ndarray) -> np.ndarray:
        """
        图像反归一化到 [0, 255]
        
        Args:
            image: 归一化的图像
            
        Returns:
            反归一化后的图像
        """
        image = np.clip(image, 0, 1)
        return (image * 255).astype(np.uint8)
    
    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        转换为灰度图
        
        Args:
            image: 输入图像
            
        Returns:
            灰度图
        """
        if len(image.shape) == 3:
            return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        return image
    
    def resize(self, image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """
        调整图像大小
        
        Args:
            image: 输入图像
            size: 目标大小 (height, width)
            
        Returns:
            调整大小后的图像
        """
        return cv2.resize(image, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
    
    def split_into_blocks(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """
        将图像分块
        
        Args:
            image: 输入图像 (H, W) 或 (H, W, C)
            
        Returns:
            (块列表, 块位置列表)
            块位置格式: (top_left_row, top_left_col)
        """
        h, w = image.shape[:2]
        blocks = []
        positions = []
        
        for i in range(0, h - self.block_size + 1, self.stride):
            for j in range(0, w - self.block_size + 1, self.stride):
                block = image[i:i+self.block_size, j:j+self.block_size]
                blocks.append(block)
                positions.append((i, j))
        
        return blocks, positions
    
    def merge_blocks(self, blocks: List[np.ndarray], positions: List[Tuple[int, int]],
                    output_shape: Tuple[int, int], use_blending: bool = True) -> np.ndarray:
        """
        将块合并回图像
        
        Args:
            blocks: 块列表
            positions: 块位置列表
            output_shape: 输出图像形状 (H, W)
            use_blending: 是否在重叠区域使用混合
            
        Returns:
            合并后的图像
        """
        h, w = output_shape
        output = np.zeros((h, w), dtype=np.float32)
        count = np.zeros((h, w), dtype=np.float32)
        
        for block, (i, j) in zip(blocks, positions):
            block_h, block_w = block.shape[:2]
            output[i:i+block_h, j:j+block_w] += block
            count[i:i+block_h, j:j+block_w] += 1
        
        # 避免除以零
        count[count == 0] = 1
        output = output / count
        
        return output
    
    def add_gaussian_noise(self, image: np.ndarray, sigma: float = 0.1) -> np.ndarray:
        """
        添加高斯噪声
        
        Args:
            image: 输入图像 (已归一化到 [0, 1])
            sigma: 噪声标准差
            
        Returns:
            含噪图像
        """
        noise = np.random.normal(0, sigma, image.shape)
        noisy_image = image + noise
        return np.clip(noisy_image, 0, 1)
    
    def pad_image(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        填充图像使其能被stride整除
        
        Args:
            image: 输入图像
            
        Returns:
            (填充后的图像, (pad_h, pad_w))
        """
        h, w = image.shape[:2]
        pad_h = (self.stride - (h - self.block_size) % self.stride) % self.stride
        pad_w = (self.stride - (w - self.block_size) % self.stride) % self.stride
        
        if pad_h > 0 or pad_w > 0:
            if len(image.shape) == 3:
                image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            else:
                image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
        
        return image, (pad_h, pad_w)
    
    def unpad_image(self, image: np.ndarray, pad_h: int, pad_w: int) -> np.ndarray:
        """
        移除填充
        
        Args:
            image: 填充后的图像
            pad_h: 高度填充量
            pad_w: 宽度填充量
            
        Returns:
            原始大小的图像
        """
        if pad_h > 0:
            image = image[:-pad_h]
        if pad_w > 0:
            image = image[:, :-pad_w]
        
        return image


def create_noisy_clean_pairs(images: List[np.ndarray], noise_sigma: float = 0.1,
                            num_noise_levels: int = 1) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    创建含噪和干净图像对
    
    Args:
        images: 干净图像列表
        noise_sigma: 高斯噪声标准差
        num_noise_levels: 噪声等级数量
        
    Returns:
        (含噪图像, 干净图像) 对的列表
    """
    pairs = []
    preprocessor = ImagePreprocessor()
    
    for clean_image in images:
        for level in range(num_noise_levels):
            sigma = noise_sigma * (1 + level)
            noisy_image = preprocessor.add_gaussian_noise(clean_image, sigma=sigma)
            pairs.append((noisy_image, clean_image))
    
    return pairs
