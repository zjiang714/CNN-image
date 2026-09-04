"""
双调滤波器(Bitonic Filter)实现
用于图像去噪的参数化滤波
"""

import numpy as np
from typing import Tuple


class BitonicFilter:
    """
    双调滤波器实现
    基于排序网络的中值滤波变体
    """
    
    def __init__(self, kernel_size: int = 3, alpha: float = 0.5, beta: float = 0.5):
        """
        初始化双调滤波器
        
        Args:
            kernel_size: 滤波核大小
            alpha: 控制平滑度的参数 (0-1)
            beta: 控制边界保留的参数 (0-1)
        """
        self.kernel_size = kernel_size
        self.alpha = alpha  # 平滑度参数
        self.beta = beta    # 边界保留参数
        
    def apply(self, image: np.ndarray, params: np.ndarray = None) -> np.ndarray:
        """
        应用双调滤波
        
        Args:
            image: 输入图像 (H, W) 或 (H, W, C)
            params: 预测的滤波参数
                - 2 个参数：[alpha, beta]
                - 8 个参数：[alpha, beta, gamma, intensity, edge_preserve, detail_enhance, smoothness, boundary_keep]
            
        Returns:
            去噪后的图像
        """
        if params is not None:
            # 支持 2 个或 8 个参数
            if len(params) >= 2:
                self.alpha = float(params[0])
                self.beta = float(params[1])
            if len(params) >= 8:
                # 扩展参数
                gamma = float(params[2])
                intensity = float(params[3])
                edge_preserve = float(params[4])
                detail_enhance = float(params[5])
                smoothness = float(params[6])
                boundary_keep = float(params[7])
                # 组合参数：权重融合多个参数效果
                self.alpha = self.alpha * (1 - gamma) + smoothness * gamma
                self.beta = self.beta * boundary_keep + intensity * (1 - boundary_keep)
        
        if len(image.shape) == 3:
            # 处理彩色图像
            denoised = np.zeros_like(image)
            for c in range(image.shape[2]):
                denoised[:, :, c] = self._apply_channel(image[:, :, c])
            return denoised
        else:
            return self._apply_channel(image)
    
    def _apply_channel(self, channel: np.ndarray) -> np.ndarray:
        """
        对单个通道应用双调滤波
        
        Args:
            channel: 单个通道的图像数据
            
        Returns:
            滤波后的通道
        """
        h, w = channel.shape
        pad = self.kernel_size // 2
        padded = np.pad(channel, pad, mode='reflect')
        
        output = np.zeros_like(channel)
        
        for i in range(h):
            for j in range(w):
                # 提取邻域
                window = padded[i:i+self.kernel_size, j:j+self.kernel_size].flatten()
                
                # 双调排序过程
                sorted_window = self._bitonic_sort(window)
                
                # 混合不同的统计量
                median = sorted_window[len(sorted_window) // 2]
                mean = np.mean(window)
                
                # 加权融合：使用alpha控制中值和均值的混合
                filtered_value = self.alpha * median + (1 - self.alpha) * mean
                
                # 边界保留：使用beta保留原始像素
                output[i, j] = self.beta * filtered_value + (1 - self.beta) * channel[i, j]
        
        return output
    
    def _bitonic_sort(self, arr: np.ndarray) -> np.ndarray:
        """
        双调排序网络
        按升序排列数组元素
        
        Args:
            arr: 输入数组
            
        Returns:
            排序后的数组
        """
        arr = arr.copy()
        n = len(arr)
        
        def bitonic_merge(arr, low, cnt, direction):
            if cnt > 1:
                k = cnt // 2
                bitonic_merge(arr, low, k, 1)
                bitonic_merge(arr, low + k, k, 0)
                bitonic_compare_and_swap(arr, low, k, direction)
        
        def bitonic_compare_and_swap(arr, low, cnt, direction):
            if cnt > 1:
                k = cnt // 2
                for i in range(low, low + k):
                    if (arr[i] > arr[i + k]) == direction:
                        arr[i], arr[i + k] = arr[i + k], arr[i]
                bitonic_compare_and_swap(arr, low, k, direction)
                bitonic_compare_and_swap(arr, low + k, k, direction)
        
        def bitonic_sort_recursive(arr, low, cnt, direction):
            if cnt > 1:
                k = cnt // 2
                bitonic_sort_recursive(arr, low, k, 1)
                bitonic_sort_recursive(arr, low + k, k, 0)
                bitonic_merge(arr, low, cnt, direction)
        
        # 转换为2的幂次以支持标准的双调排序
        power = 1
        while power < n:
            power *= 2
        
        if n != power:
            # 对不是2的幂的数组进行简单排序
            return np.sort(arr)
        
        bitonic_sort_recursive(arr, 0, n, 1)
        return arr


def apply_bitonic_filter(image: np.ndarray, alpha: float = 0.5, beta: float = 0.5,
                        kernel_size: int = 3) -> np.ndarray:
    """
    便利函数：应用双调滤波
    
    Args:
        image: 输入图像
        alpha: 平滑度参数
        beta: 边界保留参数
        kernel_size: 滤波核大小
        
    Returns:
        去噪后的图像
    """
    bf = BitonicFilter(kernel_size=kernel_size, alpha=alpha, beta=beta)
    return bf.apply(image)
