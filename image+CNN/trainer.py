"""
训练模块
结合CNN参数预测和双调滤波的端到端训练
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, List, Optional
from pathlib import Path
import json
from datetime import datetime

from models import DenoisingNetwork
from bitonic_filter import BitonicFilter
from preprocessor import ImagePreprocessor


class DenoisingTrainer:
    """
    去噪网络训练器
    """
    
    def __init__(self, model: DenoisingNetwork, device: str = 'cpu',
                 learning_rate: float = 1e-3, block_size: int = 32, num_params: int = 8):
        """
        初始化训练器
        
        Args:
            model: 去噪网络
            device: 计算设备
            learning_rate: 学习率
            block_size: 图像块大小
        """
        self.model = model
        self.device = device
        self.block_size = block_size
        self.preprocessor = ImagePreprocessor(block_size=block_size)
        self.bitonic_filter = BitonicFilter(kernel_size=3)
        
        # 优化器
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)
        
        # 损失函数
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'psnr': [],
            'ssim': []
        }
        
    def _image_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """
        将图像转换为张量
        
        Args:
            image: 输入图像 (H, W) 或 (H, W, C)
            
        Returns:
            张量 (1, C, H, W) 或 (1, 1, H, W)
        """
        if len(image.shape) == 2:
            tensor = torch.FloatTensor(image[np.newaxis, np.newaxis, :, :])
        else:
            # 交换通道顺序: (H, W, C) -> (C, H, W)
            image = np.transpose(image, (2, 0, 1))
            tensor = torch.FloatTensor(image[np.newaxis, :, :, :])
        
        return tensor.to(self.device)
    
    def _apply_bitonic_denoising(self, block: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        应用双调滤波去噪
        
        Args:
            block: 图像块
            params: 滤波参数 [alpha, beta]
            
        Returns:
            去噪后的块
        """
        return self.bitonic_filter.apply(block, params)
    
    def _calculate_psnr(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        计算PSNR (Peak Signal-to-Noise Ratio)
        
        Args:
            img1: 预测图像
            img2: 参考图像
            
        Returns:
            PSNR值
        """
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return 100.0
        
        max_pixel = 1.0  # 归一化图像的最大值
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr
    
    def _calculate_ssim(self, img1: np.ndarray, img2: np.ndarray, window_size: int = 11) -> float:
        """
        计算SSIM (Structural Similarity Index)
        
        Args:
            img1: 预测图像
            img2: 参考图像
            window_size: 窗口大小
            
        Returns:
            SSIM值
        """
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        # 简化版SSIM计算
        mean1 = np.mean(img1)
        mean2 = np.mean(img2)
        
        var1 = np.var(img1)
        var2 = np.var(img2)
        covar = np.mean((img1 - mean1) * (img2 - mean2))
        
        ssim = ((2 * mean1 * mean2 + c1) * (2 * covar + c2)) / \
               ((mean1 ** 2 + mean2 ** 2 + c1) * (var1 + var2 + c2))
        
        return float(ssim)
    
    def train_step(self, noisy_image: np.ndarray, clean_image: np.ndarray) -> dict:
        """
        单个训练步骤
        
        Args:
            noisy_image: 含噪图像
            clean_image: 干净图像
            
        Returns:
            包含损失和指标的字典
        """
        self.model.train()
        
        # 分块
        noisy_blocks, positions = self.preprocessor.split_into_blocks(noisy_image)
        clean_blocks, _ = self.preprocessor.split_into_blocks(clean_image)
        
        total_loss = 0.0
        num_blocks = len(noisy_blocks)
        
        denoised_blocks = []
        
        # 清空梯度
        self.optimizer.zero_grad()
        batch_loss = 0.0
        
        for noisy_block, clean_block in zip(noisy_blocks, clean_blocks):
            # 转换为张量
            noisy_tensor = self._image_to_tensor(noisy_block)
            clean_tensor = self._image_to_tensor(clean_block)
            
            # 前向传播
            features, params = self.model(noisy_tensor)
            
            # 应用双调滤波
            filtered_block = self._apply_bitonic_denoising(noisy_block, 
                                                           params.detach().cpu().numpy()[0])
            filtered_tensor = self._image_to_tensor(filtered_block)
            
            # 计算损失
            # 1. 重构损失：去噪图像 vs 干净图像
            reconstruction_loss = self.mse_loss(filtered_tensor, clean_tensor)
            
            # 2. 参数正则化：鼓励参数在合理范围内
            param_loss = torch.mean(torch.abs(params - 0.5))
            
            # 总损失
            loss = reconstruction_loss + 0.1 * param_loss
            batch_loss += loss
            
            denoised_blocks.append(filtered_block)
        
        # 统一反向传播
        batch_loss = batch_loss / num_blocks
        batch_loss.backward()
        self.optimizer.step()
        
        total_loss = batch_loss.item()
        
        # 合并块
        denoised_image = self.preprocessor.merge_blocks(denoised_blocks, positions, 
                                                        noisy_image.shape[:2])
        
        # 计算指标
        psnr = self._calculate_psnr(denoised_image, clean_image)
        ssim = self._calculate_ssim(denoised_image, clean_image)
        
        return {
            'loss': total_loss,
            'psnr': psnr,
            'ssim': ssim,
            'denoised_image': denoised_image
        }
    
    @torch.no_grad()
    def validate(self, noisy_image: np.ndarray, clean_image: np.ndarray) -> dict:
        """
        验证步骤
        
        Args:
            noisy_image: 含噪图像
            clean_image: 干净图像
            
        Returns:
            包含损失和指标的字典
        """
        self.model.eval()
        
        # 分块
        noisy_blocks, positions = self.preprocessor.split_into_blocks(noisy_image)
        clean_blocks, _ = self.preprocessor.split_into_blocks(clean_image)
        
        total_loss = 0.0
        denoised_blocks = []
        
        for noisy_block, clean_block in zip(noisy_blocks, clean_blocks):
            # 转换为张量
            noisy_tensor = self._image_to_tensor(noisy_block)
            clean_tensor = self._image_to_tensor(clean_block)
            
            # 前向传播
            features, params = self.model(noisy_tensor)
            
            # 应用双调滤波
            filtered_block = self._apply_bitonic_denoising(noisy_block, 
                                                           params.cpu().numpy()[0])
            filtered_tensor = self._image_to_tensor(filtered_block)
            
            # 计算损失
            loss = self.mse_loss(filtered_tensor, clean_tensor)
            total_loss += loss.item()
            
            denoised_blocks.append(filtered_block)
        
        # 合并块
        denoised_image = self.preprocessor.merge_blocks(denoised_blocks, positions,
                                                        noisy_image.shape[:2])
        
        # 计算指标
        psnr = self._calculate_psnr(denoised_image, clean_image)
        ssim = self._calculate_ssim(denoised_image, clean_image)
        
        return {
            'loss': total_loss / len(noisy_blocks),
            'psnr': psnr,
            'ssim': ssim,
            'denoised_image': denoised_image
        }
    
    def fit(self, train_data: List[Tuple[np.ndarray, np.ndarray]],
           val_data: List[Tuple[np.ndarray, np.ndarray]],
           epochs: int = 100, save_dir: Optional[str] = None) -> dict:
        """
        训练模型
        
        Args:
            train_data: 训练数据 [(noisy, clean), ...]
            val_data: 验证数据
            epochs: 训练轮次
            save_dir: 模型保存目录
            
        Returns:
            训练历史
        """
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        best_psnr = 0.0
        
        for epoch in range(epochs):
            # 训练
            train_loss = 0.0
            train_psnr = 0.0
            train_ssim = 0.0
            
            for noisy, clean in train_data:
                result = self.train_step(noisy, clean)
                train_loss += result['loss']
                train_psnr += result['psnr']
                train_ssim += result['ssim']
            
            train_loss /= len(train_data)
            train_psnr /= len(train_data)
            train_ssim /= len(train_data)
            
            # 验证
            val_loss = 0.0
            val_psnr = 0.0
            val_ssim = 0.0
            
            for noisy, clean in val_data:
                result = self.validate(noisy, clean)
                val_loss += result['loss']
                val_psnr += result['psnr']
                val_ssim += result['ssim']
            
            val_loss /= len(val_data)
            val_psnr /= len(val_data)
            val_ssim /= len(val_data)
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['psnr'].append(val_psnr)
            self.history['ssim'].append(val_ssim)
            
            # 打印进度
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.8f}, PSNR: {train_psnr:.4f}, SSIM: {train_ssim:.6f}")
            print(f"  Val Loss: {val_loss:.8f}, PSNR: {val_psnr:.4f}, SSIM: {val_ssim:.6f}")
            print(f"  Current LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # 保存最佳模型
            if val_psnr > best_psnr and save_dir:
                best_psnr = val_psnr
                checkpoint = {
                    'epoch': epoch,
                    'model_state': self.model.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'psnr': val_psnr,
                    'ssim': val_ssim
                }
                torch.save(checkpoint, f"{save_dir}/best_model.pth")
            
            # 学习率调度
            self.scheduler.step()
        
        return self.history
    
    def save_model(self, path: str):
        """保存模型"""
        # Ensure parent directory exists (Windows may pass paths like '/tmp/...')
        try:
            parent = Path(path).parent
            if str(parent) != '' and not parent.exists():
                parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            # If we fail to create the directory (permissions, invalid path),
            # fall back to saving in the current working directory.
            try:
                torch.save(self.model.state_dict(), Path(path).name)
                return
            except Exception:
                # Finally, re-raise the original save error to surface it.
                pass

        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path: str):
        """加载模型"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
