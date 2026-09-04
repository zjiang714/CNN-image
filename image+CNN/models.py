"""
CNN模型定义
用于特征提取和滤波器参数预测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
    """
    CNN特征提取器（增强版）
    从图像块中提取特征，支持残差连接
    """
    
    def __init__(self, in_channels: int = 1, num_features: int = 64):
        """
        初始化特征提取器
        
        Args:
            in_channels: 输入通道数
            num_features: 特征维度
        """
        super(FeatureExtractor, self).__init__()
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.GroupNorm(8, 32)
        
        # 中间卷积层
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.GroupNorm(8, 64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.GroupNorm(8, 128)
        
        # 增强：更深的卷积层
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.GroupNorm(8, 128)
        
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn5 = nn.GroupNorm(8, 128)
        
        self.conv6 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn6 = nn.GroupNorm(8, 64)
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 特征维度
        self.num_features = num_features
        self.feature_fc = nn.Linear(64, num_features)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (B, C, H, W)
            
        Returns:
            特征向量 (B, num_features)
        """
        # 第一个卷积块
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # 第二个卷积块
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # 第三个卷积块
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        # 增强：深层卷积块（残差连接）
        identity = x
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        
        x = self.conv5(x)
        x = self.bn5(x)
        x = x + identity  # 残差连接
        x = F.relu(x)
        
        x = self.conv6(x)
        x = self.bn6(x)
        x = F.relu(x)
        
        # 全局池化和展平
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # 特征向量
        features = self.feature_fc(x)
        features = F.relu(features)
        
        return features


class ParameterPredictor(nn.Module):
    """
    参数预测网络（增强版）
    从特征向量预测多个自适应滤波器参数
    """
    
    def __init__(self, num_features: int = 64, num_params: int = 8):
        """
        初始化参数预测器
        
        Args:
            num_features: 输入特征维度
            num_params: 预测参数个数（默认 8 个）
        """
        super(ParameterPredictor, self).__init__()
        
        self.fc1 = nn.Linear(num_features, 256)
        self.ln1 = nn.LayerNorm(256)
        self.dropout1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)
        self.dropout2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(128, 64)
        self.ln3 = nn.LayerNorm(64)
        
        # 输出参数
        self.fc_out = nn.Linear(64, num_params)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            features: 特征向量 (B, num_features)
            
        Returns:
            预测的参数 (B, num_params)，范围在 [0, 1]
        """
        x = self.fc1(features)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.ln2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.ln3(x)
        x = F.relu(x)
        
        # 输出参数，使用sigmoid限制在[0, 1]范围
        params = torch.sigmoid(self.fc_out(x))
        
        return params


class DenoisingNetwork(nn.Module):
    """
    完整的去噪网络
    结合特征提取和参数预测
    """
    
    def __init__(self, in_channels: int = 1, num_features: int = 64, 
                 num_params: int = 8):
        """
        初始化完整去噪网络
        
        Args:
            in_channels: 输入通道数
            num_features: 特征维度
            num_params: 预测参数个数
        """
        super(DenoisingNetwork, self).__init__()
        
        self.feature_extractor = FeatureExtractor(in_channels, num_features)
        self.parameter_predictor = ParameterPredictor(num_features, num_params)
        
    def forward(self, x: torch.Tensor) -> tuple:
        """
        前向传播
        
        Args:
            x: 输入图像块 (B, C, H, W)
            
        Returns:
            (特征向量, 预测参数)
        """
        features = self.feature_extractor(x)
        params = self.parameter_predictor(features)
        
        return features, params


def create_model(in_channels: int = 1, num_features: int = 64,
                num_params: int = 8, device: str = 'cpu') -> DenoisingNetwork:
    """
    创建去噪模型AI Chat
    Args:
        in_channels: 输入通道数
        num_features: 特征维度
        num_params: 预测参数个数
        device: 计算设备
        
    Returns:
        去噪网络模型
    """
    model = DenoisingNetwork(in_channels, num_features, num_params)
    model = model.to(device)
    return model
