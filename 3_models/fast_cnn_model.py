"""
快速 1D-CNN 模型
用于实时手势识别，支持 < 30秒快速训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class FastGestureCNN(nn.Module):
    """
    轻量级 1D-CNN 模型
    专为 FMG 传感器数据设计，实现快速训练和推理
    """
    
    def __init__(self, 
                 num_channels: int = 24,
                 num_gestures: int = 5,
                 window_size: int = 200):
        """
        初始化模型
        
        Args:
            num_channels: 输入通道数（传感器数）
            num_gestures: 手势类别数
            window_size: 时间窗口大小
        """
        super(FastGestureCNN, self).__init__()
        
        self.num_channels = num_channels
        self.num_gestures = num_gestures
        self.window_size = window_size
        
        # 第一层卷积块
        self.conv1 = nn.Conv1d(
            in_channels=num_channels,
            out_channels=32,
            kernel_size=5,
            padding=2,
            stride=1
        )
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # 第二层卷积块
        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=5,
            padding=2,
            stride=1
        )
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # 第三层卷积块
        self.conv3 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1,
            stride=1
        )
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.AdaptiveAvgPool1d(1)
        
        # 全连接层
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_gestures)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, num_channels, window_size)
            
        Returns:
            输出张量 (batch_size, num_gestures)
        """
        # 第一个卷积块
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # 第二个卷积块
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # 第三个卷积块
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class LightweightGestureCNN(nn.Module):
    """
    超轻量级模型（更快的训练）
    用于极度资源受限的场景
    """
    
    def __init__(self, 
                 num_channels: int = 24,
                 num_gestures: int = 5):
        super(LightweightGestureCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(num_channels, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Linear(32, num_gestures)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


class ResidualBlock(nn.Module):
    """
    残差块（用于更深的网络）
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=kernel_size//2
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, padding=kernel_size//2
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # 残差连接
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = F.relu(out)
        
        return out


class ResNetGesture(nn.Module):
    """
    基于残差网络的手势识别模型
    """
    
    def __init__(self, 
                 num_channels: int = 24,
                 num_gestures: int = 5,
                 num_blocks: int = 2):
        super(ResNetGesture, self).__init__()
        
        self.conv_in = nn.Conv1d(num_channels, 32, kernel_size=7, padding=3)
        self.bn_in = nn.BatchNorm1d(32)
        
        # 构建残差块
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(32, 32, kernel_size=5) for _ in range(num_blocks)],
            ResidualBlock(32, 64, kernel_size=5),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.fc = nn.Linear(64, num_gestures)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        x = self.bn_in(x)
        x = F.relu(x)
        
        x = self.res_blocks(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


def get_model(model_type: str = 'fast_cnn',
              num_channels: int = 24,
              num_gestures: int = 5) -> nn.Module:
    """
    获取模型实例
    
    Args:
        model_type: 模型类型 ('fast_cnn', 'lightweight', 'resnet')
        num_channels: 输入通道数
        num_gestures: 手势类别数
        
    Returns:
        模型实例
    """
    if model_type == 'fast_cnn':
        return FastGestureCNN(num_channels, num_gestures)
    elif model_type == 'lightweight':
        return LightweightGestureCNN(num_channels, num_gestures)
    elif model_type == 'resnet':
        return ResNetGesture(num_channels, num_gestures)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # 测试模型
    print("测试 FastGestureCNN...")
    model = FastGestureCNN(num_channels=24, num_gestures=5, window_size=200)
    print(model)
    
    # 输入测试
    dummy_input = torch.randn(8, 24, 200)  # (batch_size, channels, window_size)
    output = model(dummy_input)
    print(f"\n输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 测试其他模型
    print("\n" + "="*50)
    print("测试 LightweightGestureCNN...")
    lightweight_model = LightweightGestureCNN()
    output = lightweight_model(dummy_input)
    print(f"参数数量: {sum(p.numel() for p in lightweight_model.parameters())}")
    
    print("\n" + "="*50)
    print("测试 ResNetGesture...")
    resnet_model = ResNetGesture()
    output = resnet_model(dummy_input)
    print(f"参数数量: {sum(p.numel() for p in resnet_model.parameters())}")
