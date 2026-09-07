"""
快速模型训练脚本
目标：30秒内完成训练
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
from tqdm import tqdm
import numpy as np
from typing import Tuple, List

from fast_cnn_model import FastGestureCNN, LightweightGestureCNN


class FastTrainer:
    """
    快速训练器
    实现 < 30秒的训练循环
    """
    
    def __init__(self, 
                 model: nn.Module,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 learning_rate: float = 0.001,
                 use_mixed_precision: bool = True):
        """
        初始化训练器
        
        Args:
            model: 模型实例
            device: 计算设备 ('cuda' 或 'cpu')
            learning_rate: 学习率
            use_mixed_precision: 是否使用混合精度训练（加速）
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.use_mixed_precision = use_mixed_precision
        
        # 混合精度训练
        if use_mixed_precision and device == 'cuda':
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        self.train_losses = []
        self.val_accuracies = []
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        训练一个 epoch
        
        Args:
            train_loader: 训练数据加载器
            
        Returns:
            平均损失
        """
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (data, labels) in enumerate(train_loader):
            data = data.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                # 混合精度前向传播
                with torch.cuda.amp.autocast():
                    outputs = self.model(data)
                    loss = self.criterion(outputs, labels)
                
                # 混合精度反向传播
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # 标准前向传播
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        验证模型
        
        Args:
            val_loader: 验证数据加载器
            
        Returns:
            (准确率, 损失)
        """
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        
        with torch.no_grad():
            for data, labels in val_loader:
                data = data.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(val_loader)
        
        return accuracy, avg_loss
    
    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader = None,
              max_time: float = 25.0,
              epochs: int = 100) -> dict:
        """
        快速训练
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器（可选）
            max_time: 最大训练时间（秒）
            epochs: 最大 epoch 数
            
        Returns:
            训练结果字典
        """
        start_time = time.time()
        best_accuracy = 0
        patience = 3
        patience_counter = 0
        
        print(f"开始训练... (最大时间: {max_time}秒)")
        print(f"设备: {self.device}")
        print(f"模型参数数: {sum(p.numel() for p in self.model.parameters())}")
        print("-" * 50)
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # 训练
            train_loss = self.train_epoch(train_loader)
            
            # 验证
            if val_loader is not None:
                val_accuracy, val_loss = self.validate(val_loader)
                self.val_accuracies.append(val_accuracy)
                
                print(f"Epoch {epoch+1:3d} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Acc: {val_accuracy:.2f}% | "
                      f"Time: {time.time()-epoch_start:.2f}s")
                
                # 早停
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"早停 at epoch {epoch+1}")
                        break
            else:
                print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f}")
            
            self.train_losses.append(train_loss)
            
            # 检查时间
            elapsed_time = time.time() - start_time
            if elapsed_time > max_time:
                print(f"\n达到最大训练时间 ({elapsed_time:.2f}s > {max_time}s)")
                break
        
        elapsed_time = time.time() - start_time
        
        print("-" * 50)
        print(f"训练完成！")
        print(f"总耗时: {elapsed_time:.2f}秒")
        if self.val_accuracies:
            print(f"最佳验证准确率: {max(self.val_accuracies):.2f}%")
        
        return {
            'elapsed_time': elapsed_time,
            'train_losses': self.train_losses,
            'val_accuracies': self.val_accuracies,
            'best_accuracy': max(self.val_accuracies) if self.val_accuracies else None
        }


def create_dummy_dataset(num_samples: int = 200,
                       num_channels: int = 24,
                       window_size: int = 200,
                       num_classes: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    创建虚拟数据集用于测试
    
    Args:
        num_samples: 样本数
        num_channels: 通道数
        window_size: 窗口大小
        num_classes: 类别数
        
    Returns:
        (数据, 标签)
    """
    X = np.random.randn(num_samples, num_channels, window_size).astype(np.float32)
    y = np.random.randint(0, num_classes, num_samples)
    
    return X, y


if __name__ == "__main__":
    # 创建虚拟数据
    print("创建虚拟数据集...")
    X, y = create_dummy_dataset(num_samples=300, num_classes=5)
    
    # 分割训练验证集
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # 转换为 PyTorch 张量
    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train)
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_val)
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 创建模型
    print("\n创建模型...")
    model = FastGestureCNN(num_channels=24, num_gestures=5)
    
    # 创建训练器
    trainer = FastTrainer(model, learning_rate=0.001)
    
    # 训练
    print("\n开始训练...")
    results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        max_time=25.0,
        epochs=100
    )
    
    print(f"\n结果: {results}")
