"""
模型评估模块
包括准确率、F1分数、混淆矩阵等指标
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List


class ModelEvaluator:
    """
    模型评估类
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        """
        初始化评估器
        
        Args:
            model: 要评估的模型
            device: 计算设备
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def predict(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        进行预测
        
        Args:
            data_loader: 数据加载器
            
        Returns:
            (预测标签, 真实标签)
        """
        predictions = []
        ground_truths = []
        
        with torch.no_grad():
            for data, labels in data_loader:
                data = data.to(self.device)
                
                outputs = self.model(data)
                _, predicted = torch.max(outputs, 1)
                
                predictions.extend(predicted.cpu().numpy())
                ground_truths.extend(labels.numpy())
        
        return np.array(predictions), np.array(ground_truths)
    
    def evaluate(self, data_loader: DataLoader, 
                gesture_names: List[str] = None) -> dict:
        """
        评估模型
        
        Args:
            data_loader: 数据加载器
            gesture_names: 手势名称列表
            
        Returns:
            评估结果字典
        """
        predictions, ground_truths = self.predict(data_loader)
        
        # 计算指标
        accuracy = accuracy_score(ground_truths, predictions)
        precision = precision_score(ground_truths, predictions, average='weighted', zero_division=0)
        recall = recall_score(ground_truths, predictions, average='weighted', zero_division=0)
        f1 = f1_score(ground_truths, predictions, average='weighted', zero_division=0)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': confusion_matrix(ground_truths, predictions),
            'predictions': predictions,
            'ground_truths': ground_truths
        }
        
        # 打印报告
        print("\n" + "="*50)
        print("模型评估报告")
        print("="*50)
        print(f"准确率 (Accuracy):  {accuracy:.4f}")
        print(f"精准率 (Precision): {precision:.4f}")
        print(f"召回率 (Recall):    {recall:.4f}")
        print(f"F1 分数 (F1-Score):  {f1:.4f}")
        print("\n" + "-"*50)
        
        if gesture_names:
            print(classification_report(
                ground_truths, predictions,
                target_names=gesture_names,
                zero_division=0
            ))
        else:
            print(classification_report(
                ground_truths, predictions,
                zero_division=0
            ))
        
        return results
    
    def plot_confusion_matrix(self, 
                             confusion_mat: np.ndarray,
                             gesture_names: List[str] = None,
                             save_path: str = None):
        """
        绘制混淆矩阵
        
        Args:
            confusion_mat: 混淆矩阵
            gesture_names: 手势名称
            save_path: 保存路径
        """
        plt.figure(figsize=(10, 8))
        
        if gesture_names:
            sns.heatmap(confusion_mat, annot=True, fmt='d',
                       xticklabels=gesture_names,
                       yticklabels=gesture_names,
                       cmap='Blues')
        else:
            sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues')
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"混淆矩阵已保存到: {save_path}")
        else:
            plt.show()
    
    def plot_per_class_metrics(self, 
                              data_loader: DataLoader,
                              gesture_names: List[str] = None,
                              save_path: str = None):
        """
        绘制每个类别的评估指标
        
        Args:
            data_loader: 数据加载器
            gesture_names: 手势名称
            save_path: 保存路径
        """
        predictions, ground_truths = self.predict(data_loader)
        
        # 计算每个类别的指标
        precisions = precision_score(ground_truths, predictions, average=None, zero_division=0)
        recalls = recall_score(ground_truths, predictions, average=None, zero_division=0)
        f1_scores = f1_score(ground_truths, predictions, average=None, zero_division=0)
        
        num_classes = len(np.unique(ground_truths))
        x = np.arange(num_classes)
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.bar(x - width, precisions, width, label='Precision', alpha=0.8)
        ax.bar(x, recalls, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Gesture Class')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Evaluation Metrics')
        ax.set_xticks(x)
        
        if gesture_names:
            ax.set_xticklabels(gesture_names, rotation=45)
        
        ax.legend()
        ax.set_ylim([0, 1.1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        else:
            plt.show()


if __name__ == "__main__":
    # 示例：评估模型
    from fast_cnn_model import FastGestureCNN
    from torch.utils.data import TensorDataset
    
    # 创建虚拟数据
    X_test = np.random.randn(100, 24, 200).astype(np.float32)
    y_test = np.random.randint(0, 5, 100)
    
    test_dataset = TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(y_test)
    )
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # 创建模型
    model = FastGestureCNN(num_channels=24, num_gestures=5)
    
    # 评估
    evaluator = ModelEvaluator(model)
    gesture_names = ['Rest', 'Pinch', 'Power', 'Tripod', 'Extension']
    
    results = evaluator.evaluate(test_loader, gesture_names)
    
    # 绘制图表
    evaluator.plot_confusion_matrix(results['confusion_matrix'], gesture_names)
    evaluator.plot_per_class_metrics(test_loader, gesture_names)
