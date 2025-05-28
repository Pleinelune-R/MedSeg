import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        # 将预测值转换为概率
        pred = torch.sigmoid(pred)
        
        # 计算交集（在所有空间维度上求和）
        intersection = (pred * target).sum(dim=(1, 2, 3))  # 在所有空间维度上求和
        
        # 计算 Dice 系数
        dice = (2. * intersection + self.smooth) / (
            pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + self.smooth
        )
        
        # 返回批次的平均 Dice Loss
        return 1 - dice.mean()

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, smooth=1e-5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.dice_loss = DiceLoss(smooth=smooth)
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, pred, target):
        dice_loss = self.dice_loss(pred, target)
        bce_loss = self.bce_loss(pred, target)
        return self.alpha * dice_loss + (1 - self.alpha) * bce_loss 