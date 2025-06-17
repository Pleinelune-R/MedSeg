import torch 
import torch.nn as nn 
import torch.nn.functional as F 
 
class MultiClassDiceLoss(nn.Module):
    """Calculate dice loss for multi-class segmentation."""
    def __init__(self, eps: float = 1e-9, smooth: float = 1.0):
        super(MultiClassDiceLoss, self).__init__()
        self.eps = eps
        self.smooth = smooth
        
    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, C, H, W, D] - model predictions (before softmax)
            targets: [B, C, H, W, D] - one-hot encoded ground truth
        """
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=1)
        
        # Flatten tensors for calculation
        probs_flat = probs.view(probs.size(0), probs.size(1), -1)  # [B, C, H*W*D]
        targets_flat = targets.view(targets.size(0), targets.size(1), -1)  # [B, C, H*W*D]
        
        # Calculate intersection and union for each class
        intersection = (probs_flat * targets_flat).sum(dim=2)  # [B, C]
        union = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)  # [B, C]
        
        # Calculate Dice score for each class
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth + self.eps)
        
        # Average Dice score across classes (excluding background if needed)
        dice_loss = 1.0 - dice_score.mean(dim=1)  # [B]
        
        # Get predictions for visualization
        pred_masks = torch.argmax(probs, dim=1)  # [B, H, W, D]
        
        return pred_masks, dice_loss.mean()
        
class WeightedCrossEntropyLoss(nn.Module):
    """Weighted Cross Entropy Loss for handling class imbalance."""
    def __init__(self, class_weights=None):
        super(WeightedCrossEntropyLoss, self).__init__()
        if class_weights is None:
            # Default weights: give more importance to tumor classes
            # Background: 1.0, NCR/NET: 2.0, ED: 1.5, ET: 3.0
            self.class_weights = torch.tensor([1.0, 2.0, 1.5, 3.0])
        else:
            self.class_weights = torch.tensor(class_weights)
        
    def forward(self, logits, targets):
        """
        Args:
            logits: [B, C, H, W, D] - model predictions
            targets: [B, C, H, W, D] - one-hot encoded ground truth
        """
        # Convert one-hot targets to class indices
        target_indices = torch.argmax(targets, dim=1)  # [B, H, W, D]
        
        # Apply weighted cross entropy
        ce_loss = F.cross_entropy(logits, target_indices, 
                                 weight=self.class_weights.to(logits.device),
                                 reduction='mean')
        return ce_loss

class MultiClassBCEDiceLoss(nn.Module):
    """Compute objective loss: Weighted CE loss + Multi-class DICE loss."""
    def __init__(self, ce_weight=0.5, dice_weight=0.5, class_weights=None):
        super(MultiClassBCEDiceLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = WeightedCrossEntropyLoss(class_weights)
        self.dice_loss = MultiClassDiceLoss()
        
    def forward(self, 
                logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, C, H, W, D] - model predictions
            targets: [B, C, H, W, D] - one-hot encoded ground truth
        """
        assert(logits.shape == targets.shape)
        
        # Calculate losses
        pred_masks, dice_loss = self.dice_loss(logits, targets)
        ce_loss = self.ce_loss(logits, targets)
        
        # Combine losses
        total_loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss
        
        return pred_masks, ce_loss, dice_loss, total_loss

# Keep the old classes for backward compatibility
class DiceLoss(nn.Module):
    """Calculate dice loss."""
    def __init__(self, eps: float = 1e-9):
    """Calculate dice loss."""
    def __init__(self, eps: float = 1e-9):
        super(DiceLoss, self).__init__()
        self.eps = eps
        
    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        
        num = targets.size(0)
        probability_sigmoid = torch.sigmoid(logits)
        pred_masks = (probability_sigmoid > 0.5).float()  # threshold at 0.5 to get binary predictions
        probability_sigmoid = probability_sigmoid.view(num, -1)
        targets = targets.view(num, -1)
        assert(probability_sigmoid.shape == targets.shape)
        
        intersection = 2.0 * (probability_sigmoid * targets).sum()
        union = probability_sigmoid.sum() + targets.sum()
        dice_score = (intersection + self.eps) / union
        #print("intersection", intersection, union, dice_score)
        return pred_masks, 1.0 - dice_score
        
        
class BCEDiceLoss(nn.Module):
    """Compute objective loss: BCE loss + DICE loss."""
    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        
    def forward(self, 
                logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        assert(logits.shape == targets.shape)
        pre_masks, dice_loss = self.dice(logits, targets)
        bce_loss = self.bce(logits, targets)
        
        return pre_masks, bce_loss, dice_loss, bce_loss + dice_loss
        self.eps = eps
        
    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        
        num = targets.size(0)
        probability_sigmoid = torch.sigmoid(logits)
        pred_masks = (probability_sigmoid > 0.5).float()  # threshold at 0.5 to get binary predictions
        probability_sigmoid = probability_sigmoid.view(num, -1)
        targets = targets.view(num, -1)
        assert(probability_sigmoid.shape == targets.shape)
        
        intersection = 2.0 * (probability_sigmoid * targets).sum()
        union = probability_sigmoid.sum() + targets.sum()
        dice_score = (intersection + self.eps) / union
        #print("intersection", intersection, union, dice_score)
        return pred_masks, 1.0 - dice_score
        
        
class BCEDiceLoss(nn.Module):
    """Compute objective loss: BCE loss + DICE loss."""
    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        
    def forward(self, 
                logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        assert(logits.shape == targets.shape)
        pre_masks, dice_loss = self.dice(logits, targets)
        bce_loss = self.bce(logits, targets)
        
        return pre_masks, bce_loss, dice_loss, bce_loss + dice_loss