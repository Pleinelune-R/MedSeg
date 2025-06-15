import torch 
import torch.nn  as nn 
import torch.nn.functional  as F 
 
class DiceLoss(nn.Module):
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