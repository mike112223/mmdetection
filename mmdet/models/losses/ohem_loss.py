import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss


def _expand_binary_labels(labels, label_weights, label_channels, bg_label):
    # Caution: this function should only be used in RPN
    # in other files such as in ghm_loss, the _expand_binary_labels
    # is used for multi-class classification.
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(
        (labels >= 0) & (labels < bg_label), as_tuple=False).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds]] = 1
    if label_weights is None:
        bin_label_weights = None
    else:
        bin_label_weights = label_weights.view(-1, 1).expand(
            label_weights.size(0), label_channels)
    return bin_labels, bin_label_weights


@LOSSES.register_module()
class OHEMLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 background_label=0,
                 neg_pos_ratio=3,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0):
        """CrossEntropyLoss.

        Args:
            use_sigmoid (bool, optional): Whether the prediction uses sigmoid
                of softmax. Defaults to False.
            use_mask (bool, optional): Whether to use mask cross entropy loss.
                Defaults to False.
            reduction (str, optional): . Defaults to 'mean'.
                Options are "none", "mean" and "sum".
            class_weight (list[float], optional): Weight of each class.
                Defaults to None.
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        """
        super(OHEMLoss, self).__init__()
        assert (use_sigmoid is True)
        self.background_label = background_label
        self.neg_pos_ratio = neg_pos_ratio
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The prediction.
            label (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction (str, optional): The method used to reduce the loss.
                Options are "none", "mean" and "sum".
        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None
        loss = self.loss_weight * self.ohem_binary_cross_entropy(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss

    def ohem_binary_cross_entropy(self,
                                  pred,
                                  label,
                                  weight=None,
                                  reduction='mean',
                                  avg_factor=None,
                                  class_weight=None):
        """Calculate the binary CrossEntropy loss.

        Args:
            pred (torch.Tensor): The prediction with shape (N, 1).
            label (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            reduction (str, optional): The method used to reduce the loss.
                Options are "none", "mean" and "sum".
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            class_weight (list[float], optional): The weight for each class.

        Returns:
            torch.Tensor: The calculated loss
        """

        pos_inds = ((label >= 0) &
                    (label < self.background_label)).nonzero().reshape(-1)
        neg_inds = (label == self.background_label).nonzero().view(-1)

        if pred.dim() != label.dim():
            label, weight = _expand_binary_labels(label,
                                                  weight,
                                                  pred.size(-1),
                                                  self.background_label)

        # weighted element-wise losses
        if weight is not None:
            weight = weight.float()

        loss_all = F.binary_cross_entropy_with_logits(
            pred, label.float(), weight=class_weight, reduction='none')
        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes

        num_pos_samples = pos_inds.size(0)
        num_neg_samples = self.neg_pos_ratio * num_pos_samples
        if num_neg_samples > neg_inds.size(0):
            num_neg_samples = neg_inds.size(0)

        _, topk_loss_neg_idx = loss_all[neg_inds].reshape(-1).topk(num_neg_samples)
        total_idx = torch.cat([topk_loss_neg_idx, pos_inds])
        loss = loss_all[total_idx]
        weight = weight[total_idx]

        # do the reduction for the weighted loss
        loss = weight_reduce_loss(
            loss, weight, reduction=reduction, avg_factor=avg_factor)

        return loss
