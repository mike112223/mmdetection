from functools import partial

# import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss


# This method is only for debugging
def sigmoid_focal_loss(pred,
                       target,
                       label,
                       weight=None,
                       gamma=2.0,
                       alpha=0.25,
                       reduction='mean',
                       avg_factor=None):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred).reshape(-1, 1)
    label = 1 - label.reshape(-1, 1)
    weight = weight.reshape(-1, 1)

    pt = (1 - pred_sigmoid) * label + pred_sigmoid * (1 - label)
    focal_weight = (alpha * label + (1 - alpha) *
                    (1 - label)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@LOSSES.register_module()
class NoisyFocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 no_focal_pos=False,
                 bg_id=1,
                 reduction='mean',
                 loss_weight=1.0):
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float, optional): A balanced form for Focal Loss.
                Defaults to 0.25.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
        """
        super(NoisyFocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.no_focal_pos = no_focal_pos
        self.bg_id = bg_id

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.no_focal_pos:
            return self.focal_neg_only(
                pred=pred,
                target=target,
                weight=weight,
                reduction=reduction,
                avg_factor=avg_factor
            )

        if self.use_sigmoid:
            loss_cls = self.loss_weight * sigmoid_focal_loss(
                pred,
                target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss_cls

    def focal_neg_only(self,
                       pred,
                       target,
                       weight=None,
                       reduction='mean',
                       avg_factor=None):
        if self.use_sigmoid:
            losses = pred.new_zeros(pred.shape)
            lf = partial(
                sigmoid_focal_loss,
                alpha=self.alpha,
                reduction='none',
                avg_factor=None
            )
            label, score = target

            pos_mask = label != self.bg_id
            neg_mask = label == self.bg_id

            losses[pos_mask] = self.loss_weight * lf(
                pred=pred[pos_mask],
                target=score[pos_mask],
                label=label[pos_mask],
                gamma=2.,
                weight=weight[pos_mask] if weight is not None else None
            ) if pos_mask.any() else 0
            losses[neg_mask] = self.loss_weight * lf(
                pred=pred[neg_mask],
                target=score[neg_mask],
                label=label[neg_mask],
                gamma=self.gamma,
                weight=weight[neg_mask] if weight is not None else None
            ) if neg_mask.any() else 0

            # if not isinstance(loss_pos, int):
            #     x = loss_pos.detach().data.cpu().numpy()
            # else:
            #     x = loss_pos
            # if not isinstance(loss_neg, int):
            #     y = loss_neg.detach().data.cpu().numpy()
            # else:
            #     y = loss_neg

            # print('pos', x, 'neg', y, 'avg', avg_factor)

        else:
            raise NotImplementedError

        loss_cls = weight_reduce_loss(losses, reduction=reduction, avg_factor=avg_factor)

        return loss_cls
