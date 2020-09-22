
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init, constant_init

from ..builder import HEADS
from ..utils import CoordLayer
from .anchor_head import AnchorHead
from mmdet.core import (images_to_levels, multi_apply,
                        bbox_overlaps, force_fp32)


@HEADS.register_module()
class InsideSoftRetinaHead(AnchorHead):
    r"""An anchor-based head used in `RetinaNet
    <https://arxiv.org/pdf/1708.02002.pdf>`_.

    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors.

    Example:
        >>> import torch
        >>> self = RetinaHead(11, 7)
        >>> x = torch.rand(1, 7, 32, 32)
        >>> cls_score, bbox_pred = self.forward_single(x)
        >>> # Each anchor predicts a score for each class except background
        >>> cls_per_anchor = cls_score.shape[1] / self.num_anchors
        >>> box_per_anchor = bbox_pred.shape[1] / self.num_anchors
        >>> assert cls_per_anchor == (self.num_classes)
        >>> assert box_per_anchor == 4
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     octave_base_scale=4,
                     scales_per_octave=3,
                     ratios=[0.5, 1.0, 2.0],
                     strides=[8, 16, 32, 64, 128]),
                 detach=True,
                 coord_cfg=None,
                 custom_init=None,
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.detach = detach
        self.coord_cfg = coord_cfg
        self.custom_init = custom_init

        if 'train_cfg' in kwargs:
            if hasattr(kwargs['train_cfg'], 'assigner'):
                self.gpu_assign_thr = kwargs['train_cfg']['assigner'].get('gpu_assign_thr', 1e6)
            else:
                self.gpu_assign_thr = 1e6
        else:
            self.gpu_assign_thr = 1e6

        super(InsideSoftRetinaHead, self).__init__(
            num_classes,
            in_channels,
            anchor_generator=anchor_generator,
            **kwargs)

        if self.coord_cfg is not None:
            self.coord = CoordLayer(**self.coord_cfg)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            if i == 0:
                if self.coord_cfg is not None:
                    chn = self.in_channels + 2
                else:
                    chn = self.in_channels
            else:
                chn = self.feat_channels

            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 4, 3, padding=1)

        # import pdb
        # pdb.set_trace()

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.retina_cls, std=0.01, bias=bias_cls)
        constant_init(self.retina_reg, 0.)

    def forward_single(self, x):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 4.
        """
        if self.coord_cfg is not None:
            x = self.coord(x)

        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        return cls_score, bbox_pred

    def loss_single(self, cls_score, bbox_pred, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, gt_bboxes, num_total_samples):
        """Compute loss of a single scale level.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor wight
                shape (N, num_total_anchors, 4).
            bbox_weights (Tensor): BBox regression loss weights of each anchor
                with shape (N, num_total_anchors, 4).
            num_total_samples (int): If sampling, num total samples equal to
                the number of total anchors; Otherwise, it is the number of
                positive anchors.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        scores = []
        for i in range(len(gt_bboxes)):
            _label = labels[i]
            score = label_weights.new_zeros(_label.shape)

            _gt_bbox = gt_bboxes[i]
            if len(_gt_bbox) == 0:
                continue

            _bbox_pred = bbox_pred[i].permute(1, 2, 0).reshape(-1, 4)
            _bbox_target = bbox_targets[i]
            _anchor = anchors[i]
            _center = torch.cat(
                [((_anchor[:, 0] + _anchor[:, 2]) / 2).reshape(-1, 1),
                 ((_anchor[:, 1] + _anchor[:, 3]) / 2).reshape(-1, 1)], 1)

            bg_class_ind = self.num_classes
            _pos_inds = ((_label >= 0) & (_label < bg_class_ind)).nonzero().squeeze(1)
            # _neg_inds = ((_label >= 0) & (_label == bg_class_ind)).nonzero().squeeze(1)

            _bbox_pred = self.bbox_coder.decode(_anchor, _bbox_pred)

            pos_ious = bbox_overlaps(
                _bbox_pred.detach()[_pos_inds],
                _bbox_target[_pos_inds],
                is_aligned=True)

            # import pdb
            # pdb.set_trace()

            if len(_gt_bbox) > self.gpu_assign_thr:
                device = _gt_bbox
                _gt_bbox = _gt_bbox.cpu()
                _bbox_pred = _bbox_pred.cpu()

            ious = bbox_overlaps(
                _bbox_pred.detach(),
                _gt_bbox)

            max_ious, assigned_gt_ind = ious.max(axis=1)
            if len(_gt_bbox) > self.gpu_assign_thr:
                _gt_bbox = _gt_bbox.to(device)
                # import pdb
                # pdb.set_trace()
                assigned_gt_ind = assigned_gt_ind.to(device).long()
                max_ious = max_ious.to(device)

            # print(assigned_gt_ind)

            _gt_bboxes = _gt_bbox[assigned_gt_ind]
            x1 = _center[:, 0] > _gt_bboxes[:, 0]
            x2 = _center[:, 0] < _gt_bboxes[:, 2]
            y1 = _center[:, 1] > _gt_bboxes[:, 1]
            y2 = _center[:, 1] < _gt_bboxes[:, 3]

            score[x1 * y1 * x2 * y2] = max_ious[x1 * y1 * x2 * y2]
            labels[i][x1 * y1 * x2 * y2] = 0
            if len(_pos_inds) > 0:
                score[_pos_inds] = pos_ious

            scores.append(score)

        # import pdb
        # pdb.set_trace()

        scores = torch.cat(scores)

        anchors = anchors.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)

        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)

        if self.reg_decoded_bbox:
            anchors = anchors.reshape(-1, 4)
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)

        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)

        # import pdb
        # pdb.set_trace()

        loss_cls = self.loss_cls(
            cls_score, (labels, scores),
            weight=label_weights,
            avg_factor=num_total_samples)

        return loss_cls, loss_bbox

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            gt_bboxes=gt_bboxes,
            num_total_samples=num_total_samples)
        # print(num_total_samples)
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)
