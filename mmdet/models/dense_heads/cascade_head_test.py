import torch
import torch.nn as nn

from mmdet.core import (bbox2result, bbox2roi, bbox_mapping, build_assigner,
                        build_sampler, merge_aug_bboxes, multiclass_nms, build_anchor_generator)
from ..builder import HEADS, build_head


def reverse_sigmoid(x):
    return - torch.log((1 - x) / x)


@HEADS.register_module()
class CascadeTestHead(nn.Module):
    """Cascade roi head including one bbox head and one mask head.

    https://arxiv.org/abs/1712.00726
    """

    def __init__(self,
                 num_stages,
                 num_classes,
                 stage_loss_weights,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     scales=[8, 16, 32],
                     ratios=[0.5, 1.0, 2.0],
                     strides=[4, 8, 16, 32, 64]),
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None):
        super(CascadeTestHead, self).__init__()
        assert num_stages > 1
        assert bbox_head is not None
        assert len(bbox_head) == num_stages
        self.num_stages = num_stages
        self.num_classes = num_classes
        self.stage_loss_weights = stage_loss_weights

        self.anchor_generator = build_anchor_generator(anchor_generator)
        self.init_bbox_head(bbox_head, train_cfg, test_cfg)

    def init_bbox_head(self, bbox_head, train_cfg, test_cfg):
        """Initialize box head and box roi extractor.

        Args:
            bbox_roi_extractor (dict): Config of box roi extractor.
            bbox_head (dict): Config of box in box head.
        """
        if train_cfg is None:
            train_cfg = [None for _ in range(self.num_stages)]

        assert isinstance(train_cfg, list)
        assert isinstance(test_cfg, list)

        self.bbox_head = nn.ModuleList()
        for head, _train_cfg, _test_cfg in zip(bbox_head, train_cfg, test_cfg):
            head.update(train_cfg=_train_cfg)
            head.update(test_cfg=_test_cfg)
            self.bbox_head.append(build_head(head))

    def init_weights(self):
        """Initialize the weights in head.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        for i in range(self.num_stages):
            self.bbox_head[i].init_weights()

    def get_anchors(self, featmap_sizes, img_metas, device='cuda'):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.
            device (torch.device | str): Device for returned tensors

        Returns:
            tuple:
                anchor_list (list[Tensor]): Anchors of each image.
                valid_flag_list (list[Tensor]): Valid flags of each image.
        """
        num_imgs = len(img_metas)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = self.anchor_generator.valid_flags(
                featmap_sizes, img_meta['pad_shape'], device)
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list

    def forward(self, x):
        cls_scores, bbox_preds = [], []
        for i in range(self.num_stages):
            cls_score, bbox_pred = self.bbox_head[i](x)
            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)
        return cls_scores, bbox_preds

    def _bbox_forward_train(self, stage, x, bboxes, valid_flags, img_metas, gt_bboxes,
                            gt_labels=None, gt_bboxes_ignore=None):
        """Run forward function and calculate loss for box head in training."""
        num_img = len(img_metas)
        bbox_head = self.bbox_head[stage]
        # do not support caffe_c4 model anymore
        outs = bbox_head(x)
        if gt_labels is None:
            loss_inputs = outs + (bboxes, valid_flags, gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (bboxes, valid_flags, gt_bboxes, gt_labels, img_metas)
        losses, proposals_list = bbox_head.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        proposal_list = [[] for _ in range(num_img)]
        for i in range(num_img):
            for props in proposals_list:
                pre_level_prop = len(props)
                div = pre_level_prop // num_img
                proposal_list[i].append(props[div * i: div * (i + 1)])

        return losses, proposal_list

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in x]
        assert len(featmap_sizes) == self.anchor_generator.num_levels
        device = x[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)

        # anchor (num img, num level)
        bbox_list = anchor_list
        losses = dict()
        for i in range(self.num_stages):
            lw = self.stage_loss_weights[i]

            # bbox head forward and loss
            loss, proposal_list = self._bbox_forward_train(
                i, x, bbox_list, valid_flag_list, img_metas,
                gt_bboxes, gt_labels, gt_bboxes_ignore)

            for name, value in loss.items():
                losses[f's{i}.{name}'] = [v * lw for v in value]

            bbox_list = proposal_list

        return losses

    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   img_metas,
                   cfg=None,
                   rescale=False):

        # "ms" in variable names means multi-stage
        assert len(cls_scores) == len(bbox_preds)
        for cls_score, bbox_pred in zip(cls_scores, bbox_preds):
            assert len(cls_score) == len(bbox_pred)

        num_levels = len(cls_scores[0])
        device = cls_scores[0][0].device
        featmap_sizes = [cls_scores[0][i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device=device)

        sum_scores = [cls_scores[0][_].sigmoid() for _ in range(num_levels)]
        bboxes = mlvl_anchors
        for i in range(self.num_stages):
            cls_score = cls_scores[i]
            bbox_pred = bbox_preds[i]

            if i > 0:
                for l in range(num_levels):
                    sum_scores[l] += cls_score[l].sigmoid()

            if i < self.num_stages - 1:
                proposals = []
                for l in range(num_levels):
                    _bbox_pred = bbox_pred[l].permute(0, 2, 3, 1).reshape(-1, 4)
                    proposal = self.bbox_head[i].bbox_coder.decode(bboxes[l], _bbox_pred)
                    proposals.append(proposal)

                bboxes = proposals

        # get final score & bbox
        cls_scores = [reverse_sigmoid(sum_scores[i] / self.num_stages) for i in range(num_levels)]
        bbox_list = self.bbox_head[-1].get_bboxes(
            cls_scores,
            bbox_preds[-1],
            bboxes,
            img_metas,
            cfg=cfg,
            rescale=rescale)

        return bbox_list
