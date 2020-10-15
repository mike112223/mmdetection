import torch

from ..builder import BBOX_ASSIGNERS
from ..iou_calculators import build_iou_calculator
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


@BBOX_ASSIGNERS.register_module()
class PropIouAssigner(BaseAssigner):
    """Assign pixels at the center region of a bbox as positive.

    Each proposals will be assigned with `-1`, `0`, or a positive integer
    indicating the ground truth index.
    - -1: negative samples
    - semi-positive numbers: positive sample, index (0-based) of assigned gt

    Args:
        pos_scale (float): Threshold within which pixels are
          labelled as positive.
        neg_scale (float): Threshold above which pixels are
          labelled as positive.
        min_pos_iof (float): Minimum iof of a pixel with a gt to be
          labelled as positive. Default: 1e-2
        ignore_gt_scale (float): Threshold within which the pixels
          are ignored when the gt is labelled as shadowed. Default: 0.5
        foreground_dominate (bool): If True, the bbox will be assigned as
          positive when a gt's kernel region overlaps with another's shadowed
          (ignored) region, otherwise it is set as ignored. Default to False.
    """

    def __init__(self,
                 min_pos_iou=1e-2,
                 iou_calculator=dict(type='BboxOverlaps2D'),
                 gpu_assign_thr=-1):
        self.min_pos_iou = min_pos_iou
        self.iou_calculator = build_iou_calculator(iou_calculator)
        self.gpu_assign_thr = gpu_assign_thr

    def assign(self, bboxes, gt_bboxes, gt_labels=None):
        """Assign gt to bboxes.

        This method assigns gts to every bbox (proposal/anchor), each bbox \
        will be assigned with -1, or a semi-positive number. -1 means \
        negative sample, semi-positive number is the index (0-based) of \
        assigned gt.

        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (tensor, optional): Ground truth bboxes that are
              labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (tensor, optional): Label of gt_bboxes, shape (num_gts,).

        Returns:
            :obj:`AssignResult`: The assigned result. Note that \
              shadowed_labels of shape (N, 2) is also added as an \
              `assign_result` attribute. `shadowed_labels` is a tensor \
              composed of N pairs of anchor_ind, class_label], where N \
              is the number of anchors that lie in the outer region of a \
              gt, anchor_ind is the shadowed anchor index and class_label \
              is the shadowed class label.

        Example:
            >>> self = CenterRegionAssigner(0.2, 0.2)
            >>> bboxes = torch.Tensor([[0, 0, 10, 10], [10, 10, 20, 20]])
            >>> gt_bboxes = torch.Tensor([[0, 0, 10, 10]])
            >>> assign_result = self.assign(bboxes, gt_bboxes)
            >>> expected_gt_inds = torch.LongTensor([1, 0])
            >>> assert torch.all(assign_result.gt_inds == expected_gt_inds)
        """
        # There are in total 5 steps in the pixel assignment
        # 1. Find core (the center region, say inner 0.2)
        #     and shadow (the relatively ourter part, say inner 0.2-0.5)
        #     regions of every gt.
        # 2. Find all prior bboxes that lie in gt_core and gt_shadow regions
        # 3. Assign prior bboxes in gt_core with a one-hot id of the gt in
        #      the image.
        #    3.1. For overlapping objects, the prior bboxes in gt_core is
        #           assigned with the object with smallest area
        # 4. Assign prior bboxes with class label according to its gt id.
        #    4.1. Assign -1 to prior bboxes lying in shadowed gts
        #    4.2. Assign positive prior boxes with the corresponding label
        # 5. Find pixels lying in the shadow of an object and assign them with
        #      background label, but set the loss weight of its corresponding
        #      gt to zero.
        assert bboxes.size(1) == 4, 'bboxes must have size of 4'
        assign_on_cpu = True if (self.gpu_assign_thr > 0) and (
            gt_bboxes.shape[0] > self.gpu_assign_thr) else False

        if assign_on_cpu:
            device = gt_bboxes.device
            bboxes = bboxes.cpu()
            gt_bboxes = gt_bboxes.cpu()
            gt_labels = gt_labels.cpu()

        overlaps = self.iou_calculator(
            bboxes, gt_bboxes)

        num_gts, num_bboxes = gt_bboxes.size(0), bboxes.size(0)

        assigned_gt_ids = overlaps.new_zeros((num_bboxes,), dtype=torch.long)

        max_ious = None
        if num_gts > 0 and num_bboxes > 0:
            max_ious, assigned_gt_ind = overlaps.max(axis=1)
            matched_mask = max_ious > self.min_pos_iou
            assigned_gt_ids[matched_mask] = assigned_gt_ind[matched_mask] + 1

        # 4. Assign prior bboxes with class label according to its gt id.
        assigned_labels = None
        if gt_labels is not None:
            # Default assigned label is the background (-1)
            assigned_labels = assigned_gt_ids.new_full((num_bboxes, ), -1)
            pos_inds = torch.nonzero(
                assigned_gt_ids > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[assigned_gt_ids[pos_inds] - 1]

        if assign_on_cpu:
            assigned_gt_ids = assigned_gt_ids.to(device)
            assigned_labels = assigned_labels.to(device)
            max_ious = max_ious.to(device)

        # import pdb
        # pdb.set_trace()

        return AssignResult(
            num_gts, assigned_gt_ids, max_ious, labels=assigned_labels)
