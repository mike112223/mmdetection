import torch

from ..builder import BBOX_ASSIGNERS
from ..iou_calculators import build_iou_calculator
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


def scale_boxes(bboxes, scale):
    """Expand an array of boxes by a given scale.

    Args:
        bboxes (Tensor): Shape (m, 4)
        scale (float): The scale factor of bboxes

    Returns:
        (Tensor): Shape (m, 4). Scaled bboxes
    """
    assert bboxes.size(1) == 4
    w_half = (bboxes[:, 2] - bboxes[:, 0]) * .5
    h_half = (bboxes[:, 3] - bboxes[:, 1]) * .5
    x_c = (bboxes[:, 2] + bboxes[:, 0]) * .5
    y_c = (bboxes[:, 3] + bboxes[:, 1]) * .5

    w_half *= scale
    h_half *= scale

    boxes_scaled = torch.zeros_like(bboxes)
    boxes_scaled[:, 0] = x_c - w_half
    boxes_scaled[:, 2] = x_c + w_half
    boxes_scaled[:, 1] = y_c - h_half
    boxes_scaled[:, 3] = y_c + h_half
    return boxes_scaled


def is_located_in(points, bboxes):
    """Are points located in bboxes.

    Args:
      points (Tensor): Points, shape: (m, 2).
      bboxes (Tensor): Bounding boxes, shape: (n, 4).

    Return:
      Tensor: Flags indicating if points are located in bboxes, shape: (m, n).
    """
    assert points.size(1) == 2
    assert bboxes.size(1) == 4
    return (points[:, 0].unsqueeze(1) > bboxes[:, 0].unsqueeze(0)) & \
           (points[:, 0].unsqueeze(1) < bboxes[:, 2].unsqueeze(0)) & \
           (points[:, 1].unsqueeze(1) > bboxes[:, 1].unsqueeze(0)) & \
           (points[:, 1].unsqueeze(1) < bboxes[:, 3].unsqueeze(0))


def bboxes_area(bboxes):
    """Compute the area of an array of bboxes.

    Args:
        bboxes (Tensor): The coordinates ox bboxes. Shape: (m, 4)

    Returns:
        Tensor: Area of the bboxes. Shape: (m, )
    """
    assert bboxes.size(1) == 4
    w = (bboxes[:, 2] - bboxes[:, 0])
    h = (bboxes[:, 3] - bboxes[:, 1])
    areas = w * h
    return areas


@BBOX_ASSIGNERS.register_module()
class PropCenterRegionAssigner(BaseAssigner):
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
                 pos_scale,
                 min_pos_iou=1e-2,
                 iou_calculator=dict(type='BboxOverlaps2D'),
                 gpu_assign_thr=-1):
        self.pos_scale = pos_scale
        self.min_pos_iou = min_pos_iou
        self.iou_calculator = build_iou_calculator(iou_calculator)
        self.gpu_assign_thr = gpu_assign_thr

    def assign(self, anchors, bboxes, gt_bboxes, gt_labels=None):
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

        # 1. Find core positive and shadow region of every gt
        gt_core = scale_boxes(gt_bboxes, self.pos_scale)

        # 2. Find prior bboxes that lie in gt_core and gt_shadow regions
        anchors_centers = (anchors[:, 2:4] + anchors[:, 0:2]) / 2
        # The center points lie within the gt boxes
        is_anchor_in_gt_core = is_located_in(anchors_centers, gt_core)
        # Only calculate bbox and gt_core IoF. This enables small prior bboxes
        #   to match large gts

        if assign_on_cpu:
            device = gt_bboxes.device
            bboxes = bboxes.cpu()
            is_anchor_in_gt_core = is_anchor_in_gt_core.cpu()
            gt_core = gt_core.cpu()
            gt_labels = gt_labels.cpu()

        bbox_and_gt_core_overlaps = self.iou_calculator(
            bboxes, gt_core)

        idx_and_assigned_gt_anchor_in_gt_core = is_anchor_in_gt_core.nonzero()
        idx_anchor_in_gt_core = idx_and_assigned_gt_anchor_in_gt_core[:, 0].reshape(-1)
        assigned_gt_anchor_in_gt_core = idx_and_assigned_gt_anchor_in_gt_core[:, 1].reshape(-1)

        num_gts, num_bboxes = gt_bboxes.size(0), bboxes.size(0)

        assigned_gt_ids = \
            is_anchor_in_gt_core.new_zeros((num_bboxes,),
                                           dtype=torch.long)

        max_ious = None
        if num_gts > 0 and num_bboxes > 0:
            max_ious, assigned_gt_ind = bbox_and_gt_core_overlaps.max(axis=1)
            matched_mask = (assigned_gt_ind[idx_anchor_in_gt_core] ==
                            assigned_gt_anchor_in_gt_core) & (max_ious[idx_anchor_in_gt_core] >
                                                              self.min_pos_iou)
            assigned_gt_ids[idx_anchor_in_gt_core[matched_mask]] = assigned_gt_ind[idx_anchor_in_gt_core[matched_mask]] + 1

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
