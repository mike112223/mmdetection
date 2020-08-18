import os.path as osp

import numpy as np
from pycocotools.coco import COCO


class CocoDataset(object):

    # CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    #            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    #            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    #            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    #            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    #            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    #            'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    #            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    #            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    #            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    #            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    #            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    #            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    #            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

    CLASSES = ('face')


    def __init__(self,
                 ann_file,
                 data_root=None,
                 test_mode=False,
                 min_size=0,
                 filter_empty_gt=True):
        self.ann_file = ann_file
        self.data_root = data_root
        self.test_mode = test_mode
        self.min_size = min_size
        self.filter_empty_gt = filter_empty_gt

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
                self.seg_prefix = osp.join(self.data_root, self.seg_prefix)
            if not (self.proposal_file is None
                    or osp.isabs(self.proposal_file)):
                self.proposal_file = osp.join(self.data_root,
                                              self.proposal_file)
        # load annotations (and proposals)
        self.data_infos = self.load_annotations(self.ann_file)

        # filter images too small
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.data_infos = [self.data_infos[i] for i in valid_inds]

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds(self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.getImgIds()
        data_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
        return data_infos

    def get_ann_info(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(img_id)
        ann_info = self.coco.loadAnns(ann_ids)
        return self._parse_ann_info(self.data_infos[idx], ann_info)

    def get_cat_ids(self, idx):
        """Get COCO category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(img_id)
        ann_info = self.coco.loadAnns(ann_ids)
        return [ann['category_id'] for ann in ann_info]

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.data_infos):
            if self.filter_empty_gt and self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def get_area_index(self, areas):
        # self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 8 ** 2], [8 ** 2, 10 ** 2],
                        [10 ** 2, 12 ** 2], [12 ** 2, 14 ** 2], [14 ** 2, 16 ** 2],
                        [16 ** 2, 24 ** 2], [24 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', '8small', '10small', '12small', '14small', '16small',
                           '24small', '32small', 'small' 'medium', 'large']
        # self.areaRngLbl = ['small' 'medium', 'large']
        index = [[self.areaRng.index(areaRng) for areaRng in self.areaRng[1:] if areaRng[0] < area <= areaRng[1]][0]
                 for area in areas]
        return np.array(index)

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_areas = []
        gt_bboxes_ignore = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            w -= x1
            h -= y1
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if inter_w < self.min_size or inter_h < self.min_size:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                if not (ann['area'] <= 0 or w < 1 or h < 1):
                    gt_areas.append(w*h)

        areaRng_index = self.get_area_index(gt_areas)

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            areaRng_index=areaRng_index,
            bboxes_ignore=gt_bboxes_ignore,
            seg_map=seg_map)

        return ann

    def __call__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                True).
        """

        return self.prepare_train_img(idx)

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)

        return results

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        return self.prepare_train_img(idx)

    def __iter__(self):
        self.__total = self.__len__()
        self.__it = -1
        return self

    def __next__(self):
        if self.__total - 1 > self.__it:
            self.__it += 1
            return self.__getitem__(self.__it)
        else:
            raise StopIteration


class BboxResize(object):

    def __init__(self, img_scale=None, ratio=None):
        self.img_scale = img_scale
        self.ratio = ratio
        self.keep_ratio = True
        self.results = {}

    def _resize_img(self, img_data):

        img_shape = (img_data['img_info']['height'], img_data['img_info']['width'])

        if self.img_scale is not None:
            img_scale = self.img_scale
        else:
            img_scale = (img_shape[0] * self.ratio, img_shape[1] * self.ratio)

        new_size, scale_factor = self.imrescale(img_shape, img_scale)

        new_h, new_w = new_size
        h, w = img_data['img_info']['height'], img_data['img_info']['width']
        h_scale = new_h / h
        w_scale = new_w / w
        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                dtype=np.float32)

        self.results['img_ori_shape'] = np.array(img_shape)
        self.results['img_new_shape'] = np.array(new_size)
        self.results['pad_shape'] = np.array(new_size)
        self.results['scale_factor'] = scale_factor

    def _resize_bboxes(self, img_data):
        """Resize bounding boxes with ``results['scale_factor']``."""
        img_shape = self.results['img_new_shape']
        bboxes = img_data['ann_info']['bboxes']
        bboxes = bboxes * self.results['scale_factor']
        bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
        bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
        self.results['bboxes'] = bboxes

    def _scale_size(self, size, scale):
        """Rescale a size by a ratio.

        Args:
            size (tuple): w, h.
            scale (float): Scaling factor.

        Returns:
            tuple[int]: scaled size.
        """
        h, w = size
        return int(h * float(scale) + 0.5), int(w * float(scale) + 0.5)

    def imrescale(self, img_shape, scale):

        h, w = img_shape

        max_long_edge = max(scale)
        max_short_edge = min(scale)
        scale_factor = min(max_long_edge / max(h, w),
                           max_short_edge / min(h, w))

        new_size = self._scale_size((h, w), scale_factor)
        return new_size, scale_factor

    def __call__(self, img_data):

        self._resize_img(img_data)
        self._resize_bboxes(img_data)
        return self.results
