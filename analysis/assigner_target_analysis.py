import sys
import pickle

import torch
import numpy as np

sys.path.insert(0, '../analysis')
from lib import CocoDataset, BboxResize, AnchorGenerator, bbox_overlaps, MaxIoUAssigner, DeltaXYWHBBoxCoder


def get_area_index(areaRng_info):

    def index(i):
        return np.concatenate(np.where(areaRng_info == i), 0)

    area_flag = {
        '8small': index(1),
        '10small': index(2),
        '12small': index(3),
        '14small': index(4),
        '16small': index(5),
        '24small': index(6),
        '32small': index(7),
        'medium': index(8),
        'large': index(9)
        # 'small': index(1),
        # 'medium': index(2),
        # 'large': index(3)
    }
    return area_flag


def assigner_iou_analysis():

    # data = CocoDataset(ann_file='/DATA/home/chenhaowang/Datasets/COCO2017/annotations/instances_train2017.json')
    # resize = BboxResize(img_scale=(1333, 800))
    data = CocoDataset(ann_file='data/widerface_fulltrain.json', min_size=9)
    resize = BboxResize(ratio=1)
    anchor_generator = AnchorGenerator(
        strides=[4, 8, 16, 32, 64, 128],
        octave_base_scale=2 ** (4 / 3),
        scales_per_octave=3,
        ratios=[1.3])
    assigner = MaxIoUAssigner(
        pos_iou_thr=0.5,
        neg_iou_thr=0.3,
        min_pos_iou=0,
        ignore_iof_thr=-1
    )
    bbox_coder = DeltaXYWHBBoxCoder(
        target_means=(0., 0., 0., 0.),
        target_stds=(1., 1., 1., 1.))

    nums_list = []
    nums_large_list = []
    nums_medium_list = []
    nums_8small_list = []
    nums_10small_list = []
    nums_12small_list = []
    nums_14small_list = []
    nums_16small_list = []
    nums_24small_list = []
    nums_32small_list = []

    targets = []

    for img in data:

        result = resize(img)

        # get gt #################
        gt_bboxes = torch.from_numpy(result['bboxes']).cuda()
        gt_bboxes_ignore = torch.from_numpy(img['ann_info']['bboxes_ignore']).cuda()
        gt_labels = torch.from_numpy(img['ann_info']['labels']).cuda()

        # get area info ##########
        areaRng_info = img['ann_info']['areaRng_index']
        area_index = get_area_index(areaRng_info)

        # get anchor #############
        image_size = result['img_new_shape']
        strides = [4, 8, 16, 32, 64, 128]
        featmap_sizes = [(image_size / stride).astype('int') for stride in strides]
        multi_level_anchors = anchor_generator.grid_anchors(featmap_sizes)
        # multi_level_flags = anchor_generator.valid_flags(
        #     featmap_sizes, result['pad_shape'])
        anchors = torch.cat(multi_level_anchors, 0).cuda()

        # get assign result
        assign_result = assigner.assign(anchors,
                                        gt_bboxes,
                                        gt_bboxes_ignore,
                                        gt_labels)

        ####
        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
        pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds, :]
        pos_bbox_targets = bbox_coder.encode(
            anchors[pos_inds], pos_gt_bboxes)

        targets.append(pos_bbox_targets.cpu().numpy())

        # import pdb
        # pdb.set_trace()

        # # calculate assigned anchor number for each gt
        # gt_inds = assign_result.gt_inds.cpu().detach().numpy()
        # gt_inds = gt_inds[gt_inds != -1]
        # nums = np.bincount(gt_inds)[1:]
        # nums_list.append(nums)
        # nums_large_list.append(nums[area_index['large']])
        # nums_medium_list.append(nums[area_index['medium']])
        # nums_8small_list.append(nums[area_index['8small']])
        # nums_10small_list.append(nums[area_index['10small']])
        # nums_12small_list.append(nums[area_index['12small']])
        # nums_14small_list.append(nums[area_index['14small']])
        # nums_16small_list.append(nums[area_index['16small']])
        # nums_24small_list.append(nums[area_index['24small']])
        # nums_32small_list.append(nums[area_index['32small']])

    targets = np.concatenate(targets)

    import pdb
    pdb.set_trace()

    # num_all = np.concatenate(nums_list, axis=0)
    # num_large = np.concatenate(nums_large_list, axis=0)
    # num_medium = np.concatenate(nums_medium_list, axis=0)
    # num_8small = np.concatenate(nums_small_list, axis=0)
    # num_10small = np.concatenate(nums_small_list, axis=0)
    # num_12small = np.concatenate(nums_small_list, axis=0)
    # num_14small = np.concatenate(nums_small_list, axis=0)
    # num_16small = np.concatenate(nums_small_list, axis=0)
    # num_24small = np.concatenate(nums_small_list, axis=0)
    # num_32small = np.concatenate(nums_small_list, axis=0)

    # print(f"AVG: all: {num_all.mean()}, large: {num_large.mean()}, "
    #       f"medium: {num_medium.mean()}, small: {num_small.mean()}")
    # print(f"MIN: all: {num_all.min()}, large: {num_large.min()}, "
    #       f"medium: {num_medium.min()}, small: {num_small.min()}")
    # print(f"MAX: all: {num_all.max()}, large: {num_large.max()}, "
    #       f"medium: {num_medium.max()}, small: {num_small.max()}")
    # print(f"MED: all: {np.median(num_all)}, large: {np.median(num_large)}, "
    #       f"medium: {np.median(num_medium)}, small: {np.median(num_small)}")
    # print(f"MODE: all: {np.argmax(np.bincount(num_all))}, large: {np.argmax(np.bincount(num_large))}, "
    #       f"medium: {np.argmax(np.bincount(num_medium))}, small: {np.argmax(np.bincount(num_small))}")

    # num = [num_all, num_large, num_medium, num_small]
    # num = [num_all, num_large, num_medium, num_8small, num_10small, num_12small, num_14small, num_16small, num_24small, num_32small]
    # for i, limit in enumerate(['all', 'large', 'medium', '8small', '10small', '12small', '14small', '16small', '24small', '32small']):
    #     with open('train_anchor_num_per_gt_' + limit + '.pkl', 'wb') as f:
    #         pickle.dump(num[i], f)


if __name__ == '__main__':

    import time
    start = time.time()
    assigner_iou_analysis()
    print(f'Run time: {time.time() - start}')
