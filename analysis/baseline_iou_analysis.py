import sys
import pickle

import torch
import numpy as np

sys.path.insert(0, '../analysis')
from lib import CocoDataset, BboxResize, AnchorGenerator, bbox_overlaps


def get_area_index(areaRng_info):

    def index(i):
        return np.concatenate(np.where(areaRng_info == i), 0)

    area_flag = {
        # '8small': index(1),
        # '10small': index(2),
        # '12small': index(3),
        # '14small': index(4),
        # '16small': index(5),
        # '24small': index(6),
        # '32small': index(7),
        # 'medium': index(8),
        # 'large': index(9)
        'small': index(1),
        'medium': index(2),
        'large': index(3)
    }
    return area_flag


def max_iou_analysis_area_based():

    # data = CocoDataset(ann_file='data/widerface_train.json')
    # resize = BboxResize(img_scale=(1333, 800))
    data = CocoDataset(ann_file='data/widerface_fulltrain.json', min_size=9)
    resize = BboxResize(ratio=1)
    anchor_generator = AnchorGenerator(
        strides=[4, 8, 16, 32, 64],
        octave_base_scale=4,
        scales_per_octave=3,
        ratios=[1.0])

    areaRng_info = [img['ann_info']['areaRng_index'] for img in data]
    areaRng_info = np.concatenate(areaRng_info, axis=0)
    area_index = get_area_index(areaRng_info)

    bboxes = []
    for img in data:
        if len(img['ann_info']['bboxes']) == 0:
            continue
        else:
            bboxes.append(torch.from_numpy((resize(img))['bboxes']))

    # bboxes = [torch.from_numpy((resize(img))['bboxes']) for img in data]
    bboxes = torch.cat(bboxes, 0).cuda()

    w = (bboxes[:, 2] - bboxes[:, 0]).view(-1, 1)
    h = (bboxes[:, 3] - bboxes[:, 1]).view(-1, 1)
    gts = torch.cat((-w/2, -h/2, w/2, h/2), 1)

    multi_level_anchors = anchor_generator.gen_base_anchors()

    anchors = torch.cat(multi_level_anchors, 0).cuda()

    iou = bbox_overlaps(gts, anchors)

    max_iou, max_index = torch.max(iou, 1)

    max_iou_array = max_iou.cpu().detach().numpy()
    max_iou_array_8small = max_iou_array[area_index['8small']]
    max_iou_array_10small = max_iou_array[area_index['10small']]
    max_iou_array_12small = max_iou_array[area_index['12small']]
    max_iou_array_14small = max_iou_array[area_index['14small']]
    max_iou_array_16small = max_iou_array[area_index['16small']]
    max_iou_array_24small = max_iou_array[area_index['24small']]
    max_iou_array_32small = max_iou_array[area_index['32small']]
    max_iou_array_medium = max_iou_array[area_index['medium']]
    max_iou_array_large = max_iou_array[area_index['large']]

    with open('max_iou_train_area_based.pkl', 'wb') as f:
        pickle.dump(max_iou_array, f)
    with open('max_iou_train_area_based_small.pkl', 'wb') as f:
        pickle.dump(max_iou_array_small, f)
    with open('max_iou_train_area_based_medium.pkl', 'wb') as f:
        pickle.dump(max_iou_array_medium, f)
    with open('max_iou_train_area_based_large.pkl', 'wb') as f:
        pickle.dump(max_iou_array_large, f)


def max_iou_analysis_spatial_based():

    data = CocoDataset(ann_file='data/widerface_train.json')#, min_size=9)
    resize = BboxResize(ratio=1)
    # anchor_generator = AnchorGenerator(
    #     strides=[4, 8, 16, 32, 64, 128],
    #     octave_base_scale=2 ** (4 / 3),
    #     scales_per_octave=3,
    #     ratios=[1.3])

    # anchor_generator = AnchorGenerator(
    #     octave_base_scale=4,
    #     scales_per_octave=3,
    #     ratios=[0.5, 1.0, 2.0],
    #     strides=[8, 16, 32, 64, 128])

    anchor_generator = AnchorGenerator(
        octave_base_scale=4,
        scales_per_octave=3,
        ratios=[1.0],
        strides=[4, 8, 16, 32, 64])

    areaRng_info = [img['ann_info']['areaRng_index'] for img in data]
    areaRng_info = np.concatenate(areaRng_info, axis=0)

    # import pdb
    # pdb.set_trace()

    area_index = get_area_index(areaRng_info)

    max_iou_list = []
    max_index_valid_sum = 0

    for img in data:

        if len(img['ann_info']['bboxes']) == 0:
            continue

        result = resize(img)

        gts = torch.from_numpy(result['bboxes']).cuda()

        image_size = result['img_new_shape']
        # strides = [4, 8, 16, 32, 64, 128]
        # strides = [8, 16, 32, 64, 128]
        strides = [4, 8, 16, 32, 64]
        featmap_sizes = [(image_size/stride).astype('int') for stride in strides]

        multi_level_anchors = anchor_generator.grid_anchors(
                    featmap_sizes)
        multi_level_flags = anchor_generator.valid_flags(
            featmap_sizes, result['pad_shape'])

        anchors = torch.cat(multi_level_anchors, 0)

        iou = bbox_overlaps(gts, anchors)

        max_iou, max_index = torch.max(iou, 1)

        max_iou_list.append(max_iou)

        max_index_valid_sum += len(set(max_index.cpu().detach().numpy()))

    max_iou_tensor = torch.cat(max_iou_list, 0)

    unique_ratio = max_index_valid_sum/len(max_iou_tensor)
    print(f'unique_ratio: {unique_ratio}')

    max_iou_array = max_iou_tensor.cpu().detach().numpy()
    # max_iou_array_8small = max_iou_array[area_index['8small']]
    # max_iou_array_10small = max_iou_array[area_index['10small']]
    # max_iou_array_12small = max_iou_array[area_index['12small']]
    # max_iou_array_14small = max_iou_array[area_index['14small']]
    # max_iou_array_16small = max_iou_array[area_index['16small']]
    # max_iou_array_24small = max_iou_array[area_index['24small']]
    # max_iou_array_32small = max_iou_array[area_index['32small']]
    max_iou_array_small = max_iou_array[area_index['small']]
    max_iou_array_medium = max_iou_array[area_index['medium']]
    max_iou_array_large = max_iou_array[area_index['large']]

    with open('max_iou_train_spatial_based.pkl', 'wb') as f:
        pickle.dump(max_iou_array, f)
    # with open('max_iou_train_spatial_based_8small.pkl', 'wb') as f:
    #     pickle.dump(max_iou_array_8small, f)
    # with open('max_iou_train_spatial_based_10small.pkl', 'wb') as f:
    #     pickle.dump(max_iou_array_10small, f)
    # with open('max_iou_train_spatial_based_12small.pkl', 'wb') as f:
    #     pickle.dump(max_iou_array_12small, f)
    # with open('max_iou_train_spatial_based_14small.pkl', 'wb') as f:
    #     pickle.dump(max_iou_array_14small, f)
    # with open('max_iou_train_spatial_based_16small.pkl', 'wb') as f:
    #     pickle.dump(max_iou_array_16small, f)
    # with open('max_iou_train_spatial_based_24small.pkl', 'wb') as f:
    #     pickle.dump(max_iou_array_24small, f)
    # with open('max_iou_train_spatial_based_32small.pkl', 'wb') as f:
    #     pickle.dump(max_iou_array_32small, f)
    with open('max_iou_train_spatial_based_small.pkl', 'wb') as f:
        pickle.dump(max_iou_array_small, f)
    with open('max_iou_train_spatial_based_medium.pkl', 'wb') as f:
        pickle.dump(max_iou_array_medium, f)
    with open('max_iou_train_spatial_based_large.pkl', 'wb') as f:
        pickle.dump(max_iou_array_large, f)


if __name__ == '__main__':

    import time
    start = time.time()
    max_iou_analysis_spatial_based()
    print(f'spatial run time: {time.time() - start}')
    # start2 = time.time()
    # max_iou_analysis_area_based()
    # print(f'area run time: {time.time() - start2}')
