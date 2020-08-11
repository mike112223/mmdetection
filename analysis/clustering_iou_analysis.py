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
        'small': index(1),
        'medium': index(2),
        'large': index(3)
    }
    return area_flag


def get_cluster(filename):
    f = open(filename, 'r')
    cluster = []

    for line in f:
        infos = line.split(" ")
        length = len(infos)
        for i in range(0, length):
            width = int(infos[i].split(",")[0])
            height = int(infos[i].split(",")[1])
            cluster.append([width, height])
    return np.array(cluster)


def clustering_max_iou_analysis_area_based():

    data = CocoDataset(ann_file='instances_val2017.json')
    resize = BboxResize(img_scale=(1333, 800))

    areaRng_info = [img['ann_info']['areaRng_index'] for img in data]
    areaRng_info = np.concatenate(areaRng_info, axis=0)
    area_index = get_area_index(areaRng_info)

    bboxes = [torch.from_numpy((resize(img))['bboxes']) for img in data]
    bboxes = torch.cat(bboxes, 0).cuda()

    w = (bboxes[:, 2] - bboxes[:, 0]).view(-1, 1)
    h = (bboxes[:, 3] - bboxes[:, 1]).view(-1, 1)
    gts = torch.cat((-w/2, -h/2, w/2, h/2), 1)

    # cluster 2 anchors
    cluster = torch.from_numpy(get_cluster('train10_yolo_anchors.txt')).float()

    w_a = cluster[:, 0].view(-1, 1)
    h_a = cluster[:, 1].view(-1, 1)

    anchors = torch.cat((-w_a/2, -h_a/2, w_a/2, h_a/2), 1).cuda()
    ############

    iou = bbox_overlaps(gts, anchors)

    max_iou, max_index = torch.max(iou, 1)

    max_iou_array = max_iou.cpu().detach().numpy()
    max_iou_array_small = max_iou_array[area_index['small']]
    max_iou_array_medium = max_iou_array[area_index['medium']]
    max_iou_array_large = max_iou_array[area_index['large']]

    with open('clustering_max_iou_train_area_based.pkl', 'wb') as f:
        pickle.dump(max_iou_array, f)
    with open('clustering_max_iou_train_area_based_small.pkl', 'wb') as f:
        pickle.dump(max_iou_array_small, f)
    with open('clustering_max_iou_train_area_based_medium.pkl', 'wb') as f:
        pickle.dump(max_iou_array_medium, f)
    with open('clustering_max_iou_train_area_based_large.pkl', 'wb') as f:
        pickle.dump(max_iou_array_large, f)


def clustering_max_iou_analysis_spatial_based(anno_path, cluster_path, p=3):
    data = CocoDataset(ann_file=anno_path)
    resize = BboxResize(img_scale=(1333, 800))
    anchor_generator = AnchorGenerator(
        strides=[[8, 16, 32, 64, 128][p-3]],
        ratios=[0.5, 1.0, 2.0],
        octave_base_scale=4,
        scales_per_octave=3,)

    # cluster 2 anchors
    cluster = torch.from_numpy(get_cluster(cluster_path)).float()

    w_a = cluster[:, 0].view(-1, 1)
    h_a = cluster[:, 1].view(-1, 1)

    anchor_generator.base_anchors = [torch.cat((-w_a/2, -h_a/2, w_a/2, h_a/2), 1).cuda()]
    ######

    areaRng_info = [img['ann_info']['areaRng_index'] for img in data]
    areaRng_info = np.concatenate(areaRng_info, axis=0)
    area_index = get_area_index(areaRng_info)

    max_iou_list = []
    max_index_valid_sum = 0

    for img in data:

        result = resize(img)

        gts = torch.from_numpy(result['bboxes']).cuda()

        image_size = result['img_new_shape']
        strides = [8, 16, 32, 64, 128]
        # featmap_sizes = [(image_size/stride).astype('int') for stride in strides]
        featmap_sizes = [(image_size / strides[p-3]).astype('int')]

        multi_level_anchors = anchor_generator.grid_anchors(
                    featmap_sizes)
        multi_level_flags = anchor_generator.valid_flags(
            featmap_sizes, result['pad_shape'])

        anchors = torch.cat(multi_level_anchors, 0)

        iou = bbox_overlaps(gts, anchors)

        max_iou, max_index = torch.max(iou, 1)

        max_index_valid_sum += len(set(max_index.cpu().detach().numpy()))

        max_iou_list.append(max_iou)

    max_iou_tensor = torch.cat(max_iou_list, 0)

    unique_ratio = max_index_valid_sum/len(max_iou_tensor)
    print(f'unique_ratio: {unique_ratio}')

    max_iou_array = max_iou_tensor.cpu().detach().numpy()
    max_iou_array_small = max_iou_array[area_index['small']]
    max_iou_array_medium = max_iou_array[area_index['medium']]
    max_iou_array_large = max_iou_array[area_index['large']]

    with open('clustering_max_iou_train_spatial_based.pkl', 'wb') as f:
        pickle.dump(max_iou_array, f)
    with open('clustering_max_iou_train_spatial_based_small.pkl', 'wb') as f:
        pickle.dump(max_iou_array_small, f)
    with open('clustering_max_iou_train_spatial_based_medium.pkl', 'wb') as f:
        pickle.dump(max_iou_array_medium, f)
    with open('clustering_max_iou_train_spatial_based_large.pkl', 'wb') as f:
        pickle.dump(max_iou_array_large, f)


if __name__ == '__main__':

    anno_path = '/DATA/home/chenhaowang/Datasets/COCO2017/annotations/instances_train2017.json'
    cluster_path = 'train9_yolo_anchors.txt'

    import time
    start = time.time()
    clustering_max_iou_analysis_spatial_based(p=4, anno_path=anno_path, cluster_path=cluster_path)
    print(f'run time: {time.time() - start}')
