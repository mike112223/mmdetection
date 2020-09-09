"""
WiderFace evaluation code
author: wondervictor
mail: tianhengcheng@gmail.com
copyright@wondervictor
"""

import os
import argparse

import tqdm
import numpy as np
import xml.etree.ElementTree as ET
from scipy.io import loadmat
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('--gt_path', help='test config file path')
    parser.add_argument('--ann_path', help='test config file path')

    args = parser.parse_args()
    return args


def get_gt_boxes(gt_dir):
    """ gt dir: (wider_face_val.mat, wider_easy_val.mat, wider_medium_val.mat, wider_hard_val.mat)"""

    gt_mat = loadmat(os.path.join(gt_dir, 'wider_face_val.mat'))
    hard_mat = loadmat(os.path.join(gt_dir, 'wider_hard_val.mat'))
    medium_mat = loadmat(os.path.join(gt_dir, 'wider_medium_val.mat'))
    easy_mat = loadmat(os.path.join(gt_dir, 'wider_easy_val.mat'))

    facebox_list = gt_mat['face_bbx_list']
    event_list = gt_mat['event_list']
    file_list = gt_mat['file_list']

    hard_gt_list = hard_mat['gt_list']
    medium_gt_list = medium_mat['gt_list']
    easy_gt_list = easy_mat['gt_list']

    return facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list


def filter(gt_boxes):
    keep = gt_boxes[:, 2] >= 0 * (gt_boxes[:, 3] >= 0)
    return gt_boxes[keep]


def parse_xml(xml_path, keep_index, gt_boxes):

    tree = ET.parse(xml_path)
    root = tree.getroot()
    objs = root.findall('object')

    keep_index = keep_index.reshape(-1)

    gt_boxes = filter(gt_boxes)

    tmp_boxes = []

    for i in range(len(objs)):

        obj = objs[i]
        gt_box = gt_boxes[i]

        bnd_box = obj.find('bndbox')
        # TODO: check whether it is necessary to use int
        # Coordinates may be float type
        bbox = np.array([
            int(float(bnd_box.find('xmin').text)),
            int(float(bnd_box.find('ymin').text)),
            int(float(bnd_box.find('xmax').text)),
            int(float(bnd_box.find('ymax').text))
        ])

        bbox[2:] -= bbox[:2]

        assert np.sum(bbox == gt_box) == 4, (bbox, gt_box)

        if i + 1 in keep_index:
            tmp_boxes.append(gt_box)

    return tmp_boxes

    # tree.write(xml_path)


def _parse_xmls(ann_path):

    xml_paths = os.listdir(ann_path)
    print(len(xml_paths))

    for xml_path in xml_paths:
        tree = ET.parse(os.path.join(ann_path, xml_path))
        root = tree.getroot()
        objs = root.findall('object')

        for obj in objs:

            obj.find('difficult').text = '1'

        tree.write(os.path.join(ann_path, xml_path))

    # import pdb
    # pdb.set_trace()


def main():
    args = parse_args()

    # _parse_xmls(args.ann_path)

    (facebox_list, event_list, file_list, hard_gt_list,
        medium_gt_list, easy_gt_list) = get_gt_boxes(args.gt_path)

    xxx = {}

    event_num = len(event_list)
    settings = ['easy', 'medium', 'hard']
    setting_gts = [easy_gt_list, medium_gt_list, hard_gt_list]

    for setting_id in range(3):
        # different setting
        keep_bboxes = []

        xxx[setting_id] = {}

        gt_list = setting_gts[setting_id]
        count_face = 0
        # [easy, medium, hard]
        pbar = tqdm.tqdm(range(event_num))
        for i in pbar:
            pbar.set_description('Processing {}'.format(settings[setting_id]))
            img_list = file_list[i][0]
            sub_gt_list = gt_list[i][0]
            # img_pr_info_list = np.zeros((len(img_list), thresh_num, 2))
            gt_bbx_list = facebox_list[i][0]

            xxx[setting_id][i] = {}

            for j in range(len(img_list)):

                xml_path = os.path.join(args.ann_path, str(img_list[j][0][0]) + '.xml')

                gt_boxes = gt_bbx_list[j][0].astype('float')

                # gt_boxes = filter(gt_boxes)

                keep_index = sub_gt_list[j][0]

                xxx[setting_id][i][j] = keep_index.reshape(-1).tolist()
                count_face += len(keep_index)

                keep_bbox = parse_xml(xml_path, keep_index, gt_boxes)

                if len(keep_bbox) > 0:
                    keep_bboxes.append(np.concatenate([keep_bbox]))

        print(count_face)

        # import pdb
        # pdb.set_trace()

        keep_bboxes = np.concatenate(keep_bboxes)

        # draw_img_hw(keep_bboxes[:, 3], keep_bboxes[:, 2])
        # draw_ins_hw(keep_bboxes[:, 3], keep_bboxes[:, 2])
        sarea = np.sqrt(keep_bboxes[:, 2] * keep_bboxes[:, 3])

        # import pdb
        # pdb.set_trace
        area_ranges = [(0, 8), (8, 16), (16, 24), (24, 32), (32, 96), (96, 1000000)]
        area_txt = ['0-8', '8-16', '16-24', '24-32', '32-96', '>96']
        bins = [0, 0, 0, 0, 0, 0]
        for k, (min_area, max_area) in enumerate(area_ranges):
            bins[k] = ((sarea >= min_area) * (sarea < max_area)).sum()

        plt.figure(setting_id)
        x = np.arange(6)
        for a, b in zip(x, bins):
            plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=10)
        plt.bar(area_txt, bins)

        # draw_hist(keep_bboxes[:, 3])
        print('w', np.min(keep_bboxes[:, 2]), np.mean(keep_bboxes[:, 2]))
        print('h', np.min(keep_bboxes[:, 3]), np.mean(keep_bboxes[:, 3]))
        print('a', np.min(sarea), np.mean(sarea))

    plt.show()

    # l1 = len(xxx[0])
    # for i in range(l1):
    #     l2 = len(xxx[0][i])
    #     for j in range(l2):
    #         x = xxx[0][i][j]
    #         for k in range(1, 3):
    #             assert len(xxx[k][i][j]) == len(set(x + xxx[k][i][j])), (k, i, j, x, xxx[k][i][j])
    #             x = xxx[k][i][j]

def draw_img_hw(img_hs, img_ws, **kargs):
    img_hs = np.asarray(img_hs)
    img_ws = np.asarray(img_ws)

    plt.figure()
    plt.scatter(img_ws, img_hs, **kargs)
    plt.title('img(knife)')
    plt.xlabel('width')
    plt.ylabel('height')
    plt.show()

def draw_ins_hw(ins_hs, ins_ws, **kargs):
    ins_hs = np.asarray(ins_hs)
    ins_ws = np.asarray(ins_ws)

    plt.figure()
    plt.scatter(ins_ws, ins_hs, **kargs)
    plt.title('instance(knife)')
    plt.xlabel('width')
    plt.ylabel('height')
    plt.show()

def draw_hist(ins_pixs, **kargs):
    bins = min(int(len(ins_pixs) / 40), 60)

    plt.figure()
    plt.hist(ins_pixs, bins=bins, **kargs)
    plt.title('instance')
    plt.xlabel('area sqrt')
    plt.show()


if __name__ == '__main__':
    main()
