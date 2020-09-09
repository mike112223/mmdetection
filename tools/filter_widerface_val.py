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
            obj.find('difficult').text = '0'

    tree.write(xml_path)


def _parse_xmls(ann_path):

    xml_paths = os.listdir(ann_path)
    # print(len(xml_paths))

    for xml_path in xml_paths:
        print(xml_path)
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

    _parse_xmls(args.ann_path)

    (facebox_list, event_list, file_list, hard_gt_list,
        medium_gt_list, easy_gt_list) = get_gt_boxes(args.gt_path)

    event_num = len(event_list)
    settings = ['easy', 'medium', 'hard']
    setting_gts = [easy_gt_list, medium_gt_list, hard_gt_list]

    for setting_id in range(0, 3):
        # different setting
        gt_list = setting_gts[setting_id]
        count_face = 0
        # [hard, medium, easy]
        pbar = tqdm.tqdm(range(event_num))
        for i in pbar:
            pbar.set_description('Processing {}'.format(settings[setting_id]))
            img_list = file_list[i][0]
            sub_gt_list = gt_list[i][0]
            # img_pr_info_list = np.zeros((len(img_list), thresh_num, 2))
            gt_bbx_list = facebox_list[i][0]

            for j in range(len(img_list)):

                xml_path = os.path.join(args.ann_path, str(img_list[j][0][0]) + '.xml')

                gt_boxes = gt_bbx_list[j][0].astype('float')

                keep_index = sub_gt_list[j][0]
                count_face += len(keep_index)

                parse_xml(xml_path, keep_index, gt_boxes)

        print(count_face)

if __name__ == '__main__':
    main()
