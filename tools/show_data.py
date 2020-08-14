import argparse

import cv2
import numpy as np

from mmcv import Config
from mmdet.datasets import build_dataset



def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    datasets = build_dataset(cfg.data.train)

    for i, data in enumerate(datasets):
        print(i)

        img = data['img'].data
        bbox = data['gt_bboxes'].data.numpy()
        ignore_bbox = data['gt_bboxes_ignore'].data.numpy()

        mean = (123.675, 116.280, 103.530)
        std = (1., 1., 1.)
        mean = np.reshape(np.array(mean, dtype=np.float32), [1, 1, 3])
        std = np.reshape(np.array(std, dtype=np.float32), [1, 1, 3])
        denominator = np.reciprocal(std, dtype=np.float32)
        ximg = (img.numpy().transpose(1, 2, 0) / denominator + mean).astype(np.uint8)
        ximg = ximg[:, :, (2, 1, 0)]

        cv2.imwrite('img/ximg_%d.png' % (i), ximg)
        iimg = cv2.imread('img/ximg_%d.png' % (i))

        for j in range(len(bbox)):
            x1, y1, x2, y2 = bbox[j]
            cv2.rectangle(iimg, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        for j in range(len(ignore_bbox)):
            x1, y1, x2, y2 = ignore_bbox[j]
            cv2.rectangle(iimg, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)

        if len(ignore_bbox) > 0:
            cv2.imwrite('img/iimg_%d.png' % (i), iimg)

if __name__ == '__main__':
    main()
