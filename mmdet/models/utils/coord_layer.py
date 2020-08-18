
import torch
import torch.nn as nn


class CoordLayer(nn.Module):
    """ResLayer to build ResNet style backbone.

    Args:
        block (nn.Module): block used to build ResLayer.
        inplanes (int): inplanes of block.
        planes (int): planes of block.
        num_blocks (int): number of blocks.
        stride (int): stride of the first block. Default: 1
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        downsample_first (bool): Downsample at the first block or last block.
            False for Hourglass, True for ResNet. Default: True
    """

    def __init__(self, with_r=False):
        self.with_r = with_r
        super(CoordLayer, self).__init__()

    def __call__(self, feats):

        flag = False
        if not isinstance(feats, tuple):
            feats = [feats]
            flag = True

        outputs = []
        for feat in feats:

            x_range = torch.linspace(-1, 1, feat.shape[-1], device=feat.device)
            y_range = torch.linspace(-1, 1, feat.shape[-2], device=feat.device)
            y, x = torch.meshgrid(y_range, x_range)
            y = y.expand([feat.shape[0], 1, -1, -1])
            x = x.expand([feat.shape[0], 1, -1, -1])
            coord_feat = torch.cat([x, y], 1)
            feat = torch.cat([feat, coord_feat], 1)

            if self.with_r:
                rr = torch.sqrt(torch.square(x) + torch.square(y))
                feat = torch.cat([feat, rr], 1)

            outputs.append(feat)

        if flag:
            return outputs[0]
        else:
            return tuple(outputs)
