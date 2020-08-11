
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init

from ..builder import NECKS


@NECKS.register_module()
class SSH(nn.Module):
    def __init__(self,
                 in_channel,
                 num_levels,
                 num_layers=3,
                 conv_cfg=None,
                 norm_cfg=None):
        super(SSH, self).__init__()
        assert in_channel % 2**(num_layers - 1) == 0

        self.in_channel = in_channel
        self.num_levels = num_levels
        self.num_layers = num_layers
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.level_ssh_convs = nn.ModuleList()

        for i in range(self.num_levels):
            in_channel = self.in_channel
            ssh_convs = nn.ModuleList()
            for j in range(self.num_layers):
                if j < self.num_layers - 1:
                    output_channel = in_channel // 2
                else:
                    output_channel = in_channel

                ssh_conv = ConvModule(
                    in_channel,
                    output_channel,
                    3,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg)

                ssh_convs.append(ssh_conv)
                in_channel = output_channel

            self.level_ssh_convs.append(ssh_convs)

    def init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, input):

        outs = []
        for i in range(self.num_levels):
            x = input[i]
            out = []
            for j in range(self.num_layers):
                x = self.level_ssh_convs[i][j](x)
                out.append(x)
            out = F.relu(torch.cat(out, dim=1))
            outs.append(out)

        return tuple(outs)
