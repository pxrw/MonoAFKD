import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from lib.models.fusion import Fusion
from lib.models.MonoLSS import MonoLSS

from lib.losses.head_distill_loss import compute_head_distill_loss
from lib.losses.feature_distill_loss import compute_backbone_l1_loss
from lib.losses.feature_distill_loss import compute_backbone_resize_affinity_loss
from lib.losses.frequency_loss import TotalLoss

class MonoAFKD(nn.Module):
    def __init__(self, backbone='dla34', neck='DLAUp', downsample=4, flag='training', mean_size=None,
                 model_type='distill', cfg=None):
        assert downsample in [4, 8, 16, 32]
        super(MonoAFKD, self).__init__()

        self.cfg = cfg
        self.centernet_rgb = MonoLSS(backbone=backbone, neck=neck, mean_size=mean_size, model_type=model_type)
        self.centernet_depth = MonoLSS(backbone=backbone, neck=neck, mean_size=mean_size, model_type=model_type)

        for i in self.centernet_depth.parameters():
            i.requires_grad = False

        channels = self.centernet_rgb.backbone.channels  # [16, 32, 64,  128, 256, 512 ]
        tea_channels = self.centernet_depth.backbone.channels  # [16, 32, 128, 256, 512, 1024]
        input_channels = channels[2:]
        out_channels = channels[2:]
        mid_channel = channels[-1]

        rgb_fs = nn.ModuleList()
        for idx, in_channel in enumerate(input_channels):
            rgb_fs.append(Fusion(in_channel, mid_channel, out_channels[idx], idx < len(input_channels) - 1))
        self.rgb_fs = rgb_fs[::-1]

        self.align_freq_loss = TotalLoss(channels[2:])

        self.adapt_list = ['adapt_layer4', 'adapt_layer8', 'adapt_layer16', 'adapt_layer32']
        for i, adapt_name in enumerate(self.adapt_list):
            fc = nn.Sequential(
                nn.Conv2d(channels[i + 2], tea_channels[i + 2], kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(tea_channels[i + 2], tea_channels[i + 2], kernel_size=1, padding=0, bias=True)
            )
            self.__setattr__(adapt_name, fc)

        self.flag = flag

    def norm(self, feat: torch.Tensor) -> torch.Tensor:
        assert len(feat.shape) == 4
        N, C, H, W = feat.shape
        feat = feat.permute(1, 0, 2, 3).reshape(C, -1)
        mean = feat.mean(dim=-1, keepdim=True)
        std = feat.std(dim=-1, keepdim=True)
        feat = (feat - mean) / (std + 1e-6)
        return feat.reshape(C, N, H, W).permute(1, 0, 2, 3)

    def forward(self, input, coord_ranges, calibs, target=None, K=50, mode='train'):
        if mode == 'train' and target != None:
            rgb = input['rgb']
            depth = input['depth']

            rgb_feat, rgb_outputs, rgb_fs_feat = self.centernet_rgb(rgb, coord_ranges, calibs, target, K, mode)
            depth_feat, depth_outputs, depth_fs_feat = self.centernet_depth(depth, coord_ranges, calibs, target, K, mode)

            rgb_fs_feat = rgb_fs_feat[::-1]
            depth_fs_feat = depth_fs_feat[::-1]
            distill_feature = []
            for i, adapt in enumerate(self.adapt_list):
                distill_feature.append(self.__getattr__(adapt)(rgb_fs_feat[i]))

            head_loss, _ = compute_head_distill_loss(rgb_outputs, depth_outputs, target)
            align_freq_loss = self.align_freq_loss(distill_feature, depth_fs_feat)
            backbone_loss_affinity = compute_backbone_resize_affinity_loss(distill_feature, depth_fs_feat)
            return rgb_outputs, backbone_loss_affinity, head_loss, align_freq_loss
        else:
            rgb = input['rgb']
            rgb_feat, rgb_outputs, fusion_features = self.centernet_rgb(rgb, coord_ranges, calibs, target, mode='val')
            return rgb_feat, rgb_outputs, rgb_outputs

    def fill_fc_weights(self, layers):
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
