import functools

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.RepVit import repvit_m0_9
from lib.CrossNorm import cn_op_2ins_space_chan_v2


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class ConvBR(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1):
        super(ConvBR, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class MS_CAM(nn.Module):

    def __init__(self, channels=64, r=8):
        super(MS_CAM, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        return wei * x


class BAM(nn.Module):
    def __init__(self, channel):
        super(BAM, self).__init__()

        self.edge_conv0 = BasicConv2d(48, channel, kernel_size=3, stride=1, padding=1)
        self.edge_conv1 = BasicConv2d(384, channel, kernel_size=3, stride=1, padding=1)
        self.edge_conv2 = BasicConv2d(128, channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x1, x4):
        size = x1.size()[2:]
        edge2 = self.edge_conv0(x1)
        edge3 = F.interpolate(x4, size, mode='bilinear', align_corners=True)
        edge3 = self.edge_conv1(edge3)

        out = torch.cat((edge2, edge3), dim=1)
        out = self.edge_conv2(out)
        return out


class CFM(nn.Module):
    def __init__(self, channels, crop='neither', beta=None):
        self.init__ = super(CFM, self).__init__()

        self.cn_op = functools.partial(cn_op_2ins_space_chan_v2,
                                       crop=crop, beta=beta)

        self.layer1 = BasicConv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.layer2 = BasicConv2d(channels, channels, kernel_size=3, stride=1, padding=1)

        self.layer2_1 = BasicConv2d(channels, channels // 4, kernel_size=3, stride=1, padding=1)
        self.layer2_2 = BasicConv2d(channels, channels // 4, kernel_size=3, stride=1, padding=1)

        self.layer_fu = BasicConv2d(channels // 2, channels, kernel_size=3, stride=1, padding=1)

        self.layer_out = BasicConv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x1, x2):

        x_cn1, x_cn2 = self.cn_op(x1, x2)

        x1_1 = self.layer1(x1) + x_cn1
        x2_2 = self.layer2(x2) + x_cn2

        fuse = self.layer_fu(torch.cat((self.layer2_1(x1_1), self.layer2_2(x2_2)), dim=1))

        out = self.layer_out(fuse)

        return out


class EGAM0(nn.Module):
    def __init__(self, channel=64):
        super(EGAM0, self).__init__()
        self.conv_1 = ConvBR(channel, channel, kernel_size=3, stride=1, padding=1)
        self.conv_l = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1)

        # 执行积分注意力操作
        self.attention = MS_CAM()

        self.convfuse1 = nn.Sequential(
            BasicConv2d(channel * 2, channel, kernel_size=3, stride=1, padding=1),
            BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x1, edge):
        x1_1 = self.conv_1(x1)
        x1_2 = self.conv_l(x1_1)

        fuse = self.convfuse1(torch.cat((x1_2, edge), 1))
        # fuse and Integrity enhence
        out = self.attention(fuse)

        return out


class EGAM(nn.Module):
    def __init__(self, channel=64):
        super(EGAM, self).__init__()
        self.conv_1 = ConvBR(channel, channel, kernel_size=3, stride=1, padding=1)
        self.conv_2 = ConvBR(channel, channel, kernel_size=3, stride=1, padding=1)
        self.conv_3 = ConvBR(channel, channel, kernel_size=3, stride=1, padding=1)
        self.conv_d1 = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.conv_l = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1)

        # 执行积分注意力操作
        self.attention = MS_CAM()
        self.convfuse1 = nn.Sequential(
            BasicConv2d(channel * 3, channel, kernel_size=3, stride=1, padding=1),
            BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x1, x2, edge):
        x1_1 = self.conv_1(x1)
        x2_1 = self.conv_2(x2)

        x2_2 = self.conv_d1(x2_1)
        x1_2 = self.conv_l(x1_1)

        # 计算x1对x2的积分关注
        if x2_2.size()[2:] != x1.size()[2:]:
            x2_2 = F.interpolate(x2_2, size=x1.size()[2:], mode='bilinear', align_corners=True)
        z1 = F.relu(x1_1 * x2_2, inplace=True)

        # 计算x2对x1的积分关注
        if x2_1.size()[2:] != x1.size()[2:]:
            x2_1 = F.interpolate(x2_1, size=x1.size()[2:], mode='bilinear', align_corners=True)
        z2 = F.relu(x2_1 * x1_2, inplace=True)

        fuse = self.convfuse1(torch.cat((z1, z2, edge), 1))
        # fuse and Integrity enhence

        out = self.attention(fuse)

        return out


class Network(nn.Module):
    def __init__(self, channel=64):
        super(Network, self).__init__()

        self.repvit = self.initialize_weights()

        self.downSample = nn.MaxPool2d(2, stride=2)

        self.bam = BAM(channel)

        self.conv_1 = BasicConv2d(48, channel, kernel_size=3, stride=1, padding=1)
        self.conv_2 = BasicConv2d(96, channel, kernel_size=3, stride=1, padding=1)
        self.conv_3 = BasicConv2d(192, channel, kernel_size=3, stride=1, padding=1)
        self.conv_4 = BasicConv2d(384, channel, kernel_size=3, stride=1, padding=1)

        self.cfm1 = CFM(channel)
        self.cfm2 = CFM(channel)
        self.cfm3 = CFM(channel)

        self.fusion1 = EGAM0(channel)
        self.fusion2 = EGAM(channel)
        self.fusion3 = EGAM(channel)

        ##
        self.edge_conv3 = BasicConv2d(channel, 1, kernel_size=3, padding=1)

        ##
        self.layer_out3 = nn.Sequential(nn.Conv2d(channel, 1, kernel_size=3, stride=1, padding=1))
        self.layer_out2 = nn.Sequential(nn.Conv2d(channel, 1, kernel_size=3, stride=1, padding=1))
        self.layer_out1 = nn.Sequential(nn.Conv2d(channel, 1, kernel_size=3, stride=1, padding=1))

        self.up_2 = lambda x: F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        self.up_4 = lambda x: F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        self.up_8 = lambda x: F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=True)
        self.up_16 = lambda x: F.interpolate(x, scale_factor=16, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.repvit.stem(x)
        stage_outputs = []
        for stage in self.repvit.stages:
            x = stage(x)
            stage_outputs.append(x)

        # features
        x1 = stage_outputs[0]  # (BS, 48, 56, 56)
        x2 = stage_outputs[1]  # (BS, 96, 28, 28)
        x3 = stage_outputs[2]  # (BS, 192, 14, 14)
        x4 = stage_outputs[3]  # (BS, 384, 7, 7)

        # edge
        edge_guidance = self.bam(x1, x4)
        edge_out = self.up_4(self.edge_conv3(edge_guidance))  # (BS, 1, 224, 224)

        #
        x1_rf = self.conv_1(x1)  # (BS, 64, 56, 56)
        x2_rf = self.conv_2(x2)  # (BS, 64, 28, 28)
        x3_rf = self.conv_3(x3)  # (BS, 64, 14, 14)
        x4_rf = self.conv_4(x4)  # (BS, 64, 7, 7)

        # FeaFusion
        x3_fe = self.cfm1(self.up_2(x4_rf), x3_rf)  # (BS, 64, 14, 14)
        x2_fe = self.cfm2(self.up_2(x3_rf), x2_rf)  # (BS, 64, 28, 28)
        x1_fe = self.cfm3(self.up_2(x2_rf), x1_rf)  # (BS, 64, 56, 56)

        # Fusion
        x1_fusion = self.fusion1(x3_fe,
                                 F.interpolate(edge_guidance, scale_factor=1 / 4, mode='bilinear', align_corners=True))  # (BS, 64, 14, 14)
        out3 = self.up_16(self.layer_out3(x1_fusion))

        x2_fusion = self.fusion2(x2_fe, self.up_2(x1_fusion),
                                 F.interpolate(edge_guidance, scale_factor=1 / 2, mode='bilinear', align_corners=True))  # (BS, 64, 28, 28)
        out2 = self.up_8(self.layer_out2(x2_fusion))

        x3_fusion = self.fusion3(x1_fe, self.up_2(x2_fusion), edge_guidance)  # (BS, 64, 56, 56)
        out1 = self.up_4(self.layer_out1(x3_fusion))

        # out
        return out3, out2, out1, edge_out

    def initialize_weights(self):
        model_name = 'repvit_m0_9.dist_300e_in1k'
        # 加载模型
        model = timm.create_model(
            model_name,
            pretrained=False,  # 设置为False，因为我们将从本地加载
            num_classes=1000,  # 根据您的需要设置分类数量
        )
        # 加载预训练权重
        model.load_state_dict(torch.load("repvit_m0_9.dist_300e_in1k/pytorch_model.bin"))

        # 删除指定模块
        if 'head_drop' in model._modules:
            del model._modules['head_drop']
        if 'head' in model._modules:
            del model._modules['head']

        return model
