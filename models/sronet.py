import torch
import torch.nn as nn
import torch.nn.functional as F
from models.galerkin import simple_attn
from models import register
from utils import make_coord

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        self.norm1 = nn.InstanceNorm2d(planes)
        self.norm2 = nn.InstanceNorm2d(planes)
        if not stride == 1:
            self.norm3 = nn.InstanceNorm2d(planes)

        self.downsample = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, padding=0, stride=stride))

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class BasicEncoder256(nn.Module):
    def __init__(self, output_dim=64, norm_fn='instance', dropout=0.0):
        super(BasicEncoder256, self).__init__()

        self.norm_fn = norm_fn
        self.norm1 = nn.InstanceNorm2d(64)
        self.repPad = nn.ReplicationPad2d(1)  # 因为kernel_size=1，所以需要填充1个像素
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=0)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=1)
        self.layer3 = self._make_layer(128, stride=1)

        # output convolution
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)
        x = self.repPad(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x



class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=256, input_dim=256):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))


    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        return h
@register('piv_gt')
class PIVNO(nn.Module):

    def __init__(self, width=128, blocks=16,iters=5):
        super().__init__()
        self.iters = iters
        self.width = width
        self.snet = BasicEncoder256(output_dim=64, norm_fn='instance', dropout=0.)

        self.conv0 = simple_attn(self.width, blocks)
        self.conv1 = simple_attn(self.width, blocks)
        self.fc1 = nn.Conv2d(self.width*4+4*2+2, 64, kernel_size=3, stride=1, padding=1)
        self.fc2 = nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1)
        # self.fc1 = nn.Conv2d(self.width*4+4*2+2, 64, kernel_size=1)
        # self.fc2 = nn.Conv2d(64, 2, kernel_size=1)
        self.gru = SepConvGRU(hidden_dim=128, input_dim=128)

    def query_rgb(self, feat):

        x = self.conv0(feat, 0)
        x = self.conv1(x, 1)

        return x

    def spatial_interpolation(self, feat, coord, cell):

        pos_lr = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2
        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6

        rel_coords = []
        feat_s = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                #
                coord_ = coord.clone()
                coord_[:, :, :, 0] += vx * rx + eps_shift
                coord_[:, :, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                #   根据偏移后坐标 coord_ 从特征图 feat 中采样到的邻域特征，表示当前位置的特征值。
                # feat_ = feat
                feat_ = F.grid_sample(feat, coord_.flip(-1), mode='nearest', align_corners=False)

                old_coord = F.grid_sample(pos_lr, coord_.flip(-1), mode='nearest', align_corners=False)
                rel_coord = coord.permute(0, 3, 1, 2) - old_coord
                rel_coord[:, 0, :, :] *= feat.shape[-2]
                rel_coord[:, 1, :, :] *= feat.shape[-1]

                area = torch.abs(rel_coord[:, 0, :, :] * rel_coord[:, 1, :, :])
                areas.append(area + 1e-9)
                # rel_coord是坐标差，是高分辨率点与最近4个点的坐标差
                rel_coords.append(rel_coord)
                feat_s.append(feat_)

        rel_cell = cell.clone()
        rel_cell[:, 0] *= feat.shape[-2]
        rel_cell[:, 1] *= feat.shape[-1]

        tot_area = torch.stack(areas).sum(dim=0)
        t = areas[0]; areas[0] = areas[3];areas[3] = t
        t = areas[1];areas[1] = areas[2];areas[2] = t

        for index, area in enumerate(areas):
            feat_s[index] = feat_s[index] * (area / tot_area).unsqueeze(1)

        grid = torch.cat([*rel_coords, *feat_s, \
                          rel_cell.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, coord.shape[1], coord.shape[2])], dim=1)
        return grid


    def forward(self, img1, img2, hr_coord, cell):
        # 位移特征的获取
        smap1, smap2 = self.snet([img1, img2])
        smap = torch.cat((smap1, smap2), dim=1)
        smap = self.query_rgb(smap)
        flow_future = smap.clone()
        flow_predictions = []

        for itr in range(self.iters):
            flow_future = self.gru(flow_future, smap)
            hl_future = self.spatial_interpolation(flow_future, hr_coord, cell)
            flow = self.fc2(F.gelu(self.fc1(hl_future)))
            flow_predictions.append(flow)

        return flow_predictions