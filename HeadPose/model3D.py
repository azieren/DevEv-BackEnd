import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import math
from backbone.repvgg import get_RepVGG_func_by_name
import utils

def inflate_conv(conv2d,
                 time_dim=3,
                 time_padding=0,
                 time_stride=1,
                 time_dilation=1,
                 center=False):
    # To preserve activations, padding should be by continuity and not zero
    # or no padding in time dimension
    kernel_dim = (time_dim, conv2d.kernel_size[0], conv2d.kernel_size[1])
    padding = (time_padding, conv2d.padding[0], conv2d.padding[1])
    stride = (time_stride, conv2d.stride[0], conv2d.stride[0])
    dilation = (time_dilation, conv2d.dilation[0], conv2d.dilation[1])
    conv3d = torch.nn.Conv3d(
        conv2d.in_channels,
        conv2d.out_channels,
        kernel_dim,
        padding=padding,
        dilation=dilation,
        stride=stride,
        groups =  conv2d.groups)
    # Repeat filter time_dim times along time dimension
    weight_2d = conv2d.weight.data
    if center:
        weight_3d = torch.zeros(*weight_2d.shape)
        weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        middle_idx = time_dim // 2
        weight_3d[:, :, middle_idx, :, :] = weight_2d
    else:
        weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        weight_3d = weight_3d / time_dim

    # Assign new params
    conv3d.weight = Parameter(weight_3d)
    conv3d.bias = conv2d.bias
    return conv3d


def inflate_linear(linear2d, time_dim):
    """
    Args:
        time_dim: final time dimension of the features
    """
    linear3d = torch.nn.Linear(linear2d.in_features * time_dim,
                               linear2d.out_features)
    weight3d = linear2d.weight.data.repeat(1, time_dim)
    weight3d = weight3d / time_dim

    linear3d.weight = Parameter(weight3d)
    linear3d.bias = linear2d.bias
    return linear3d


def inflate_batch_norm(batch2d):
    # In pytorch 0.2.0 the 2d and 3d versions of batch norm
    # work identically except for the check that verifies the
    # input dimensions

    batch3d = torch.nn.BatchNorm3d(batch2d.num_features)
    # retrieve 3d _check_input_dim function
    batch2d._check_input_dim = batch3d._check_input_dim
    return batch2d


def inflate_pool(pool2d,
                 time_dim=1,
                 time_padding=0,
                 time_stride=None,
                 time_dilation=1):
    if isinstance(pool2d, torch.nn.AdaptiveAvgPool2d):
        pool3d = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
    else:
        kernel_dim = (time_dim, pool2d.kernel_size, pool2d.kernel_size)
        padding = (time_padding, pool2d.padding, pool2d.padding)
        if time_stride is None:
            time_stride = time_dim
        stride = (time_stride, pool2d.stride, pool2d.stride)
        if isinstance(pool2d, torch.nn.MaxPool2d):
            dilation = (time_dilation, pool2d.dilation, pool2d.dilation)
            pool3d = torch.nn.MaxPool3d(
                kernel_dim,
                padding=padding,
                dilation=dilation,
                stride=stride,
                ceil_mode=pool2d.ceil_mode)
        elif isinstance(pool2d, torch.nn.AvgPool2d):
            pool3d = torch.nn.AvgPool3d(kernel_dim, stride=stride)
        else:
            raise ValueError('{} is not among known pooling classes'.format(type(pool2d)))

    return pool3d

class SixDRepNet3D(nn.Module):
    def __init__(self,
                 backbone_name, backbone_file, deploy,
                 bins=(1, 2, 3, 6),
                 droBatchNorm=nn.BatchNorm2d,
                 pretrained=True):
        super(SixDRepNet3D, self).__init__()

        repvgg_fn = get_RepVGG_func_by_name(backbone_name)
        backbone = repvgg_fn(deploy)
        if pretrained:
            checkpoint = torch.load(backbone_file)
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
            ckpt = {k.replace('module.', ''): v for k,
                    v in checkpoint.items()}  # strip the names
            backbone.load_state_dict(ckpt)

        self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = backbone.stage0, backbone.stage1, backbone.stage2, backbone.stage3, backbone.stage4
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)

        self.gap = inflate_pool(self.gap, time_dim=3, time_padding=1, time_stride=2)

        time_stride = 2
        for n, m in self.layer0.named_modules():
            if isinstance(m, nn.Conv2d): 
                m = inflate_conv(m)
            elif "rbr_reparam" in str(m) and isinstance(m.rbr_reparam, nn.Conv2d):
                m.rbr_reparam = inflate_conv(m.rbr_reparam, time_dim=3, time_padding=1, center=True, time_stride = time_stride)
                time_stride = 1
        
        time_stride = 2
        for m in self.layer1:
            if "rbr_reparam" in str(m) and isinstance(m.rbr_reparam, nn.Conv2d):
                m.rbr_reparam = inflate_conv(m.rbr_reparam, time_dim=3, time_padding=1, center=True, time_stride = time_stride)
                time_stride = 1

        time_stride = 2
        for m in self.layer2:
            if "rbr_reparam" in str(m) and isinstance(m.rbr_reparam, nn.Conv2d):
                m.rbr_reparam = inflate_conv(m.rbr_reparam, time_dim=3, time_padding=1, center=True, time_stride = time_stride)
                time_stride = 1

        for m in self.layer3:
            if "rbr_reparam" in str(m) and isinstance(m.rbr_reparam, nn.Conv2d):
                m.rbr_reparam = inflate_conv(m.rbr_reparam, time_dim=3, time_padding=1, center=True, time_stride = time_stride)
                time_stride = 1

        last_channel = 0
        for n, m in self.layer4.named_modules():
            if ('rbr_dense' in n or 'rbr_reparam' in n) and isinstance(m, nn.Conv2d):
                last_channel = m.out_channels

        for m in self.layer4:
            if "rbr_reparam" in str(m) and isinstance(m.rbr_reparam, nn.Conv2d):
                m.rbr_reparam = inflate_conv(m.rbr_reparam, time_dim=3, time_padding=1, center=True, time_stride = time_stride)
                time_stride = 1

        fea_dim = last_channel

        self.linear_reg = nn.Linear(fea_dim, 6)
        self.linear_reg = inflate_linear(self.linear_reg, 1)


    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x= self.gap(x)
        x = torch.flatten(x, 1)
        x = self.linear_reg(x)
        return utils.compute_rotation_matrix_from_ortho6d(x)

if __name__ == "__main__":

    model = SixDRepNet3D(backbone_name='RepVGG-B1g2', backbone_file='', deploy=True, pretrained=False)
    model = model.cuda()
    x = torch.zeros((32,3,16,224,224)).cuda()
    y = model(x)