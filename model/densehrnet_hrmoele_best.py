# This implementation is based on the DenseNet-BC implementation in torchvision
# https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict

class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        offset = self.p_conv(x)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset

class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        # res卷积
        self.res_conv = nn.Conv2d(in_dim, in_dim,
                                 kernel_size=1, stride=1, bias=False)
        # 下采样
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
        )
        self.upsample = nn.Sequential(
            #nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
        )
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=1, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=1, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B,C,X,W,X,H)
            returns :
                out : self attention value + input feature
                attention: B, N, N (N is Width*Height)
        """
        res = x
        res = self.res_conv(res)
        x = self.maxpool_conv(x)
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B, N,C
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B,C,N
        energy = torch.bmm(proj_query, proj_key)  # 批次乘
        attention = self.softmax(energy)  # B,N,N
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B,C,N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        out = F.interpolate(out,size=(res.size()[2],res.size()[3]),mode='bilinear')
        out = self.upsample(out)
        out = self.gamma * out + res
        out = self.conv(out)
        return out, attention


def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function


class double_conv(nn.Module):
    def __init__(self, indim=3,outdim = 3, drop_rate = 0.1):
        super(double_conv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(indim),
            nn.ReLU(inplace=True),
            nn.Conv2d(indim, indim*2, kernel_size=3, stride=1, padding=1,groups=indim, bias=False),
            nn.BatchNorm2d(indim*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(indim*2, outdim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Dropout(drop_rate)
        )
    def forward(self, x):
        out = self.conv1(x)
        return out


class hrmodel(nn.Module):
    def __init__(self, indim=3):
        super(hrmodel, self).__init__()
        self.res = nn.Sequential(
            DeformConv2d(indim, indim, 3, padding=1, modulation=True),
            # nn.Conv2d(indim, indim, 3, padding=1, bias=False),
            nn.BatchNorm2d(indim),
            nn.ReLU(inplace=True)
        )
        # self.feature_map = self.res
        # self.res.register_forward_hook(self._get_res_hook())
        self.conv1 = double_conv(indim,indim)
        self.maxpool_conv1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(indim, indim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(indim),
            nn.ReLU(inplace=True),
        )
        self.conv2 = double_conv(indim*2,indim*2)
        self.conv3 = double_conv(indim*4,indim*4)
        self.conv1_1 = double_conv(indim,indim*2)
        self.maxpool_conv2 = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(indim,indim)
        )
        self.conv1_2 = double_conv(indim*2,indim*2)
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(indim*4, indim*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(indim*2),
            nn.ReLU(inplace=True)
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(indim*5, indim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(indim),
            nn.ReLU(inplace=True)
        )

    # def _get_res_hook(self):
    #     def hook(module, input, output):
    #         self.feature_map = output.detach()
    #     return hook

    def forward(self, x):
        res = self.res(x)
        # self.feature_map = res
        x1 = self.conv1(x)
        x1_1 = self.maxpool_conv1(x1)
        x1_1_up = F.interpolate(x1_1,size=(x.size()[2],x.size()[3]),mode='bilinear')
        x2 = self.conv2(torch.cat([x1, x1_1_up], dim=1))
        x2_1 = self.maxpool_conv2(x1_1)
        x2_1_up = F.interpolate(x2_1,size=(x1_1.size()[2],x1_1.size()[3]),mode='bilinear')
        x1_2 = self.conv1_2(torch.cat([x1_1, x2_1_up], dim=1))
        x1_2_up = F.interpolate(x1_2,size=(x.size()[2],x.size()[3]),mode='bilinear')
        x3 = self.conv3(torch.cat([x2,x1_2_up],dim=1))
        # x3 = x3+res
        x3 = torch.cat([x3,res],dim = 1)
        x3 = self.conv_out(x3)
        
        return x3

class ECABlock(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECABlock, self).__init__()
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        v = self.avg_pool(x)
        w = self.max_pool(x)
        v = v + w
        v = self.conv(v.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        v = self.sigmoid(v)
        return x * v


class SEBlock(nn.Module):
    def __init__(self, mode, channels, ratio):
        super(SEBlock, self).__init__()
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_pooling = nn.AdaptiveMaxPool2d(1)
        if mode == "max":
            self.global_pooling = self.max_pooling
        elif mode == "avg":
            self.global_pooling = self.avg_pooling
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=channels, out_features=channels // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=channels // ratio, out_features=channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        v = self.global_pooling(x).view(b, c)
        v = self.fc_layers(v).view(b, c, 1, 1)
        v = self.sigmoid(v)
        return x * v


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, out_features=32, drop_rate=0, efficient=False):
        super(_DenseLayer, self).__init__()
        # res
        self.deform = nn.Sequential(
            nn.Conv2d(num_input_features, 32, kernel_size=1, stride=1, bias=False),
            hrmodel(32)
        )
        self.dbconv = double_conv(num_input_features,num_input_features)
        # 1
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(num_input_features, out_features, kernel_size=1, stride=1, bias=False),
            nn.Dropout(0.1)
        )
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(out_features, out_features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Dropout(0.1)
        )
        # 2
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(num_input_features + out_features, out_features, kernel_size=1, stride=1, bias=False),
            nn.Dropout(0.1)
        )
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(out_features, out_features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Dropout(0.1)
        )
        # 3
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(num_input_features + out_features * 2, out_features, kernel_size=1, stride=1, bias=False),
            nn.Dropout(0.1)
        )
        self.conv3_3 = nn.Sequential(
            nn.Conv2d(out_features, out_features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Dropout(0.1)
        )
        # 4

        self.conv1_4 = nn.Sequential(
            nn.Conv2d(num_input_features + out_features * 3, out_features, kernel_size=1, stride=1, bias=False),
            nn.Dropout(0.1)
        )
        self.conv3_4 = nn.Sequential(
            nn.Conv2d(out_features, out_features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Dropout(0.1)
        )

        self.outconv = nn.Sequential(
            nn.Conv2d(num_input_features*2+32, num_input_features*2+32, kernel_size=1, stride=1,groups = num_input_features*2+32, padding=0, bias=False),
            SEBlock('avg',num_input_features*2+32,1),
            nn.Conv2d(num_input_features*2+32, num_input_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Dropout(0.1)
        )
        self.bn = nn.BatchNorm2d(num_input_features)
        self.act = nn.LeakyReLU(0.3)

    def forward(self, x):
        # res
        res = self.dbconv(x)
        deform = self.deform(x)

        # 1
        x1 = self.conv1_1(x)
        x1 = self.conv3_1(x1)
        cat1 = torch.cat([x, x1], dim=1)
        # 2
        x2 = self.conv1_2(cat1)
        x2 = self.conv3_2(x2)
        cat2 = torch.cat([cat1, x2], dim=1)
        # 3
        x3 = self.conv1_3(cat2)
        x3 = self.conv3_3(x3)
        cat3 = torch.cat([cat2, x3], dim=1)
        # 4
        x4 = self.conv1_4(cat3)
        x4 = self.conv3_4(x4)
        # sum
        sum = torch.cat([res, deform,x4], dim=1)
        # out
        out = self.outconv(sum)
        out = self.bn(out)
        out = self.act(out)
        return out


class densehrnet(nn.Module):
    def __init__(self, drop_rate=0, num_classes=2):
        super(densehrnet, self).__init__()
        # block1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.denselayer1 = _DenseLayer(32, 32, 0)
        # block2
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.denselayer2 = _DenseLayer(64, 64, 0)
        # block3
        self.conv3 = nn.Sequential(
            SEBlock('avg',96,96),
            nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.denselayer3 = _DenseLayer(128, 128, 0)
        # block4
        self.conv4 = nn.Sequential(
            SEBlock('avg',192,192),
            nn.Conv2d(192, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.denselayer4 = _DenseLayer(256, 256, 0)
        self.outconv = nn.Sequential(
            SEBlock('avg',256,256),
            nn.Conv2d(256, num_classes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True))
        self.hrmodel = hrmodel(64)
    def forward(self, x):
        x1 = self.conv1(x)
        # self.feature_map = x1
        x1 = self.denselayer1(x1)
        # self.feathermap1 = x1
        x2 = self.conv2(x1)
        x2 = self.denselayer2(x2)
        # self.feature_map2 = x2
        # x2 = self.hrmodel(x2)
        # 3
        x3 = torch.cat([x1, x2], dim=1)
        x3_ = self.conv3(x3)
        x3_ = self.denselayer3(x3_)
        # self.feature_map3 = x3_
        # 4
        x4 = torch.cat([x2, x3_], dim=1)
        x4_ = self.conv4(x4)
        x4_ = self.denselayer4(x4_)
        # self.feature_map4 = x4_
        # 480
        # cat = torch.cat([x1,x2,x3,x4], dim=1)
        x5 = self.outconv(x4_)
        return x5
        # return x5


from torchsummary import summary

if __name__ == '__main__':
    img = torch.randn(1, 3, 64, 64).cuda()
    device = torch.device('cuda')
    model = densehrnet().to(device)
    output = model.forward(img)
    # visualize_feature_map_avg(model.feature_map)
    # summary(model,input_size=(3,64,64))
    # print(output.shape)
