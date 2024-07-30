import torch
import numpy as np


class InvBlock(torch.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, clamp=1.):
        super(InvBlock, self).__init__()
        self.H = None
        self.G = None
        self.F = None
        self.s = None
        self.split_len1 = channel_split_num
        self.split_len2 = channel_num - channel_split_num
        self.clamp = clamp
        self.construct(subnet_constructor)

    def jacobian(self, x, rev=False):
        if not rev:
            jac = torch.sum(self.s)
        else:
            jac = -torch.sum(self.s)
        return jac / x.shape[0]

    def forward(self, x, rev=False):
        a = x.narrow(1, 0, self.split_len1)
        b = x.narrow(1, self.split_len1, self.split_len2)
        if not rev:
            c = a + self.F(b)
            self.s = self.clamp * (torch.sigmoid(self.H(c)) * 2 - 1)
            d = b.mul(torch.exp(self.s)) + self.G(c)
        else:
            self.s = self.clamp * (torch.sigmoid(self.H(a)) * 2 - 1)
            d = (b - self.G(a)).div(torch.exp(self.s))
            c = a - self.F(d)
        return torch.cat((c, d), 1)

    def construct(self, subnet_constructor):
        self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.H = subnet_constructor(self.split_len1, self.split_len2)


class ConvGrey(torch.nn.Module):
    def __init__(self, rgb_type, learnable=True):
        super(ConvGrey, self).__init__()
        self.conv_weights = None
        self.channel_in = 3
        self.setWeights(learnable, rgb_type)

    def setWeights(self, learnable, rgb_type):
        self.conv_weights = torch.eye(self.channel_in)
        self.convWeights(rgb_type)
        self.conv_weights = torch.nn.Parameter(self.conv_weights)
        if not learnable:
            self.conv_weights.requires_grad = False

    def forward(self, x, rev=False):
        if not rev:
            conv_weights = self.conv_weights.reshape(self.channel_in, self.channel_in, 1, 1)
            return torch.nn.functional.conv2d(x, conv_weights, bias=None, stride=1)
        else:
            inv_weights = torch.inverse(self.conv_weights)
            inv_weights = inv_weights.reshape(self.channel_in, self.channel_in, 1, 1)
            return torch.nn.functional.conv2d(x, inv_weights, bias=None, stride=1)

    def convWeights(self, rgb_type):
        if rgb_type == 'RGB':
            self.conv_weights[0] = torch.Tensor([0.299, 0.587, 0.114])
            self.conv_weights[1] = torch.Tensor([-0.147, -0.289, 0.436])
            self.conv_weights[2] = torch.Tensor([0.615, -0.515, -0.100])
        elif rgb_type == 'BGR':
            self.conv_weights[0] = torch.Tensor([0.114, 0.587, 0.299])
            self.conv_weights[1] = torch.Tensor([0.436, -0.289, -0.147])
            self.conv_weights[2] = torch.Tensor([-0.100, -0.515, 0.615])
        else:
            exit(1)


class GreyNet(torch.nn.Module):
    def __init__(self, rgb_type, subnet_constructor=None, block_num=[], Conv1x1Grey_learnable=True):
        super(GreyNet, self).__init__()
        channel_in = 3
        channel_out = 1
        operations = []
        b = ConvGrey(rgb_type, Conv1x1Grey_learnable)
        operations.append(b)
        for j in range(block_num[0]):
            b = InvBlock(subnet_constructor, channel_in, channel_out)
            operations.append(b)
        self.operations = torch.nn.ModuleList(operations)

    def forward(self, x, rev=False, c=False):
        out = x
        j = 0
        if not rev:
            for op in self.operations:
                out = op.forward(out, rev)
                if c:
                    j += op.jacobian(out, rev)
        else:
            for op in reversed(self.operations):
                out = op.forward(out, rev)
                if c:
                    j += op.jacobian(out, rev)
        if c:
            return out, j
        else:
            return out


class HaarDS(torch.nn.Module):
    def __init__(self, channel_in):
        super(HaarDS, self).__init__()
        self.haar_weights = None
        self.last_jac = None
        self.elements = None
        self.channel_in = channel_in
        self.setWeights()

    def forward(self, x, rev=False):
        if not rev:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(1 / 16.)
            out = torch.nn.functional.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.channel_in) / 4.0
            out = out.reshape([x.shape[0], self.channel_in, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2] // 2, x.shape[3] // 2])
            return out
        else:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(16.)
            out = x.reshape([x.shape[0], 4, self.channel_in, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2], x.shape[3]])
            return torch.nn.functional.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups=self.channel_in)

    def setWeights(self):
        self.haar_weights = torch.ones(4, 1, 2, 2)
        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1
        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1
        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1
        self.haar_weights = torch.cat([self.haar_weights] * self.channel_in, 0)
        self.haar_weights = torch.nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False

    def jb(self, x, rev=False):
        return self.last_jac


class ConvDS(torch.nn.Module):
    def __init__(self, scale):
        super(ConvDS, self).__init__()
        self.conv_weights = None
        self.scale = scale
        self.scale2 = self.scale ** 2
        self.setWeights()

    def forward(self, x, rev=False):
        if not rev:
            # 获取输入张量的高度和宽度
            h = x.shape[2]
            w = x.shape[3]
            wpad = 0
            hpad = 0
            # 计算需要填充的宽度和高度以确保输入可以被整除
            if w % self.scale != 0:
                wpad = self.scale - w % self.scale
            if h % self.scale != 0:
                hpad = self.scale - h % self.scale
            # 如果需要填充，进行填充操作
            if wpad != 0 or hpad != 0:
                padding = (wpad // 2, wpad - wpad // 2, hpad // 2, hpad - hpad // 2)
                pad = torch.nn.ReplicationPad2d(padding)
                x = pad(x)
            # 获取填充后的张量的形状
            [B, C, H, W] = list(x.size())
            # 重塑张量以便进行空间重排
            x = x.reshape(B, C, H // self.scale, self.scale, W // self.scale, self.scale)
            x = x.permute(0, 1, 3, 5, 2, 4)
            x = x.reshape(B, C * self.scale2, H // self.scale, W // self.scale)
            # 准备卷积权重
            conv_weights = self.conv_weights.reshape(self.scale2, self.scale2, 1, 1)
            conv_weights = conv_weights.repeat(C, 1, 1, 1)
            # 应用卷积操作
            out = torch.nn.functional.conv2d(x, conv_weights, bias=None, stride=1, groups=C)
            # 重塑卷积结果并返回
            out = out.reshape(B, C, self.scale2, H // self.scale, W // self.scale)
            out = torch.transpose(out, 1, 2)
            return out.reshape(B, C * self.scale2, H // self.scale, W // self.scale)
        else:
            # 计算反向卷积权重
            inv_weights = torch.inverse(self.conv_weights)
            inv_weights = inv_weights.reshape(self.scale2, self.scale2, 1, 1)
            # 获取输入张量的形状
            [B, C_, H_, W_] = list(x.size())
            C = C_ // self.scale2
            H = H_ * self.scale
            W = W_ * self.scale
            # 准备反向卷积权重
            inv_weights = inv_weights.repeat(C, 1, 1, 1)
            # 重塑张量以便进行反向空间重排
            x = x.reshape(B, self.scale2, C, H_, W_)
            x = torch.transpose(x, 1, 2)
            x = x.reshape(B, C_, H_, W_)
            # 应用反向卷积操作
            out = torch.nn.functional.conv2d(x, inv_weights, bias=None, stride=1, groups=C)
            # 重塑结果并返回
            out = out.reshape(B, C, self.scale, self.scale, H_, W_)
            out = out.permute(0, 1, 4, 2, 5, 3)
            return out.reshape(B, C, H, W)

    def setWeights(self):
        self.conv_weights = torch.eye(self.scale2)
        if self.scale == 2:
            self.conv_weights[0] = torch.Tensor([1. / 4, 1. / 4, 1. / 4, 1. / 4])
            self.conv_weights[1] = torch.Tensor([1. / 4, -1. / 4, 1. / 4, -1. / 4])
            self.conv_weights[2] = torch.Tensor([1. / 4, 1. / 4, -1. / 4, -1. / 4])
            self.conv_weights[3] = torch.Tensor([1. / 4, -1. / 4, -1. / 4, 1. / 4])
        else:
            self.conv_weights[0] = torch.Tensor([1. / self.scale2] * self.scale2)
        self.conv_weights = torch.nn.Parameter(self.conv_weights)


class Rescale(torch.nn.Module):
    def __init__(self, channel_in=3, channel_out=3, subnet_constructor=None, block_num=[], down_num=2, down_first=False, use_ConvDownsampling=False, down_scale=4):
        super(Rescale, self).__init__()
        operations = []
        if use_ConvDownsampling:
            down_num = 1
            down_first = True
        current_channel = channel_in
        if down_first:
            for i in range(down_num):
                if use_ConvDownsampling:
                    b = ConvDS(down_scale)
                    current_channel *= down_scale ** 2
                else:
                    b = HaarDS(current_channel)
                    current_channel *= 4
                operations.append(b)
            for j in range(block_num[0]):
                b = InvBlock(subnet_constructor, current_channel, channel_out)
                operations.append(b)
        else:
            for i in range(down_num):
                b = HaarDS(current_channel)
                operations.append(b)
                current_channel *= 4
                for j in range(block_num[i]):
                    b = InvBlock(subnet_constructor, current_channel, channel_out)
                    operations.append(b)
        self.operations = torch.nn.ModuleList(operations)

    def forward(self, x, rev=False, c=False):
        out = x
        j = 0
        if not rev:
            for op in self.operations:
                out = op.forward(out, rev)
                if c:
                    j += op.jacobian(out, rev)
        else:
            for op in reversed(self.operations):
                out = op.forward(out, rev)
                if c:
                    j += op.jacobian(out, rev)
        if c:
            return out, j
        else:
            return out
