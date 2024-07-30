import torch
import utils


class DenseBlock(torch.nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv2 = torch.nn.Conv2d(channel_in + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = torch.nn.Conv2d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = torch.nn.Conv2d(channel_in + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = torch.nn.Conv2d(channel_in + 4 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)
        if init == 'xavier':
            utils.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        else:
            utils.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        utils.initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        return self.conv5(torch.cat((x, x1, x2, x3, x4), 1))


def subnet(net_structure, init='xavier', gc=32):
    def constructor(channel_in, channel_out):
        if net_structure == 'DBNet':
            if init == 'xavier':
                return DenseBlock(channel_in, channel_out, init, gc=gc)
            else:
                return DenseBlock(channel_in, channel_out, gc=gc)
        else:
            return None

    return constructor
