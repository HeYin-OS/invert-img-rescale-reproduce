import functools
import torch
import utils


class ResidualDB(torch.nn.Module):
    def __init__(self, nf, gc=32):
        super(ResidualDB, self).__init__()
        self.RDB1 = DenseBlock(nf, gc)
        self.RDB2 = DenseBlock(nf, gc)
        self.RDB3 = DenseBlock(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class DenseBlock(torch.nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = torch.nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = torch.nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = torch.nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = torch.nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)
        utils.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        utils.initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class BlockNet(torch.nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super(BlockNet, self).__init__()
        RRDB_block_f = functools.partial(ResidualDB, nf=nf, gc=gc)
        self.conv_first = torch.nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = utils.make_layer(RRDB_block_f, nb)
        self.trunk_conv = torch.nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = torch.nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk
        out = self.conv_last(fea)
        return out
