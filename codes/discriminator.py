import torch
import torchvision


class Discriminator(torch.nn.Module):
    def __init__(self, in_nc, nf, *args, **kwargs):
        """
        初始化 Discriminator 类
        :param in_nc: 输入通道数
        :param nf: 卷积核数量的基数
        """
        # 定义卷积层和批归一化层
        super().__init__(*args, **kwargs)
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_nc, nf, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
            torch.nn.Conv2d(nf, nf, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(nf),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
            torch.nn.Conv2d(nf, nf * 2, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(nf * 2),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
            torch.nn.Conv2d(nf * 2, nf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(nf * 2),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
            torch.nn.Conv2d(nf * 2, nf * 4, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(nf * 4),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
            torch.nn.Conv2d(nf * 4, nf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(nf * 4),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
            torch.nn.Conv2d(nf * 4, nf * 8, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(nf * 8),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
            torch.nn.Conv2d(nf * 8, nf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(nf * 8),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
            torch.nn.Conv2d(nf * 8, nf * 8, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(nf * 8),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
            torch.nn.Conv2d(nf * 8, nf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(nf * 8),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        # 全连接层
        self.fc1 = torch.nn.Linear(nf * 8 * 4 * 4, 100)
        self.fc2 = torch.nn.Linear(100, 1)
        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        """
        前向传播
        :param x: 输入的张量
        :return: 判别器的输出
        """
        temp = self.lrelu(self.conv0_0(x))
        temp = self.lrelu(self.bn0_1(self.conv0_1(temp)))
        temp = self.lrelu(self.bn1_0(self.conv1_0(temp)))
        temp = self.lrelu(self.bn1_1(self.conv1_1(temp)))
        temp = self.lrelu(self.bn2_0(self.conv2_0(temp)))
        temp = self.lrelu(self.bn2_1(self.conv2_1(temp)))
        temp = self.lrelu(self.bn3_0(self.conv3_0(temp)))
        temp = self.lrelu(self.bn3_1(self.conv3_1(temp)))
        temp = self.lrelu(self.bn4_0(self.conv4_0(temp)))
        temp = self.lrelu(self.bn4_1(self.conv4_1(temp)))
        temp = temp.view(temp.size(0), -1)
        temp = self.lrelu(self.linear1(temp))
        return self.linear2(temp)


class Extractor(torch.nn.Module):
    def __init__(self, feature_layer=34, use_bn=False, use_input_norm=True, device=torch.device('cpu')):
        super(Extractor, self).__init__()
        self.use_input_norm = use_input_norm
        # 选择 VGG 网络模型
        if use_bn:
            model = torchvision.models.vgg19_bn(pretrained=True)
        else:
            model = torchvision.models.vgg19(pretrained=True)
        # 如果需要，对输入进行归一化
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        # 截取 VGG 网络的特征层
        self.features = torch.nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])
        # 冻结特征提取层的参数
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        return self.features(x)
