import cv2
import lmdb
import numpy as np
import torch
import random

from codes import utils


class LQDataset(torch.data.Dataset):
    """
    初始化 LQDataset 类
    :param opt: 包含数据集配置的字典
    """

    def __init__(self, opt):
        """
        初始化 LMDB 环境
        """
        super(LQDataset, self).__init__()
        self.opt = opt
        self.type = self.opt['data_type']
        self.path, self.size = None, None
        self.env = None
        self.path, self.size = utils.getImgPath(opt['data_type'], opt['dataroot_LQ'])

    def __getitem__(self, index):
        """
        获取数据集中的一个样本
        :param index: 样本索引
        :return: 包含图像和路径的字典
        """
        if self.type == 'lmdb' and self.env is None:
            self._init_sub()
        LQ_path = self.path[index]
        if self.type == 'lmdb':
            resolution = [int(s) for s in self.size[index].split('_')]
        else:
            resolution = None
        img_LQ = utils.read_img(self.env, LQ_path, resolution)
        H, W, C = img_LQ.shape
        if self.opt.get('color'):
            img_LQ = utils.CvtChannel(C, self.opt['color'], img_list=[img_LQ])[0]
        if img_LQ.shape[2] == 3:
            img_LQ = img_LQ[:, :, [2, 1, 0]]  # 将 RGB 转换为 BGR
        img_LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))).float()
        return {'LQ': img_LQ, 'LQ_path': LQ_path}

    def __len__(self):
        return len(self.path)

    def _init_sub(self):
        self.env = lmdb.open(self.opt['dataroot_LQ'], readonly=True, lock=False, readahead=False, meminit=False)


class LQGTDataset(torch.data.Dataset):

    def __init__(self, opt):
        """
        初始化 LQGTDataset 类
        :param opt: 包含数据集配置的字典
        """
        super(LQGTDataset, self).__init__()
        self.opt = opt
        self.type = self.opt['data_type']
        self.path_LQ, self.path_GT = None, None
        self.size_LQ, self.size_GT = None, None
        self.LQ_env, self.GT_env = None, None
        # 获取图像路径和尺寸
        self.path_GT, self.size_GT = utils.getImgPath(self.type, opt['dataroot_GT'])
        self.path_LQ, self.size_LQ = utils.getImgPath(self.type, opt['dataroot_LQ'])
        # 随机缩放列表
        self.random_list = [1]
        self.use_grey = self.opt.get('use_grey', False)

    def __getitem__(self, index):
        """
        获取数据集中的一个样本
        :param index: 样本索引
        :return: 包含图像和路径的字典
        """
        global img_LQ, img_Grey
        if self.data_type == 'lmdb' and (self.GT_env is None or self.LQ_env is None):
            self._init_lmdb()
        GT_path = self.path_GT[index]
        LQ_path = self.path_LQ[index] if self.path_LQ else None
        scale = self.opt['scale']
        GT_size = self.opt['GT_size']
        # 读取 GT 图像
        if self.type == 'lmdb':
            resolution = [int(s) for s in self.size_GT[index].split('_')]
        else:
            resolution = None
        img_GT = utils.read_img(self.GT_env, GT_path, resolution)
        # 进行图像剪裁
        if self.opt['phase'] != 'train':
            img_GT = utils.modcrop(img_GT, scale)
        # 转换颜色通道
        if self.opt.get('color'):
            img_GT = utils.CvtChannel(img_GT.shape[2], self.opt['color'], [img_GT])[0]
        img_GT = self.readLQ(GT_size, LQ_path, img_GT, index, scale)
        img_GT = self.dataAugment(GT_size, img_GT, scale)
        # 转换颜色通道
        if not self.use_grey and self.opt.get('color'):
            img_LQ = utils.CvtChannel(img_LQ.shape[2], self.opt['color'], [img_LQ])[0]
        # 灰度图像处理
        if self.use_grey:
            img_Grey = cv2.cvtColor(img_GT, cv2.COLOR_BGR2GRAY)
        img_GT = self.cvtBGR(img_GT)
        return self.cvtSensor(GT_path, LQ_path, img_GT, img_Grey)

    def readLQ(self, GT_size, LQ_path, img_GT, index, scale):
        global img_LQ
        if not self.use_grey and LQ_path:
            if self.type == 'lmdb':
                resolution = [int(s) for s in self.size_LQ[index].split('_')]
            else:
                resolution = None
            img_LQ = utils.read_img(self.LQ_env, LQ_path, resolution)
            if self.opt['phase'] == 'train':
                random_scale = random.choice(self.random_list)
                H_s, W_s, _ = img_GT.shape

                def _mod(n, rs, s, t):
                    rlt = int(n * rs)
                    rlt = (rlt // s) * s
                    return t if rlt < t else rlt

                H_s = _mod(H_s, random_scale, scale, GT_size)
                W_s = _mod(W_s, random_scale, scale, GT_size)
                img_GT = cv2.resize(np.copy(img_GT), (W_s, H_s), interpolation=cv2.INTER_LINEAR)
                if img_GT.ndim == 2:
                    img_GT = cv2.cvtColor(img_GT, cv2.COLOR_GRAY2BGR)
            img_LQ = utils.resizeImg(img_GT, 1 / scale, True)
            if img_LQ.ndim == 2:
                img_LQ = np.expand_dims(img_LQ, axis=2)
        return img_GT

    def dataAugment(self, GT_size, img_GT, scale):
        global img_LQ
        if self.opt['phase'] == 'train':
            H, W, _ = img_GT.shape
            if H < GT_size or W < GT_size:
                img_GT = cv2.resize(np.copy(img_GT), (GT_size, GT_size), interpolation=cv2.INTER_LINEAR)
                if not self.use_grey:
                    img_LQ = utils.resizeImg(img_GT, 1 / scale, True)
                    if img_LQ.ndim == 2:
                        img_LQ = np.expand_dims(img_LQ, axis=2)
            if not self.use_grey:
                H, W, C = img_LQ.shape
                LQ_size = GT_size // scale
                rnd_h = random.randint(0, max(0, H - LQ_size))
                rnd_w = random.randint(0, max(0, W - LQ_size))
                img_LQ = img_LQ[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
                rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)
                img_GT = img_GT[rnd_h_GT:rnd_h_GT + GT_size, rnd_w_GT:rnd_w_GT + GT_size, :]
            else:
                rnd_h_GT = random.randint(0, max(0, H - GT_size))
                rnd_w_GT = random.randint(0, max(0, W - GT_size))
                img_GT = img_GT[rnd_h_GT:rnd_h_GT + GT_size, rnd_w_GT:rnd_w_GT + GT_size, :]
            if not self.use_grey:
                img_LQ, img_GT = utils.aug([img_LQ, img_GT], self.opt['use_flip'], self.opt['use_rot'])
            else:
                img_GT = utils.aug([img_GT], self.opt['use_flip'], self.opt['use_rot'])[0]
        return img_GT

    def cvtBGR(self, img_GT):
        global img_LQ
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
            if not self.use_grey:
                img_LQ = img_LQ[:, :, [2, 1, 0]]
        return img_GT

    def __len__(self):
        return len(self.path_GT)

    def _init_lmdb(self):
        """
        初始化 LMDB 环境
        """
        self.GT_env = lmdb.open(self.opt['dataroot_GT'], readonly=True, lock=False, readahead=False, meminit=False)
        if not self.use_grey:
            self.LQ_env = lmdb.open(self.opt['dataroot_LQ'], readonly=True, lock=False, readahead=False, meminit=False)

    def cvtSensor(self, GT_path, LQ_path, img_GT, img_Grey):
        global img_LQ
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        if not self.use_grey:
            img_LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))).float()
        if self.use_grey:
            img_Grey = torch.from_numpy(np.ascontiguousarray(np.expand_dims(img_Grey, 0))).float()
        if LQ_path is None:
            LQ_path = GT_path
        if not self.use_grey:
            return {'LQ': img_LQ, 'GT': img_GT, 'LQ_path': LQ_path, 'GT_path': GT_path}
        else:
            return {'Grey': img_Grey, 'GT': img_GT, 'LQ_path': LQ_path, 'GT_path': GT_path}
