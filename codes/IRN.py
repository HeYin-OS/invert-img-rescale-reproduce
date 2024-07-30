import math
import os
from collections import OrderedDict, Counter, defaultdict
import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler
import networks


class Quant(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        input = torch.clamp(input, 0, 1)
        return (input * 255.).round() / 255.

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class MRestart(_LRScheduler):
    def __init__(self, optimizer, milestones, restarts=None, weights=None, gamma=0.1, clear_state=False, last_epoch=-1):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.clear_state = clear_state
        self.restarts = restarts if restarts else [0]
        self.restart_weights = weights if weights else [1]
        super(MRestart, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch in self.restarts:
            if self.clear_state:
                self.optimizer.state = defaultdict(dict)
            weight = self.restart_weights[self.restarts.index(self.last_epoch)]
            return [group['initial_lr'] * weight for group in self.optimizer.param_groups]
        if self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma ** self.milestones[self.last_epoch] for group in self.optimizer.param_groups]


class CRestart(_LRScheduler):
    def __init__(self, optimizer, T_period, restarts=None, weights=None, eta_min=0, last_epoch=-1):
        self.T_period = T_period
        self.T_max = self.T_period[0]
        self.eta_min = eta_min
        self.restarts = restarts if restarts else [0]
        self.restart_weights = weights if weights else [1]
        self.last_restart = 0
        super(CRestart, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        elif self.last_epoch in self.restarts:
            self.last_restart = self.last_epoch
            self.T_max = self.T_period[self.restarts.index(self.last_epoch) + 1]
            weight = self.restart_weights[self.restarts.index(self.last_epoch)]
            return [group['initial_lr'] * weight for group in self.optimizer.param_groups]
        elif (self.last_epoch - self.last_restart - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [group['lr'] + (base_lr - self.eta_min) * (1 - math.cos(math.pi / self.T_max)) / 2 for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)]
        return [(1 + math.cos(math.pi * (self.last_epoch - self.last_restart) / self.T_max)) / (1 + math.cos(math.pi * ((self.last_epoch - self.last_restart) - 1) / self.T_max)) * (
                group['lr'] - self.eta_min) + self.eta_min for group in self.optimizer.param_groups]


class ReLoss(torch.nn.Module):
    def __init__(self, LT='l2', eps=1e-6):
        super(ReLoss, self).__init__()
        self.LT = LT
        self.eps = eps

    def forward(self, x, target):
        if self.LT == 'l2':
            return torch.mean(torch.sum((x - target) ** 2, (1, 2, 3)))
        elif self.LT == 'l1':
            diff = x - target
            return torch.mean(torch.sum(torch.sqrt(diff * diff + self.eps), (1, 2, 3)))
        else:
            return 0


class GANLoss(torch.nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val
        if self.gan_type == 'gan' or self.gan_type == 'ragan':
            self.loss = torch.nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = torch.nn.MSELoss()
        elif self.gan_type == 'wgan-gp':
            def wgan_loss(input, target):
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss


class GPL(torch.nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super(GPL, self).__init__()
        self.register_buffer('grad_outputs', torch.Tensor())
        self.output = self.grad_outputs.to(device)

    def get_grad_outputs(self, input):
        if self.output.size() != input.size():
            self.output.resize_(input.size()).fill_(1.0)
        return self.output

    def forward(self, interp, interp_crit):
        output = self.get_grad_outputs(interp_crit)
        interp = torch.autograd.grad(outputs=interp_crit, inputs=interp, grad_outputs=output, create_graph=True, retain_graph=True, only_inputs=True)[0]
        interp = interp.view(interp.size(0), -1)
        norm = interp.norm(2, dim=1)
        return ((norm - 1) ** 2).mean()


class Quantization(torch.nn.Module):
    def __init__(self):
        super(Quantization, self).__init__()

    def forward(self, input):
        return Quant.apply(input)


class IRN:
    def __init__(self, opt):
        self.output = self.input = self.real_H = self.ref_L = None
        self.Reconstruction_forw = self.Reconstruction_back = None
        self.optimizer_G = None
        self.log_dict = None
        self.opt = opt
        self.device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.is_train = opt['is_train']
        self.schedulers = []
        self.optimizers = []
        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1
        train_opt = opt['train']
        test_opt = opt['test']
        self.train_opt = train_opt
        self.test_opt = test_opt
        self.netG = networks.makeG(opt).to(self.device)
        self.dataParallel(opt)
        self.printNet()
        self.load()
        self.Quantization = Quantization()
        self.changeIntoNetTraining(train_opt)

    def dataParallel(self, opt):
        if opt['dist']:
            self.netG = torch.nn.parallel.DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = torch.nn.parallel.DataParallel(self.netG)

    def downscale(self, HR_img):
        self.netG.eval()
        with torch.no_grad():
            LR_img = self.netG(x=HR_img)[:, :3, :, :]
            LR_img = self.Quantization(LR_img)
        self.netG.train()

        return LR_img

    def feed(self, data):
        self.ref_L = data['LQ'].to(self.device)
        self.real_H = data['GT'].to(self.device)

    def optimize(self, step):
        self.optimizer_G.zero_grad()
        self.input = self.real_H
        self.output = self.netG(x=self.input)
        shape = self.output[:, 3:, :, :].shape
        LR_ref = self.ref_L.detach()
        LFF, LFC = self.lossF(self.output[:, :3, :, :], LR_ref, self.output[:, 3:, :, :])
        LR = self.Quantization(self.output[:, :3, :, :])
        if self.train_opt['add_noise_on_y']:
            probability = self.train_opt['y_noise_prob']
            noise_scale = self.train_opt['y_noise_scale']
            prob = np.random.rand()
            if prob < probability:
                LR = LR + noise_scale * self.gsb(LR.shape)
        gaussian_scale = self.train_opt['gaussian_scale'] if self.train_opt['gaussian_scale'] is not None else 1
        y_ = torch.cat((LR, gaussian_scale * self.gsb(shape)), dim=1)
        l_back_rec = self.lossB(self.real_H, y_)
        loss = LFF + l_back_rec + LFC
        loss.backward()
        if self.train_opt['gradient_clipping']:
            torch.nn.utils.clip_grad_norm_(self.netG.parameters(), self.train_opt['gradient_clipping'])
        self.optimizer_G.step()
        self.log_dict['l_forw_fit'] = LFF.item()
        self.log_dict['l_forw_ce'] = LFC.item()
        self.log_dict['l_back_rec'] = l_back_rec.item()

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['LR_ref'] = self.ref_L.detach()[0].float().cpu()
        out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        out_dict['LR'] = self.forw_L.detach()[0].float().cpu()
        out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    def get_current_losses(self):
        pass

    def changeIntoNetTraining(self, train_opt):
        if self.is_train:
            self.netG.train()
            self.Reconstruction_forw = ReLoss(LT=self.train_opt['pixel_criterion_forw'])
            self.Reconstruction_back = ReLoss(LT=self.train_opt['pixel_criterion_back'])
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'], weight_decay=wd_G, betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(MRestart(optimizer, train_opt['lr_steps'], restarts=train_opt['restarts'], weights=train_opt['restart_weights'], gamma=train_opt['lr_gamma'], clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(CRestart(optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'], restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            self.log_dict = OrderedDict()

    def printNet(self):
        pass

    def log(self):
        return self.log_dict

    def upscale(self, LR_img, scale, gaussian_scale=1):
        Lshape = LR_img.shape
        zshape = [Lshape[0], Lshape[1] * (scale ** 2 - 1), Lshape[2], Lshape[3]]
        y_ = torch.cat((LR_img, gaussian_scale * self.gsb(zshape)), dim=1)
        self.netG.eval()
        with torch.no_grad():
            HR_img = self.netG(x=y_, rev=True)[:, :3, :, :]
        self.netG.train()
        return HR_img

    def save(self, iter_label):
        self.saveN(self.netG, 'G', iter_label)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            self.loadN(load_path_G, self.netG, self.opt['path']['strict_load'])

    def _set_lr(self, lr_groups_l):
        for optimizer, lr_groups in zip(self.optimizers, lr_groups_l):
            for param_group, lr in zip(optimizer.param_groups, lr_groups):
                param_group['lr'] = lr

    def _get_init_lr(self):
        init_lr_groups_l = []
        for optimizer in self.optimizers:
            init_lr_groups_l.append([v['initial_lr'] for v in optimizer.param_groups])
        return init_lr_groups_l

    def update(self, cur_iter, warmup_iter=-1):
        for scheduler in self.schedulers:
            scheduler.step()
        if cur_iter < warmup_iter:
            init_lr_g_l = self._get_init_lr()
            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append([v / warmup_iter * cur_iter for v in init_lr_g])
            self._set_lr(warm_up_lr_l)

    def get(self):
        return self.optimizers[0].param_groups[0]['lr']

    def lossF(self, out, y, z):
        LFF = self.train_opt['lambda_fit_forw'] * self.Reconstruction_forw(out, y)
        z = z.reshape([out.shape[0], -1])
        LFC = self.train_opt['lambda_ce_forw'] * torch.sum(z ** 2) / z.shape[0]
        return LFF, LFC

    def description(self, network):
        if isinstance(network, torch.nn.DataParallel) or isinstance(network, torch.nn.parallel.DistributedDataParallel):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n

    def saveN(self, network, network_label, iter_label):
        save_filename = '{}_{}.pth'.format(iter_label, network_label)
        save_path = os.path.join(self.opt['path']['models'], save_filename)
        if isinstance(network, torch.nn.DataParallel) or isinstance(network, torch.nn.parallel.DistributedDataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def gsb(self, dims):
        return torch.randn(tuple(dims)).to(self.device)

    def test(self):
        Lshape = self.ref_L.shape
        input_dim = Lshape[1]
        self.input = self.real_H
        zshape = [Lshape[0], input_dim * (self.opt['scale'] ** 2) - Lshape[1], Lshape[2], Lshape[3]]
        gaussian_scale = 1
        if self.test_opt and self.test_opt['gaussian_scale'] is not None:
            gaussian_scale = self.test_opt['gaussian_scale']
        self.netG.eval()
        with torch.no_grad():
            self.forw_L = self.netG(x=self.input)[:, :3, :, :]
            self.forw_L = self.Quantization(self.forw_L)
            y_forw = torch.cat((self.forw_L, gaussian_scale * self.gsb(zshape)), dim=1)
            self.fake_H = self.netG(x=y_forw, rev=True)[:, :3, :, :]
        self.netG.train()

    def lossB(self, x, y):
        temp = self.netG(x=y, rev=True)
        temp_img = temp[:, :3, :, :]
        return self.train_opt['lambda_rec_back'] * self.Reconstruction_back(x, temp_img)

    def loadN(self, load_path, network, strict=True):
        if isinstance(network, torch.nn.DataParallel) or isinstance(network, torch.nn.parallel.DistributedDataParallel):
            network = network.module
        load_net = torch.load(load_path)
        load_net_clean = OrderedDict()
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        network.load_state_dict(load_net_clean, strict=strict)

    def saveT(self, epoch, iter_step):
        state = {'epoch': epoch, 'iter': iter_step, 'schedulers': [], 'optimizers': []}
        for s in self.schedulers:
            state['schedulers'].append(s.state_dict())
        for o in self.optimizers:
            state['optimizers'].append(o.state_dict())
        save_filename = '{}.state'.format(iter_step)
        save_path = os.path.join(self.opt['path']['training_state'], save_filename)
        torch.save(state, save_path)

    def resume(self, resume_state):
        resume_optimizers = resume_state['optimizers']
        resume_schedulers = resume_state['schedulers']
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)
