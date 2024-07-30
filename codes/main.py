import argparse
import math
import os
import random
from datetime import datetime
import utils
import numpy as np
import torch
import yaml
from codes.sampler import IterSampler
from codes.utils import OrderedYamlSupport
Loader, Dumper = OrderedYamlSupport()


def optionParse(path, is_train=True):
    with open(path, mode='r') as f:
        opt = yaml.load(f, Loader=Loader)
    gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    opt['is_train'] = is_train
    for phase, dataset in opt['datasets'].items():
        phase = phase.split('_')[0]
        dataset['phase'] = phase
        if dataset.get('GT', None) is not None:
            dataset['GT'] = os.path.expanduser(dataset['GT'])
        if dataset.get('LQ', None) is not None:
            dataset['LQ'] = os.path.expanduser(dataset['LQ'])
        dataset['data_type'] = 'img'
    for key, path in opt['path'].items():
        if path and key in opt['path'] and key != 'strict_load':
            opt['path'][key] = os.path.expanduser(path)
    opt['path']['root'] = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir))
    if is_train:
        experiments_root = os.path.join(opt['path']['root'], 'experiments', opt['name'])
        opt['path']['experiments_root'] = experiments_root
        opt['path']['models'] = os.path.join(experiments_root, 'models')
        opt['path']['training_state'] = os.path.join(experiments_root, 'training_state')
        opt['path']['log'] = experiments_root
        opt['path']['val_images'] = os.path.join(experiments_root, 'val_images')
    else:
        result_root = os.path.join(opt['path']['root'], 'results', opt['name'])
        opt['path']['results_root'] = result_root
        opt['path']['log'] = result_root
    return opt


def resume(opt, resume_iter):
    if opt['path']['resume_state']:
        opt['path']['pretrain_model_G'] = os.path.join(opt['path']['models'], '{}_G.pth'.format(resume_iter))
        if 'gan' in opt['model']:
            opt['path']['pretrain_model_D'] = os.path.join(opt['path']['models'], '{}_D.pth'.format(resume_iter))


def mkdirAndRename(path):
    if os.path.exists(path):
        new_name = path + '_completed_' + datetime.now().strftime('%y%m%d-%H%M%S')
        os.rename(path, new_name)
    os.makedirs(path)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


class NoneDict(dict):
    def __missing__(self, key):
        return None


def toNoneDict(opt):
    if isinstance(opt, dict):
        temp = dict()
        for key, sub_opt in opt.items():
            temp[key] = toNoneDict(sub_opt)
        return NoneDict(**temp)
    elif isinstance(opt, list):
        return [toNoneDict(sub_opt) for sub_opt in opt]
    else:
        return opt


def loadData(dataset, dataset_opt, opt=None, sampler=None):
    phase = dataset_opt['phase']
    if phase == 'train':
        if opt['dist']:
            world_size = torch.distributed.get_world_size()
            num_workers = dataset_opt['n_workers']
            batch_size = dataset_opt['batch_size'] // world_size
            shuffle = False
        else:
            num_workers = dataset_opt['n_workers'] * len(opt['gpu_ids'])
            batch_size = dataset_opt['batch_size']
            shuffle = True
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, sampler=sampler, drop_last=True, pin_memory=False)
    else:
        return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)


def setData(dataset_opt):
    global D
    mode = dataset_opt['mode']
    if mode == 'LQ':
        from dataset import LQDataset as D
    elif mode == 'LQGT':
        from dataset import LQGTDataset as D
    return D(dataset_opt)


def trainingProcedure(current_step, model, opt, start_epoch, total_epochs, total_iters, train_loader, train_sampler, val_loader):
    for epoch in range(start_epoch, total_epochs + 1):
        if opt['dist']:
            train_sampler.set_epoch(epoch)
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break
            modelTraining(current_step, model, opt, train_data)
            if current_step % opt['logger']['print_freq'] == 0:
                loggingForTraining(current_step, epoch, model)
            if current_step % opt['train']['val_freq'] == 0:
                avg_psnr = 0.0
                idx = 0
                validate(avg_psnr, current_step, idx, model, opt, val_loader)
            if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                checkpoint(current_step, epoch, model)


def modelTraining(current_step, model, opt, train_data):
    model.feed(train_data)
    model.optimize(current_step)
    model.update(current_step, warmup_iter=opt['train']['warmup_iter'])


def loggingForTraining(current_step, epoch, model):
    logs = model.log()
    message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(
        epoch, current_step, model.get())
    for k, v in logs.items():
        message += '{:s}: {:.4e} '.format(k, v)


def checkpoint(current_step, epoch, model):
    model.save(current_step)
    model.saveT(epoch, current_step)


def validate(avg_psnr, current_step, idx, model, opt, val_loader):
    for val_data in val_loader:
        img_dir, img_name = validateProcess(idx, opt, val_data)
        input_and_test(model, val_data)
        gt_img, gtl_img, lr_img, sr_img = getViusals(model)
        save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(img_name, current_step))
        utils.save_img(sr_img, save_img_path)
        save_img_path_L = os.path.join(img_dir, '{:s}_forwLR_{:d}.png'.format(img_name, current_step))
        utils.save_img(lr_img, save_img_path_L)
        if current_step == opt['train']['val_freq']:
            saveExample(current_step, gt_img, gtl_img, img_dir, img_name)
        psnr(avg_psnr, gt_img, opt, sr_img)


def psnr(avg_psnr, gt_img, opt, sr_img):
    crop_size = opt['scale']
    gt_img = gt_img / 255.
    sr_img = sr_img / 255.
    cropped_sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size, :]
    cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size, :]
    avg_psnr += utils.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)


def saveExample(current_step, gt_img, gtl_img, img_dir, img_name):
    save_img_path_gt = os.path.join(img_dir, '{:s}_GT_{:d}.png'.format(img_name, current_step))
    utils.save_img(gt_img, save_img_path_gt)
    save_img_path_gtl = os.path.join(img_dir, '{:s}_LR_ref_{:d}.png'.format(img_name, current_step))
    utils.save_img(gtl_img, save_img_path_gtl)


def getViusals(model):
    visuals = model.get_current_visuals()
    sr_img = utils.tensor2img(visuals['SR'])
    gt_img = utils.tensor2img(visuals['GT'])
    lr_img = utils.tensor2img(visuals['LR'])
    gtl_img = utils.tensor2img(visuals['LR_ref'])
    return gt_img, gtl_img, lr_img, sr_img


def input_and_test(model, val_data):
    model.feed(val_data)
    model.test()


def validateProcess(idx, opt, val_data):
    idx += 1
    img_name = os.path.splitext(os.path.basename(val_data['LQ_path'][0]))[0]
    img_dir = os.path.join(opt['path']['val_images'], img_name)
    mkdir(img_dir)
    return img_dir, img_name


def resumeModelState(model, resume_state):
    if resume_state:
        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume(resume_state)
    else:
        current_step = 0
        start_epoch = 0
    return current_step, start_epoch


def loadDataset(dataset_ratio, opt):
    global total_iters, total_epochs, train_sampler, train_loader, val_loader
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = setData(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))
            if opt['dist']:
                train_sampler = IterSampler(train_set, None, -1, dataset_ratio)
                total_epochs = int(math.ceil(total_iters / (train_size * dataset_ratio)))
            else:
                train_sampler = None
            train_loader = loadData(train_set, dataset_opt, opt, train_sampler)
        elif phase == 'val':
            val_set = setData(dataset_opt)
            val_loader = loadData(val_set, dataset_opt, opt, None)


def sewing(opt):
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def mkdirPlus(opt, resume_state):
    if resume_state is None:
        mkdirAndRename(
            opt['path']['experiments_root'])
        mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                and 'pretrain_model' not in key and 'resume' not in key))


def resumeTraining(opt):
    if opt['path'].get('resume_state', None):
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'], map_location=lambda storage, loc: storage.cuda(device_id))
        print(resume_state.keys())
        resume(opt, resume_state['iter'])
    else:
        resume_state = None
    return resume_state


def readCfg(args):
    opt = optionParse(args.opt, is_train=True)
    opt['dist'] = False
    return opt


def buildArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str)
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    return args


def main():
    global val_loader, total_epochs, train_loader, train_sampler, total_iters
    args = buildArgs()
    opt = readCfg(args)
    resume_state = resumeTraining(opt)
    mkdirPlus(opt, resume_state)
    opt = toNoneDict(opt)
    sewing(opt)
    dataset_ratio = 200
    loadDataset(dataset_ratio, opt)
    model = utils.create_model(opt)
    current_step, start_epoch = resumeModelState(model, resume_state)
    trainingProcedure(current_step, model, opt, start_epoch, total_epochs, total_iters, train_loader, train_sampler, val_loader)


if __name__ == '__main__':
    main()
