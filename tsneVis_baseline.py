# python tsneVis.py -d cifar10 --nclass 10 -n convnet --seed 2023
import torch
import torch.nn as nn
import torch.optim as optim
from openTSNE import TSNE
# from sklearn.manifold import TSNE
import os
import numpy as np
from examples import utils
from data import ClassDataLoader, ClassMemDataLoader
from train import define_model, train_epoch, validate
from condense_con_ema_kmeans import load_resized_data, diffaug
import torch.nn.functional as F
from math import ceil

def decode_zoom(img, target, factor, size=-1):
    if size == -1:
        size = img.shape[-1]
    resize = nn.Upsample(size=size, mode='bilinear')

    h = img.shape[-1]
    remained = h % factor
    if remained > 0:
        img = F.pad(img, pad=(0, factor - remained, 0, factor - remained), value=0.5)
    s_crop = ceil(h / factor)
    n_crop = factor**2

    cropped = []
    for i in range(factor):
        for j in range(factor):
            h_loc = i * s_crop
            w_loc = j * s_crop
            cropped.append(img[:, :, h_loc:h_loc + s_crop, w_loc:w_loc + s_crop])
    cropped = torch.cat(cropped)
    data_dec = resize(cropped)
    target_dec = torch.cat([target for _ in range(n_crop)])

    return data_dec, target_dec


def decode_zoom_multi(img, target, factor_max):
    data_multi = []
    target_multi = []
    for factor in range(1, factor_max + 1):
        decoded = decode_zoom(img, target, factor)
        data_multi.append(decoded[0])
        target_multi.append(decoded[1])

    return torch.cat(data_multi), torch.cat(target_multi)


def decode_fn(data, target, factor, decode_type, bound=128):
    if factor > 1:
        if decode_type == 'multi':
            data, target = decode_zoom_multi(data, target, factor)
        else:
            data, target = decode_zoom(data, target, factor)

    return data, target

def decode(data, target):
    data_dec = []
    target_dec = []
    nclass=10
    ipc = len(data) // nclass
    for c in range(nclass):
        idx_from = ipc * c
        idx_to = ipc * (c + 1)
        data_ = data[idx_from:idx_to].detach()
        target_ = target[idx_from:idx_to].detach()
        data_, target_ = decode_fn(data_,
                                   target_,
                                   2,
                                   "single",
                                   bound=128)
        data_dec.append(data_)
        target_dec.append(target_)
    # data_dec:[nclass*[factor**2*nclass,3,32,32]]
    data_dec = torch.cat(data_dec)
    target_dec = torch.cat(target_dec)

    print("Dataset is decoded! ", data_dec.shape)
    # save_img('./results/test_dec.png', data_dec, unnormalize=False, dataname=args.dataset)
    return data_dec, target_dec


def main(args, logger, repeat=1):
    trainset, val_loader = load_resized_data(args)
    if args.load_memory:
        loader_real = ClassMemDataLoader(trainset, batch_size=args.batch_real)
    else:
        loader_real = ClassDataLoader(trainset,
                                      batch_size=args.batch_real,
                                      num_workers=args.workers,
                                      shuffle=True,
                                      pin_memory=True,
                                      drop_last=False)
    nclass = trainset.nclass
    _, aug_rand = diffaug(args)
    dd_path = "/home/dd/DL/DataDistillation/Acc-DD/results/cifar10/Baseline/conv3in_grad_mse_inloop100_cut_niter1000_factor2_lr0.01_mix_ipc10_mnum5/data.pt"
    dd_data, dd_target = torch.load(dd_path)
    for i in range(repeat):
        logger(f"\nRepeat: {i + 1}/{repeat}")
        model = define_model(args, nclass, logger)
        checkpoint = torch.load(
            "/home/dd/DL/DataDistillation/Acc-DD/pretrained_models/cifar10/conv3in_cut_seed_2023_lr_0.01_aug_color_crop_cutout/checkpoint_best_0.pth.tar"
        )
        model.load_state_dict(checkpoint)
        model.eval()
        pretrain(model, loader_real, dd_data, dd_target, aug_rand, args)


def pretrain(model, loader_real, dd_data, dd_target, aug_rand, args):
    model = model.cuda()
    save_target, save_features = None, None
    for i, (input, target) in enumerate(loader_real):
        input = input.cuda()
        target = target.cuda()
        output, features = model(input, return_features=True)
        if save_target == None:
            save_target, save_features = target.detach().cpu(
            ), features.detach().cpu()
        else:
            save_target = torch.cat([save_target, target.detach().cpu()])
            save_features = torch.cat([save_features, features.detach().cpu()])
    print(save_target.shape, save_features.shape)
    dd_data, dd_target=torch.load("/home/dd/DL/DataDistillation/Acc-DD/results/cifar10/DNE/conv3in_all_grad_mse_inloop100_cut_niter1000_factor2_lr0.01_mix_ipc10_mnum5/data.pt")
    dd_data, dd_target = decode(dd_data,dd_target)
    dd_data, dd_target = dd_data.cuda(), dd_target.cuda()
    dd_output, dd_features = model(dd_data, return_features=True)
    save_dd_target, save_dd_features = dd_target.detach().cpu(
            ), dd_features.detach().cpu()
    ###
    save_target = torch.cat([save_target, save_dd_target])
    save_features = torch.cat([save_features, save_dd_features])

    tsne = TSNE(perplexity=50,
                metric="euclidean",
                n_jobs=8,
                random_state=42,
                verbose=True)
    embedding_train = tsne.fit(save_features)
    # embedding_train = tsne.fit_transform(save_features)
    # utils.plot(embedding_train, save_target, colors=utils.MACOSKO_COLORS)
    # embedding_train_dd = tsne.fit(save_dd_features)
    utils.plot(embedding_train, save_target,colors=utils.MACOSKO_COLORS)


if __name__ == '__main__':
    import shutil
    from misc.utils import Logger
    from argument import args
    import torch.backends.cudnn as cudnn

    # assert args.pt_from > 0, "set args.pt_from positive! (epochs for pretraining)"

    cudnn.benchmark = True
    if args.seed > 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    args.save_dir = f"./Visualize/{args.datatag}/{args.modeltag}{args.tag}_seed_{args.seed}_lr_{args.lr}_aug_{args.aug_type}"
    os.makedirs(args.save_dir, exist_ok=True)

    # cur_file = os.path.join(os.getcwd(), __file__)
    # shutil.copy(cur_file, args.save_dir)

    logger = Logger(args.save_dir)
    logger(f"Save dir: {args.save_dir}")
    logger(f"Seed: {args.seed}")
    logger(f"Lr: {args.lr}")
    logger(f"Aug-type: {args.aug_type}")

    np.random.seed(2023)
    torch.manual_seed(2023)
    main(args, logger, 1)