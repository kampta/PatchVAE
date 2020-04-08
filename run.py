#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" run.py
Code to run the PatchVAE on different datasets

Usage:
# Run with default arguments on mnist
python run.py

Basic VAE borrowed from
https://github.com/pytorch/examples/tree/master/vae
"""

__author__ = "Kamal Gupta"
__email__ = "kampta@cs.umd.edu"
__version__ = "0.1"

import sys
from collections import OrderedDict
import shutil

import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid

from utils import Timer
from utils.torchsummary import summary
from utils.commons import data_loaders, load_vae_model, count_parameters, EdgeWeights
from loss import BetaVaeLoss, VaeConcreteLoss, BetaVaeConcreteLoss,\
    BetaVaeConcretePartsLoss, BetaVaeConcretePartsEntropyLoss, DiscLoss
from model import Discriminator
import utils.commons as commons

from torch.utils.tensorboard import SummaryWriter


def train_vaegan(data_loader, model_d, model_v, opt_d, opt_v, d_loss_fn, v_loss_fn, writer):
    model_v.train()
    model_d.train()
    fwd_clock = Timer()
    bwd_clock = Timer()

    num_batches = args.img_per_epoch // args.batch_size
    data_iterator = iter(data_loader)
    overall_losses = OrderedDict()

    # for batch_idx, (x, _) in enumerate(data_loader):
    for batch_idx in range(num_batches):
        batch_losses = OrderedDict()

        try:
            x, _ = next(data_iterator)

        except StopIteration:
            data_iterator = iter(data_loader)
            continue

        x = x.to(args.device)

        ########################################################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        #######################################################
        # train with real
        model_d.zero_grad()
        real_x = x
        real_y = torch.ones(x.size(0)).cuda()

        outputs = model_d(real_x)
        err_d_real = d_loss_fn(outputs.squeeze(), real_y.squeeze())
        err_d_real.backward()
        batch_losses['err_d_real'] = err_d_real.item()
        batch_losses['d_x'] = outputs.data.mean()

        # train with fake
        fake_y = torch.zeros(x.size(0)).cuda()
        x_tilde, z_app_mean, z_app_var, z_vis_mean = model_v(x, args.temp)
        # recon_x, _ = x_tilde
        outputs = model_d(x_tilde.detach())
        err_d_fake = d_loss_fn(outputs.squeeze(), fake_y.squeeze())
        err_d_fake.backward()
        batch_losses['err_d_fake'] = err_d_fake.item()
        batch_losses['d_v1'] = outputs.data.mean()

        opt_d.step()

        ###########################
        # (2) Update G network: VAE
        ###########################

        model_v.zero_grad()

        loss, loss_dict = v_loss_fn(
            x_tilde, x, z_app_mean, z_app_var, z_vis_mean,
            categorical=args.categorical, py=args.py, beta_p=args.beta_p,
            beta_a=args.beta_a, beta_v=args.beta_v,
            beta_ea=args.beta_ea, beta_ew=args.beta_ew
        )
        loss.backward()
        for loss_key, loss_value in loss_dict.items():
            batch_losses[loss_key] = loss_value.item()
        opt_v.step()

        ############################
        # (3) Update G network: maximize log(D(G(z)))
        ###########################

        x_tilde, z_app_mean, z_app_var, z_vis_mean = model_v(x, args.temp)
        # recon_x, _ = x_tilde
        outputs = model_d(x_tilde)
        real_y.fill_(1)
        err_g = d_loss_fn(outputs.squeeze(), real_y.squeeze())
        err_g.backward()
        batch_losses['err_g'] = err_g.item()
        batch_losses['d_v2'] = outputs.data.mean()
        opt_v.step()

        # Logs
        for loss_key, loss_value in batch_losses.items():
            writer.add_scalar('loss/train/' + loss_key, loss_value, args.steps)
            overall_losses[loss_key] = overall_losses[loss_key] + loss_value \
                if loss_key in overall_losses else loss_value

        args.steps += 1

        if args.steps % 1000 == 1:
            args.temp = max(args.temp * np.exp(-args.anneal * args.steps),
                            args.min_temp)

        if batch_idx % args.log_interval != 0:
            continue

        logstr = '\t'.join(['{}: {:0.4f}'.format(k, v) for k, v in batch_losses.items()])
        print('[{}/{} ({:0.0f}%)]\t{}'.format(batch_idx, num_batches,
                                              100. * batch_idx / num_batches, logstr))

    overall_losses = OrderedDict([(k, v / num_batches) for k, v in overall_losses.items()])
    logstr = '\t'.join(['{}: {:0.4f}'.format(k, v) for k, v in overall_losses.items()])

    print('[End of train epoch]\t# steps: {}\t# images: {}, temp: {:0.2f}'.format(
        args.steps, num_batches * args.batch_size, args.temp))
    print(logstr)
    print('[End of train epoch]\t# calls: {}, Fwd: {:.3f} ms\tBwd: {:.3f} ms'.format(
        fwd_clock.calls, 1000 * fwd_clock.average_time, 1000 * bwd_clock.average_time))

    return overall_losses


def train(data_loader, model, optimizer, loss_function, writer):
    model.train()
    fwd_clock = Timer()
    bwd_clock = Timer()

    losses = OrderedDict()
    losses['loss'] = 0

    num_batches = args.img_per_epoch // args.batch_size
    data_iterator = iter(data_loader)

    for batch_idx in range(num_batches):
        try:
            x, _ = next(data_iterator)

            x = x.to(args.device)
            optimizer.zero_grad()

            # Forward Pass
            fwd_clock.tic()
            x_tilde, z_app_mean, z_app_var, z_vis_mean = model(x, args.temp)

            # Compute Loss
            loss, loss_dict = loss_function(
                x_tilde, x, z_app_mean, z_app_var, z_vis_mean,
                categorical=args.categorical, py=args.py, beta_p=args.beta_p,
                beta_a=args.beta_a, beta_v=args.beta_v,
                beta_ea=args.beta_ea, beta_ew=args.beta_ew
            )
            fwd_clock.toc()

            # Backprop
            bwd_clock.tic()
            loss.backward()
            bwd_clock.toc()

            # Update Adam
            optimizer.step()

            # Logs
            losses['loss'] += loss.item()
            writer.add_scalar('loss/train/loss', loss.item(), args.steps)
            for loss_key, loss_value in loss_dict.items():
                writer.add_scalar('loss/train/' + loss_key, loss_value.item(), args.steps)
                losses[loss_key] = losses[loss_key] + loss_value.item() \
                    if loss_key in losses else loss_value.item()

            args.steps += 1

            if args.steps % 1000 == 1:
                args.temp = max(args.temp * np.exp(-args.anneal * args.steps),
                                args.min_temp)

            if batch_idx % args.log_interval != 0:
                continue

            logstr = '\t'.join(['{}: {:0.4f}'.format(k, v.item()) for k, v in loss_dict.items()])
            print('[{}/{} ({:0.0f}%)]\t{}'.format(batch_idx, num_batches,
                  100. * batch_idx / num_batches, logstr))

        except StopIteration:
            data_iterator = iter(data_loader)

    losses = OrderedDict([(k, v / num_batches) for k, v in losses.items()])
    logstr = '\t'.join(['{}: {:0.4f}'.format(k, v) for k, v in losses.items()])

    print('[End of train epoch]\t# steps: {}\t# images: {}, temp: {:0.2f}'.format(
        args.steps, num_batches * args.batch_size, args.temp))
    print(logstr)
    print('[End of train epoch]\t# calls: {}, Fwd: {:.3f} ms\tBwd: {:.3f} ms'.format(
        fwd_clock.calls, 1000 * fwd_clock.average_time, 1000 * bwd_clock.average_time))

    return losses['loss']


def test(data_loader, model, loss_function, writer):
    model.eval()
    losses = OrderedDict()
    losses['loss'] = 0

    data_iterator = iter(data_loader)
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(data_iterator):
            x = x.to(args.device)
            x_tilde, z_app_mean, z_app_var, z_vis_mean = model(x, args.temp)
            loss, loss_dict = loss_function(
                x_tilde, x, z_app_mean, z_app_var, z_vis_mean,
                categorical=args.categorical, py=args.py, beta_p=args.beta_p,
                beta_a=args.beta_a, beta_v=args.beta_v,
                beta_ea=args.beta_ea, beta_ew=args.beta_ew
            )

            losses['loss'] += loss.item()
            for loss_key, loss_value in loss_dict.items():
                losses[loss_key] = losses[loss_key] + loss_value.item() \
                    if loss_key in losses else loss_value.item()

    losses = OrderedDict([(k, v / (batch_idx+1)) for k, v in losses.items()])
    logstr = '\t'.join(['{}: {:0.4f}'.format(k, v) for k, v in losses.items()])

    print('[End of test epoch]')
    print(logstr)

    # Logs
    for loss_key, loss_value in losses.items():
        writer.add_scalar('loss/test/' + loss_key, loss_value, args.steps)

    return losses['loss']


def plot_graph(height, width, channels, model, writer):
    fake = torch.from_numpy(np.random.randn(args.batch_size,
                            channels, height, width).astype(np.float32))
    fake = fake.to(args.device)
    writer.add_graph(model, fake)


def main():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.steps = 0

    writer = SummaryWriter(args.log_dir)
    save_filename = args.model_dir

    train_loader, test_loader, (channels, height, width), num_classes, _ = \
        data_loaders(args.dataset, data_folder=args.data_folder,
                     classify=False, size=args.size, inet=args.inet,
                     batch_size=args.batch_size, num_workers=args.workers)

    # Fixed images for Tensorboard
    fixed_images, _ = next(iter(test_loader))
    fixed_images = fixed_images.to(args.device)
    fixed_grid = make_grid(commons.unnorm(fixed_images).cpu().data, nrow=32, pad_value=1)
    writer.add_image('original', fixed_grid, 0)

    # build a VAE model
    vae_model, _ = load_vae_model((channels, height, width),
                                  args.arch,
                                  encoder_arch=args.encoder_arch,
                                  decoder_arch=args.decoder_arch,
                                  hidden_size=args.hidden_size,
                                  num_parts=args.num_parts,
                                  base_depth=args.ngf,
                                  independent=args.independent,
                                  hard=args.hard,
                                  categorical=args.categorical,
                                  scale=args.scale,
                                  device=args.device)
    args.py = 1 / args.num_parts if args.py is None else args.py

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        vae_model = nn.DataParallel(vae_model)

    vae_model.to(args.device)

    if args.pretrained is not None:
        print("Loading pretrained model from %s" % args.pretrained)
        pretrained_dict = torch.load(args.pretrained, map_location=args.device)
        if type(pretrained_dict) == OrderedDict:
            vae_model.load_state_dict(pretrained_dict)
        elif 'vae_dict' in pretrained_dict:
            vae_model.load_state_dict(pretrained_dict['vae_dict'])
        else:
            print('debug')
            sys.exit(0)

    # Generate samples only, no training
    if args.evaluate:
        with torch.no_grad():
            # Reconstructions after current epoch
            if torch.cuda.device_count() > 1:
                reconstructions = vae_model.module.get_reconstructions(
                    fixed_images, temp=args.temp)
            else:
                reconstructions = vae_model.get_reconstructions(
                    fixed_images, temp=args.temp)

            for key in reconstructions:
                grid = make_grid(reconstructions[key].cpu(), nrow=32, pad_value=1)
                writer.add_image(key, grid, 0)

            # Random samples after current epoch
            if torch.cuda.device_count() > 1:
                random_samples = vae_model.module.get_random_samples(py=args.py)
            else:
                random_samples = vae_model.get_random_samples(py=args.py)
            for key in random_samples:
                grid = make_grid(random_samples[key].cpu(), nrow=32, pad_value=1)
                writer.add_image(key, grid, 0)
        sys.exit(0)

    opt_v = torch.optim.Adam(vae_model.parameters(), lr=args.lr, betas=(0.5, 0.999))

    recon_mask = None
    if args.recon_mask == 'edge':
        recon_mask = EdgeWeights(nc=channels, scale=args.scale)

    if args.arch == 'vae':
        loss_function = BetaVaeLoss(beta=args.beta_a, mask_nn=recon_mask)
    elif args.arch == 'convvae':
        loss_function = VaeConcreteLoss(
                beta_v=args.beta_v,
                py=args.py,
                categorical=args.categorical,
                mask_nn=recon_mask
            )
    elif args.arch == 'patchy':
        if args.beta_p == 0. and args.beta_ea == 0. and args.beta_ew == 0.:
            loss_function = BetaVaeConcreteLoss(
                beta_a=args.beta_a,
                beta_v=args.beta_v,
                py=args.py,
                categorical=args.categorical,
                mask_nn=recon_mask
            )
        elif args.beta_ea == 0. and args.beta_ew == 0.:
            loss_function = BetaVaeConcretePartsLoss(
                beta_a=args.beta_a,
                beta_v=args.beta_v,
                beta_p=args.beta_p,
                py=args.py,
                categorical=args.categorical,
            )
        else:
            loss_function = BetaVaeConcretePartsEntropyLoss(
                beta_a=args.beta_a,
                beta_v=args.beta_v,
                beta_p=args.beta_p,
                beta_ea=args.beta_ea,
                beta_ew=args.beta_ew,
                py=args.py,
                categorical=args.categorical,
            )

    else:
        print('Unknown model architecture: %s' % args.arch)
        sys.exit(0)

    if args.gan:
        gan_model = Discriminator(height, nc=channels, ndf=args.ndf, scale=args.scale).to(args.device)
        opt_d = torch.optim.Adam(gan_model.parameters(), lr=args.lr, betas=(0.5, 0.999))
        d_loss_fn = DiscLoss(args.beta_g)

    # test after seeing approx. every 50000 images
    # num_epochs = (args.num_epochs * len(train_loader.dataset)) // 50000

    for epoch in range(1, args.num_epochs + 1):
        print("================== Epoch: {} ==================".format(epoch))
        if args.gan:
            train_loss = train_vaegan(train_loader, gan_model, vae_model, opt_d, opt_v, d_loss_fn, loss_function, writer)
        else:
            train_loss = train(train_loader, vae_model, opt_v, loss_function, writer)
        test_loss = test(test_loader, vae_model, loss_function, writer)

        if epoch == 1:
            best_loss = test_loss

        if epoch % args.save_interval != 0:
            continue

        # Save model
        with torch.no_grad():
            # Reconstructions after current epoch
            if torch.cuda.device_count() > 1:
                reconstructions = vae_model.module.get_reconstructions(
                    fixed_images, temp=args.temp)
            else:
                reconstructions = vae_model.get_reconstructions(
                    fixed_images, temp=args.temp)

            for key in reconstructions:
                grid = make_grid(reconstructions[key].cpu(), nrow=32, pad_value=1, normalize=True)
                writer.add_image(key, grid, epoch)

            # Random samples after current epoch
            if torch.cuda.device_count() > 1:
                random_samples = vae_model.module.get_random_samples(py=args.py)
            else:
                random_samples = vae_model.get_random_samples(py=args.py)
            for key in random_samples:
                grid = make_grid(random_samples[key].cpu(), nrow=32, pad_value=1, normalize=True)
                writer.add_image(key, grid, epoch)

        f = '{0}/model_{1}.pt'.format(save_filename, epoch)
        save_state = {
            'args': args,
            'vae_dict': vae_model.state_dict(),
            'loss': train_loss,
        }
        if args.gan:
            save_state['disc_dict'] = gan_model.state_dict()
        torch.save(save_state, f)

        if test_loss < best_loss:
            best_loss = test_loss
            shutil.copyfile(f, '{0}/best.pt'.format(save_filename))

    print("Model saved at: {0}/best.pt".format(save_filename))
    print("# Parameters: {}".format(count_parameters(vae_model)))
    if torch.cuda.device_count() > 1:
        summary(vae_model.module, (channels, height, width))
    else:
        summary(vae_model, (channels, height, width))


if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Patchy VAE')

    # Dataset
    parser.add_argument('--dataset', type=str, default='cifar100',
                        help='name of the dataset (default: cifar100)')
    parser.add_argument('--data-folder', type=str, default='./data',
                        help='name of the data folder (default: ./data)')
    parser.add_argument('--workers', type=int, default=4,
                        help='number of threads (default: 4)')
    parser.add_argument('--pretrained', default=None,
                        help='path of pre-trained model')
    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='just sample no training (default: False)')
    parser.add_argument('--size', type=int, default=64,
                        help='size of image (default: 64)')
    parser.add_argument('--inet', default=False, action='store_true',
                        help='Whether or not to do imagenet normalization')

    # Model
    parser.add_argument('--arch', type=str, default='patchy',
                        help='model architecture (default: patchy)')
    parser.add_argument('--encoder-arch', type=str, default='resnet',
                        help='encoder architecture (default: resnet)')
    parser.add_argument('--decoder-arch', type=str, default='pyramid',
                        help='decoder architecture (default: pyramid)')
    parser.add_argument('--independent', action='store_true', default=False,
                        help='independent decoders (default: False)')
    parser.add_argument('--ngf', type=int, default=64,
                        help='depth of first layer of encoder (default: 64)')

    # Optimization
    parser.add_argument('--recon-mask', type=str, default=None,
                        help="Use 'edge' mask for improved reconstruction (default: None.)")
    parser.add_argument('--batch-size', type=int, default=128,
                        help='batch size (default: 128)')
    parser.add_argument('--img-per-epoch', type=int, default=50000,
                        help='images per epoch (default: 50000)')
    parser.add_argument('--num-epochs', type=int, default=30,
                        help='number of epochs (default: 30)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training (default: False)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate for Adam optimizer (default: 1e-4)')
    parser.add_argument('--beta-a', type=float, default=1.0,
                        help='contribution of KLD App loss (default: 1.0)')
    parser.add_argument('--beta-v', type=float, default=10.,
                        help='contribution of KLD Vis loss (default: 10.)')
    parser.add_argument('--beta-p', type=float, default=0.,
                        help='contribution of MSE Parts loss (default: 0.)')
    parser.add_argument('--beta-ea', type=float, default=0.,
                        help='contribution of Entropy Across loss (default: 0.)')
    parser.add_argument('--beta-ew', type=float, default=0.,
                        help='contribution of Entropy Within loss (default: 0.)')

    # GAN
    parser.add_argument('--gan', action='store_true', default=False,
                        help='enable gan (default: False)')
    parser.add_argument('--ndf', type=int, default=64,
                        help='depth of first layer of discrimnator (default: 64)')
    parser.add_argument('--beta-g', type=float, default=1.0,
                        help='contribution of GAN loss (default: 0.)')

    # Latent space
    parser.add_argument('--scale', type=int, default=8,
                        help='scale down by (default: 8)')
    parser.add_argument('--num-parts', type=int, default=16,
                        help='number of parts (default: 16)')
    parser.add_argument('--hidden-size', type=int, default=6,
                        help='size of the latent vectors (default: 6)')
    parser.add_argument('--py', type=float, default=None,
                        help='part visibility prior (default: 1 / num_parts)')
    parser.add_argument('--categorical', action='store_true', default=False,
                        help='take only 1 part per location (default: False)')

    # Annealing
    parser.add_argument('--hard', action='store_true', default=False,
                        help='hard samples from bernoulli (default: False)')
    parser.add_argument('--temp', type=float, default=1.0,
                        help='Initial temperature (default: 1.0)')
    parser.add_argument('--anneal', type=float, default=0.00003,
                        help='Anneal rate (default: 00003)')
    parser.add_argument('--min-temp', type=float, default=0.1,
                        help='minimum temperature')

    # Miscellaneous
    parser.add_argument('--debug-grad', action='store_true', default=False,
                        help='debug gradients (default: False)')
    parser.add_argument('--output-folder', type=str, default='./scratch',
                        help='name of the output folder (default: ./scratch)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=50,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-interval', type=int, default=1,
                        help='how many batches to wait before logging training status')

    args = parser.parse_args()

    print("All arguments")
    print(args)
    print("PID: ", os.getpid())

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda:0"
                               if args.cuda and torch.cuda.is_available() else "cpu")

    # Slurm
    if 'SLURM_JOB_NAME' in os.environ and 'SLURM_JOB_ID' in os.environ:
        # running with sbatch and not srun
        if os.environ['SLURM_JOB_NAME'] != 'bash':
            args.output_folder = os.path.join(args.output_folder,
                                              os.environ['SLURM_JOB_ID'])
            print("SLURM_JOB_ID: ", os.environ['SLURM_JOB_ID'])
        else:
            args.output_folder = os.path.join(args.output_folder, str(os.getpid()))
    else:
        args.output_folder = os.path.join(args.output_folder, str(os.getpid()))

    # Create logs and models folder if they don't exist
    if not os.path.exists(args.output_folder):
        print("Creating output directory: %s" % args.output_folder)
        os.makedirs(args.output_folder)

    log_dir = os.path.join(args.output_folder, 'logs')
    if not os.path.exists(log_dir):
        print("Creating log directory: %s" % log_dir)
        os.makedirs(log_dir)

    model_dir = os.path.join(args.output_folder, 'models')
    if not os.path.exists(model_dir):
        print("Creating model directory: %s" % model_dir)
        os.makedirs(model_dir)

    args.log_dir = log_dir
    args.model_dir = model_dir

    main()
