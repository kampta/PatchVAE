#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" model.py
Definition of different VAE models
"""

__author__ = "Kamal Gupta"
__email__ = "kampta@cs.umd.edu"
__version__ = "0.1"

import numpy as np
import torch
from torch import nn
import torch.distributions as td
from torch.nn.functional import interpolate

from utils.commons import init_weights, rf
import utils.commons as commons
import pdb

eps = 1e-7
float_max = 1e7


class Discriminator(nn.Module):
    def __init__(self, image_size, nc=3, ndf=64, scale=8):
        super(Discriminator, self).__init__()
        n = int(np.log2(image_size))

        self.features = commons.make_encoder(nc, ndf, arch='pyramid', scale=scale)

        self.features.add_module('output-conv', nn.Conv2d(ndf * 2 ** (n - 4), 1, 1, bias=False))
        self.features.add_module('output-pool', nn.AvgPool2d(image_size // scale))
        # self.features.add_module('output-sigmoid', nn.Sigmoid())

    def forward(self, input):
        output = self.features(input)
        return output.view(-1, 1)


def reparameterize(mu, var):
    # std = torch.exp(0.5*logvar)
    std = var.pow(0.5)
    eps = torch.randn_like(std)
    return mu + eps * std


def straight_through(logits):
    # Reparameterize
    u = torch.rand_like(logits)
    q_y = torch.sigmoid(logits) + u
    # stop_gradient trick
    # There is no step function in torch so using sign
    q_y_hard = q_y + (0.5 * (torch.sign(logits) + 1) - q_y).detach()
    return q_y_hard


class PatchyVAE(nn.Module):

    def __init__(self, input_size=(1, 32, 32),
                 base_depth=16, hidden_size=32, num_parts=10, independent=False,
                 hard=False, categorical=False,
                 encoder_arch='pyramid',
                 decoder_arch='pyramid',
                 scale=8, **kwargs):
        super(PatchyVAE, self).__init__()

        self.C = input_size[0]
        self.H = input_size[1]
        self.W = input_size[2]
        self.L = hidden_size
        self.P = num_parts
        self.scale = scale
        self.independent = independent
        self.hard = hard
        self.categorical = categorical
        self.interpolate_output = False

        self.BH = self.H // scale   # Bottleneck height
        self.BW = self.W // scale   # Bottleneck width

        if encoder_arch == 'resnet':
            # just keep as many residual blocks from resnet18 as required
            encoder_out_channels = 16 * scale
        elif encoder_arch == 'alexnet':
            encoder_out_channels = 256
            self.scale = 32
            self.BH = 6
            self.BW = 6
            self.interpolate_output = True
        elif encoder_arch == 'resnet18':
            # keep the complete resnet 18 but remove the downsampling
            encoder_out_channels = 512
            self.scale = 32
        elif encoder_arch == 'resnet50':
            # keep the complete resnet 18 but remove the downsampling
            encoder_out_channels = 2048
            self.scale = 32
        else:
            encoder_out_channels = base_depth * scale // 2

        self.bottleneck = encoder_out_channels * self.BH * self.BW

        """
        Encoder stuff
        features
        app_mu
        app_logvar
        vis_mean logit
        """

        self.features = commons.make_encoder(
            self.C, base_depth, arch=encoder_arch, scale=scale)

        # Appearance factors
        self.app_mu = nn.Sequential(
            nn.Conv2d(encoder_out_channels, num_parts * hidden_size, 3, 1, 1),
        )

        self.app_logvar = nn.Sequential(
            nn.Conv2d(encoder_out_channels, num_parts * hidden_size, 3, 1, 1),
        )

        # Visibility factors
        self.vis_mean_logit = nn.Sequential(
            nn.Conv2d(encoder_out_channels, num_parts, 3, 1, 1),
        )

        # Receptive field
        encoder = nn.Sequential(self.features, self.vis_mean_logit)
        self.rf = rf(encoder)

        """
        Decoder stuff
        """

        # Convert Z into a feature map before applying a decoder
        groups = self.P if self.independent else 1

        self.decoder = commons.make_decoder(
            self.C, base_depth, arch=decoder_arch, groups=groups,
            nz=num_parts * hidden_size, scale=scale)

        self.features.apply(init_weights)
        self.app_mu.apply(init_weights)
        self.app_logvar.apply(init_weights)
        self.vis_mean_logit.apply(init_weights)
        self.decoder.apply(init_weights)

    def encode(self, x):
        encode = self.features(x)

        # Appearance encoder
        app_mu = self.app_mu(encode)
        app_logvar = self.app_logvar(encode)

        # Visibility encoder
        vis_mean_logit = self.vis_mean_logit(encode)
        return app_mu, app_logvar, vis_mean_logit

    def decode(self, z):
        decoded = self.decoder(z)
        if self.interpolate_output:
            return interpolate(decoded, size=(self.H, self.W), mode='bilinear')
        else:
            return self.decoder(z)

    def analyze(self, x, temp=0.5):
        app_mu, app_logvar, vis_mean_logit = self.encode(x)

        if self.categorical:
            flatten = vis_mean_logit.permute(0, 2, 3, 1)\
                .reshape(-1, self.BH * self.BW, self.P)
            q_z_vis = td.relaxed_categorical.RelaxedOneHotCategorical(temp, logits=flatten)
            vis_mean = q_z_vis.probs.reshape(-1, self.BH, self.BW, self.P).permute(0, 3, 1, 2)
            z_vis = q_z_vis.rsample().reshape(-1, self.BH, self.BW, self.P).permute(0, 3, 1, 2)

        else:
            q_z_vis = td.relaxed_bernoulli.RelaxedBernoulli(temp, logits=vis_mean_logit)
            vis_mean = q_z_vis.probs
            z_vis = q_z_vis.rsample()
            
        return app_mu, vis_mean

    def forward(self, x, temp=0.5):
        # with torch.autograd.profiler.profile(use_cuda=True) as prof:
        app_mu, app_logvar, vis_mean_logit = self.encode(x)

        if self.categorical:
            flatten = vis_mean_logit.permute(0, 2, 3, 1)\
                .reshape(-1, self.BH * self.BW, self.P)
            q_z_vis = td.relaxed_categorical.RelaxedOneHotCategorical(temp, logits=flatten)
            vis_mean = q_z_vis.probs.reshape(-1, self.BH, self.BW, self.P).permute(0, 3, 1, 2)
            z_vis = q_z_vis.rsample().reshape(-1, self.BH, self.BW, self.P).permute(0, 3, 1, 2)

        else:
            q_z_vis = td.relaxed_bernoulli.RelaxedBernoulli(temp, logits=vis_mean_logit)
            vis_mean = q_z_vis.probs
            z_vis = q_z_vis.rsample()

        # straight through
        # z_vis_hard = 0.5 * (torch.sign(vis_mean_logit) + 1)   # no step function in torch, so using sign
        # z_vis_hard = z_vis + (z_vis_hard - z_vis).detach()    # backpropagatable z_vis_hard

        if self.hard:
            # z_vis = z_vis_hard
            z_vis_hard = 0.5 * (torch.sign(vis_mean_logit) + 1)   # no step function in torch, so using sign
            z_vis = z_vis + (z_vis_hard - z_vis).detach()    # backpropagatable z_vis_hard

        z_vis_expand = z_vis[:, :, None, :, :]
        z_vis_expand = z_vis_expand.expand(-1, -1, self.L, -1, -1)
        z_vis_expand = z_vis_expand.reshape(-1, self.L * self.P, self.BH, self.BW)

        """
        appearance pooling (1) weighted pooling according to prob vis sample
        """
        # vis_mean_detach = vis_mean
        vis_mean_detach = vis_mean.detach()
        vis_mean_detach = vis_mean_detach[:, :, None, :, :]
        vis_mean_detach = vis_mean_detach.expand(-1, -1, self.L, -1, -1)
        vis_mean_detach = vis_mean_detach.reshape(-1, self.L * self.P, self.BH, self.BW)

        app_var = torch.clamp(app_logvar.exp(), min=eps, max=float_max)

        app_var_weighted = torch.mul(app_var, vis_mean_detach)
        app_mu_weighted = torch.mul(app_mu, vis_mean_detach)

        vis_mean_detach_sum = torch.sum(vis_mean_detach, (2, 3)) + eps
        app_var_weighted = (torch.sum(app_var_weighted, (2, 3)) + eps) / vis_mean_detach_sum
        app_mu_weighted = torch.sum(app_mu_weighted, (2, 3)) / vis_mean_detach_sum
        z_app = reparameterize(app_mu_weighted, app_var_weighted)

        # app_std_weighted = app_var_weighted.pow(0.5)
        # q_z_app = td.normal.Normal(app_mu_weighted, app_std_weighted)
        # z_app = q_z_app.rsample()

        z_app = z_app[:, :, None, None]
        z_app_expand = z_app.expand(-1, -1, self.BH, self.BW)

        z_app_vis = torch.mul(z_app_expand, z_vis_expand)

        recon_x = self.decode(z_app_vis)

        return recon_x, app_mu_weighted, app_var_weighted, vis_mean

    def get_reconstructions(self, x, temp=0.5):
        reconstructions = {}

        # x = x.to(device)

        # Reconstruction samples after current epoch
        recon_images, z_app_mean, z_app_std, vis_mean = self.forward(x, temp)

        scale_factor = self.H / self.BH
        reconstructions['reconstruction_image'] = commons.unnorm(recon_images)

        # Visibility means
        vis_mean = interpolate(vis_mean, scale_factor=scale_factor, mode='nearest')
        vis_mean_first_row = vis_mean[:32, :, :, :].cpu()
        h, w = vis_mean_first_row.shape[-2:]
        vis_mean_first_row = np.reshape(vis_mean_first_row, (-1, 1, h, w), order='F')
        reconstructions['reconstruction_bottleneck_soft'] = vis_mean_first_row

        return reconstructions

    def get_random_samples(self, py=0.5):
        random_samples = {}

        samples = self.generate_samples(py=py, batch_size=128)
        random_samples['samples_random'] = samples

        # Random samples for each part after current epoch
        # samples = self.generate_part_samples(py=py)
        # random_samples['samples_part_random'] = samples

        # Random samples for each part location after current epoch
        # samples = self.generate_part_location_samples()
        # random_samples['samples_part_location_random'] = samples

        return random_samples

    def generate_samples(self, py, batch_size=32):
        z_app = np.float32(np.random.randn(batch_size, self.L * self.P, 1, 1))

        z_vis = []
        for _ in range(batch_size):
            sample = []
            for _ in range(self.P):
                img = np.random.binomial(1, py, size=(self.BH, self.BW))
                img = np.repeat(img[np.newaxis, :, :], self.L, axis=0)
                sample.append(img)
            sample = np.array(sample)
            sample = sample.reshape(self.L * self.P, self.BH, self.BW)
            z_vis.append(sample)

        z_vis = np.float32(np.array(z_vis))

        z_app = torch.from_numpy(z_app)
        z_app_expand = z_app.expand(-1, -1, self.BH, self.BW)
        z_vis = torch.from_numpy(z_vis)
        z_app = torch.mul(z_app_expand, z_vis)

        # z_app = z_app.to(device)
        if torch.cuda.is_available():
            z_app = z_app.cuda()
        samples = self.decode(z_app)
        samples = commons.unnorm(samples)
        return samples.cpu().data.view(batch_size, self.C, self.H, self.W)

    def generate_part_samples(self, py):
        samples_per_part = 32
        batch_size = self.P * samples_per_part

        z_app = np.float32(np.random.randn(batch_size, self.L * self.P, 1, 1))

        # z_vis = []
        # for p_idx in range(self.P):
        #     for s_idx in range(samples_per_part):
        #         sample = self.generate_z_vis(p_idx, py)
        #         z_vis.append(sample)

        z_vis = [self.generate_z_vis(p_idx, py)
                 for p_idx in range(self.P)
                 for _ in range(samples_per_part)]

        z_app = torch.from_numpy(z_app)
        z_app_expand = z_app.expand(-1, -1, self.BH, self.BW)
        z_vis = torch.from_numpy(np.float32(np.array(z_vis)))
        z_app_vis = torch.mul(z_app_expand, z_vis)

        # z_app_vis = z_app_vis.to(device)
        if torch.cuda.is_available():
            z_app_vis = z_app_vis.cuda()
        samples, _ = self.decode(z_app_vis)
        return samples.cpu().data.view(batch_size, self.C, self.H, self.W)

    def generate_part_location_samples(self):
        samples_per_part = 32
        batch_size = self.P * samples_per_part

        z_app = np.random.randn(batch_size, self.L * self.P, 1, 1)
        z_vis = np.zeros((batch_size, self.L * self.P, self.BH, self.BW))

        for p_idx in range(self.P):
            for row in range(samples_per_part):
                image_idx = p_idx * samples_per_part + row
                mask_row_idx = (self.BH * self.BW // samples_per_part * row) // self.BW
                mask_col_idx = (self.BH * self.BW // samples_per_part * row) % self.BW
                z_vis[image_idx, p_idx * self.L: p_idx * self.L + self.L, mask_row_idx, mask_col_idx] = 1.

        z_vis = torch.from_numpy(np.float32(z_vis))
        z_app = torch.from_numpy(np.float32(z_app))
        z_app_expand = z_app.expand(-1, -1, self.BH, self.BW)
        z_app_vis = torch.mul(z_app_expand, z_vis)

        # z_app_vis = z_app_vis.to(device)
        if torch.cuda.is_available():
            z_app_vis = z_app_vis.cuda()
        _, recon_parts = self.decode(z_app_vis)

        samples = torch.zeros(batch_size, self.C, self.H, self.W)
        for p_idx in range(self.P):
            for row in range(samples_per_part):
                image_idx = p_idx * samples_per_part + row
                samples[image_idx, :, :, :] = recon_parts[image_idx, p_idx * self.C: p_idx * self.C + self.C, :, :]
        return samples.data.view(batch_size, self.C, self.H, self.W)

    def generate_z_vis(self, part_idx=0, py=0.5):
        sample = []
        for idx in range(self.P):
            if idx == part_idx:
                img = np.random.binomial(1, py, size=(self.BH, self.BW))
                part = np.repeat(img[np.newaxis, :, :], self.L, axis=0)
                sample.append(part)
            else:
                part = np.zeros((self.L, self.BH, self.BW))
                sample.append(part)

        return np.array(sample).reshape(-1, self.BH, self.BW)


class VAE(nn.Module):
    def __init__(self, input_size=(1, 32, 32),
                 base_depth=16, hidden_size=32, num_parts=10,
                 encoder_arch='tinyimage_5layer',
                 decoder_arch='tinyimage_5layer',
                 scale=8, **kwargs):
        super(VAE, self).__init__()
        self.C = input_size[0]
        self.H = input_size[1]
        self.W = input_size[2]
        self.scale = scale
        self.BH = self.H // scale  # Bottleneck height
        self.BW = self.W // scale  # Bottleneck width

        self.hidden = hidden_size * num_parts
        if encoder_arch == 'resnet':
            # just keep as many residual blocks from resnet18 as required
            encoder_out_channels = 16 * scale
        elif encoder_arch == 'resnet18':
            # keep the complete resnet 18 but remove the downsampling
            encoder_out_channels = 512
        elif encoder_arch == 'alexnet':
            encoder_out_channels = 256
        else:
            encoder_out_channels = base_depth * scale // 2

        self.bottleneck = encoder_out_channels * self.BH * self.BW

        self.features = commons.make_encoder(self.C, base_depth, arch=encoder_arch, scale=scale)

        self.reduce = nn.Sequential(
            nn.Conv2d(encoder_out_channels, base_depth, 1, bias=False),
            nn.BatchNorm2d(base_depth),
            nn.ReLU(inplace=True)
        )

        self.app_mu = nn.Sequential(
            nn.Conv2d(base_depth, self.hidden, self.BH),
        )

        self.app_logvar = nn.Sequential(
            nn.Conv2d(base_depth, self.hidden, self.BH),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.hidden, base_depth, self.BH, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            commons.make_decoder(self.C, base_depth, arch=decoder_arch,
                                 nz=base_depth, scale=scale)
        )

        self.features.apply(init_weights)
        self.reduce.apply(init_weights)
        self.app_mu.apply(init_weights)
        self.app_logvar.apply(init_weights)
        self.decoder.apply(init_weights)

    def encode(self, x):
        encode = self.reduce(self.features(x))

        # Appearance encoder
        app_mu = self.app_mu(encode)
        app_logvar = self.app_logvar(encode)

        return app_mu, app_logvar

    def decode(self, z):
        # z_map = self.z_map(z)
        # return torch.sigmoid(self.decoder(z_map))
        return self.decoder(z)

    def forward(self, x, temp=0.5):
        app_mu, app_logvar = self.encode(x)
        app_var = app_logvar.exp()

        # q_z_app = td.normal.Normal(app_mu, app_std)
        # z_app = q_z_app.rsample()
        z_app = reparameterize(app_mu, app_var)

        recon_x = self.decode(z_app)

        # recon_x, app_mu_weighted, app_var_weighted, vis_mean
        return recon_x, app_mu, app_var, torch.zeros_like(app_mu)

    def generate_samples(self, batch_size=32):
        # z_app = np.float32(np.random.randn(batch_size, self.hidden, self.H // 8, self.W // 8))
        z_app = np.float32(np.random.randn(batch_size, self.hidden, 1, 1))
        z_app = torch.from_numpy(z_app)
        # z_app = z_app.to(device)

        if torch.cuda.is_available():
            z_app = z_app.cuda()

        sample = commons.unnorm(self.decode(z_app))

        return sample.cpu().data.view(batch_size, self.C, self.H, self.W)

    def get_reconstructions(self, x, **kwargs):
        reconstructions = {}
        # x = x.to(device)

        # Reconstruction samples after current epoch
        # recon_x, q_z_app, q_z_vis = self.forward(x)
        recon_x, _, _, _ = self.forward(x)

        # recon_images, _ = recon_x

        reconstructions['reconstruction_image'] = commons.unnorm(recon_x)

        return reconstructions

    def get_random_samples(self, **kwargs):
        random_samples = {}

        samples = self.generate_samples(batch_size=128)
        random_samples['samples_random'] = samples

        return random_samples
