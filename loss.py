from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as td
from torch.nn.functional import interpolate

eps = 1e-7


class BceLoss(nn.Module):
    """
    Numerically stable version of the binary cross-entropy loss function.
    As per https://github.com/pytorch/pytorch/issues/751
    See the TensorFlow docs for a derivation of this formula:
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
    Inputs:
    - input: PyTorch Tensor of shape (N, ) giving scores.
    - target: PyTorch Tensor of shape (N,) containing 0 and 1 giving targets.
    Returns:
    - A PyTorch Tensor containing the mean BCE loss over the minibatch of
      input data.
    """

    def __init__(self):
        super(BceLoss, self).__init__()

    def forward(self, prediction, target):
        neg_abs = -prediction.abs()
        loss = prediction.clamp(min=0) - prediction * target + (1 + neg_abs.exp()).log()
        return loss.mean()


class MseLoss(nn.Module):
    def __init__(self):
        super(MseLoss, self).__init__()

    def forward(self, recon_x, x):
        loss = F.mse_loss(recon_x, x, reduction='sum')
        return loss / x.size(0)


class WeightedMseLoss(nn.Module):
    def __init__(self, weight_module):
        super(WeightedMseLoss, self).__init__()
        self.weight_nn = weight_module

    def forward(self, recon_x, x):
        weights = self.weight_nn(x)
        loss = F.mse_loss(recon_x, x, reduction='none')
        loss = torch.mul(loss, weights)
        return loss.sum() / x.size(0)


class ReconPartsLoss(nn.Module):
    def __init__(self):
        super(ReconPartsLoss, self).__init__()

    def forward(self, recon_parts, x, z_vis):
        # stop gradients
        z_vis = z_vis.detach()

        # batch_size x channels x height x width
        num_channels = x.size(1)
        height = x.size(2)
        width = x.size(3)
        num_parts = z_vis.size(1)

        z_vis_expand = z_vis[:, :, None, :, :]
        z_vis_expand = z_vis_expand.expand(-1, -1, num_channels, -1, -1)
        z_vis_expand = z_vis_expand.reshape(-1, num_parts * num_channels,
                                            height // 8, width // 8)
        scale_factor = height / (height // 8)
        z_vis_expand = interpolate(z_vis_expand, scale_factor=scale_factor, mode='nearest')
        parts_target = torch.mul(x.repeat(1, num_parts, 1, 1), z_vis_expand)
        parts_prediction = torch.mul(recon_parts, z_vis_expand)

        # divide only where mask is 1
        num_visible = torch.sum(z_vis_expand) + eps
        bce_parts = F.mse_loss(parts_prediction, parts_target, reduction='sum')
        bce_parts /= num_visible
        return bce_parts


class KldNormalLoss(nn.Module):
    def __init__(self):
        super(KldNormalLoss, self).__init__()

    def forward(self, mu, var):
        # p_z = td.normal.Normal(torch.zeros_like(q_z.loc), torch.ones_like(q_z.scale))
        # loss = td.kl_divergence(q_z, p_z).sum() / q_z.loc.size(0)
        loss = -0.5 * torch.sum(1 + var.log() - mu.pow(2) - var) / mu.size(0)
        return loss


class KldCategoricalLoss(nn.Module):
    def __init__(self, py=0.5):
        super(KldCategoricalLoss, self).__init__()
        self.py = py

    def forward(self, vis_mean):
        t = vis_mean * ((vis_mean + eps) / self.py).log()
        return torch.sum(t, dim=-1).sum() / vis_mean.size(0)


class KldBinaryLoss(nn.Module):
    def __init__(self, py=0.5):
        super(KldBinaryLoss, self).__init__()
        self.py = py

    def forward(self, vis_mean):
        t1 = vis_mean * ((vis_mean + eps) / self.py).log()
        t2 = (1 - vis_mean) * ((1 - vis_mean + eps) / (1 - self.py)).log()
        # return torch.mean(t1 + t2)
        return torch.sum(t1 + t2, dim=-1).sum() / vis_mean.size(0)


# Entropy across images, maximize so that different images use different parts
class EntAcrossLoss(nn.Module):
    def __init__(self):
        super(EntAcrossLoss, self).__init__()

    def forward(self, z_vis):
        # Dimensions are batch_size, num_parts, BH, BW
        batch_size = z_vis.size(0)

        # Move batch_size to the end and
        # reshape to (num_parts * BH * BW) x batch_size
        z_vis = z_vis.permute(1, 2, 3, 0).reshape(-1, 2)

        # Convert to probabilities
        p_z_vis = z_vis.sum(dim=-1) / batch_size + eps

        # Compute \sum p * log p (No need to negate since we want to maximize
        p_logp = p_z_vis * p_z_vis.log()
        return p_logp.sum() / z_vis.size(0)


# Entropy within images, maximize so that image doesn't have ones for entire part
class EntWithinLoss(nn.Module):
    def __init__(self):
        super(EntWithinLoss, self).__init__()

    def forward(self, z_vis):
        # Dimensions are batch_size, num_parts, BH, BW
        batch_size = z_vis.size(0)
        num_parts = z_vis.size(1)
        height = z_vis.size(2)
        width = z_vis.size(3)

        # Reshape to (batch_size * num_parts) x (BH * BW)
        z_vis = z_vis.reshape(batch_size * num_parts, height * width)

        # Convert to probabilities
        p_z_vis = z_vis.sum(dim=-1) / height / width + eps

        # Compute \sum p * log p (No need to negate since we want to maximize
        p_logp = p_z_vis * p_z_vis.log()
        return p_logp.sum() / z_vis.size(0)


class BetaVaeLoss(nn.Module):
    def __init__(self, beta=1.0, mask_nn=None):
        super(BetaVaeLoss, self).__init__()
        print('BetaVaeLoss')
        self.beta = beta
        if mask_nn:
            self.recon = WeightedMseLoss(mask_nn)
        else:
            self.recon = MseLoss()
        self.kld = KldNormalLoss()

    def forward(self, recon_x, x, z_app_mean, z_app_var, z_vis_mean, *args, **kwargs):
        # recon_images, _ = recon_x

        recon = self.recon(recon_x, x)
        kld = self.kld(z_app_mean, z_app_var)

        loss = recon + self.beta * kld

        loss_dict = OrderedDict()
        loss_dict['recon'] = recon
        loss_dict['kld'] = kld

        return loss, loss_dict


class VaeConcreteLoss(nn.Module):
    def __init__(self, beta_v=0.0, py=0.5, categorical=False, mask_nn=None):
        super(VaeConcreteLoss, self).__init__()
        print('BetaVaeConcreteLoss')
        self.beta_v = beta_v

        if mask_nn:
            self.recon = WeightedMseLoss(mask_nn)
        else:
            self.recon = MseLoss()

        if categorical:
            self.kld_vis = KldCategoricalLoss(py=py)
        else:
            self.kld_vis = KldBinaryLoss(py=py)

    def forward(self, recon_x, x, z_app_mean, z_app_var, z_vis_mean, *args, **kwargs):
        recon = self.recon(recon_x, x)

        kld_vis = self.kld_vis(z_vis_mean)
        kld = self.beta_v * kld_vis

        loss = recon + kld
        loss_dict = OrderedDict()
        loss_dict['recon'] = recon
        loss_dict['kld'] = kld
        loss_dict['kld_vis'] = kld_vis

        return loss, loss_dict


class BetaVaeConcreteLoss(nn.Module):
    def __init__(self, beta_a=1.0, beta_v=0.0, py=0.5, categorical=False, mask_nn=None):
        super(BetaVaeConcreteLoss, self).__init__()
        print('BetaVaeConcreteLoss')
        self.beta_a = beta_a
        self.beta_v = beta_v

        if mask_nn:
            self.recon = WeightedMseLoss(mask_nn)
        else:
            self.recon = MseLoss()

        self.kld_app = KldNormalLoss()
        if categorical:
            self.kld_vis = KldCategoricalLoss(py=py)
        else:
            self.kld_vis = KldBinaryLoss(py=py)

    def forward(self, recon_x, x, z_app_mean, z_app_var, z_vis_mean, *args, **kwargs):
        recon = self.recon(recon_x, x)

        # z_app = td.normal.Normal(z_app_mean, z_app_std)
        kld_app = self.kld_app(z_app_mean, z_app_var)
        kld_vis = self.kld_vis(z_vis_mean)
        kld = self.beta_a * kld_app + self.beta_v * kld_vis

        loss = recon + kld
        loss_dict = OrderedDict()
        loss_dict['recon'] = recon
        loss_dict['kld'] = kld
        loss_dict['kld_app'] = kld_app
        loss_dict['kld_vis'] = kld_vis

        return loss, loss_dict


class BetaVaeConcretePartsLoss(nn.Module):
    def __init__(self, beta_a=1.0, beta_v=0.0, beta_p=0.0,
                 py=0.5, categorical=False):
        super(BetaVaeConcretePartsLoss, self).__init__()
        print('BetaVaeConcretePartsLoss')
        self.beta_a = beta_a
        self.beta_v = beta_v
        self.beta_p = beta_p

        self.recon_image = MseLoss()
        self.recon_parts = ReconPartsLoss()

        self.kld_app = KldNormalLoss()
        if categorical:
            self.kld_vis = KldCategoricalLoss(py=py)
        else:
            self.kld_vis = KldBinaryLoss(py=py)

    def forward(self, recon_x, x, q_z_app, q_z_vis, *args, **kwargs):
        recon_images, recon_parts = recon_x
        vis_mean, z_vis = q_z_vis

        recon_image = self.recon_image(recon_images, x)
        recon_parts = self.recon_parts(recon_parts, x, z_vis)

        kld_app = self.kld_app(q_z_app)
        kld_vis = self.kld_vis(vis_mean)

        loss = recon_image + self.beta_p * recon_parts + self.beta_a * kld_app + self.beta_v * kld_vis
        loss_dict = OrderedDict()
        loss_dict['recon_image'] = recon_image
        loss_dict['recon_parts'] = recon_parts
        loss_dict['kld_app'] = kld_app
        loss_dict['kld_vis'] = kld_vis

        return loss, loss_dict


class BetaVaeConcretePartsEntropyLoss(nn.Module):
    def __init__(self, beta_a=1.0, beta_v=0.0, beta_p=0.0,
                 beta_ea=0.0, beta_ew=0.0, py=0.5, categorical=False):
        super(BetaVaeConcretePartsEntropyLoss, self).__init__()
        print('BetaVaeConcretePartsEntropyLoss')
        self.beta_a = beta_a
        self.beta_v = beta_v
        self.beta_p = beta_p
        self.beta_ea = beta_ea
        self.beta_ew = beta_ew

        self.recon_image = MseLoss()
        self.recon_parts = ReconPartsLoss()

        self.kld_app = KldNormalLoss()
        if categorical:
            self.kld_vis = KldCategoricalLoss(py=py)
        else:
            self.kld_vis = KldBinaryLoss(py=py)

        self.ent_a = EntAcrossLoss()
        self.ent_w = EntWithinLoss()

    def forward(self, recon_x, x, q_z_app, q_z_vis, *args, **kwargs):
        recon_images, recon_parts = recon_x
        vis_mean, z_vis = q_z_vis

        recon_image = self.recon_image(recon_images, x)
        recon_parts = self.recon_parts(recon_parts, x, z_vis)
        recon = recon_image + self.beta_p * recon_parts

        kld_app = self.kld_app(q_z_app)
        kld_vis = self.kld_vis(vis_mean)
        kld = self.beta_a * kld_app + self.beta_v * kld_vis

        ent_a = self.ent_a(z_vis)
        ent_w = self.ent_w(z_vis)
        ent = self.beta_ea * ent_a + self.beta_ew * ent_w

        loss = recon + kld + ent

        loss_dict = OrderedDict()
        loss_dict['recon_image'] = recon_image
        loss_dict['recon_parts'] = recon_parts
        loss_dict['kld_app'] = kld_app
        loss_dict['kld_vis'] = kld_vis
        loss_dict['ent_a'] = ent_a
        loss_dict['ent_w'] = ent_w

        return loss, loss_dict


class DiscLoss(nn.Module):
    def __init__(self, beta=1.0):
        super(DiscLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss(reduction='sum')
        self.beta = beta

    def forward(self, prediction, target):
        loss = self.loss(prediction, target)
        return self.beta * loss / target.size(0)
