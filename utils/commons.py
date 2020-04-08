import sys
import os
import numpy as np
import argparse
import logging
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import collections

import torch
import torch.nn as nn
from torch.nn.functional import interpolate

from torchvision import transforms, datasets, models, utils as tvutils

# Just placeholders, values updated later
inet_normalize = transforms.Normalize(
    mean=[0., 0., 0.],
    std=[1., 1., 1.]
)

inet_unnormalize = transforms.Normalize(
    mean=[0., 0., 0.],
    std=[1., 1., 1.]
)

# Again just placeholder
unnorm = None


class UnNormalize(nn.Module):
    def __init__(self, mean, std):
        super(UnNormalize, self).__init__()

        mean, std = torch.tensor(mean), torch.tensor(std)
        if torch.cuda.is_available():
            mean, std = mean.to('cuda'), std.to('cuda')

        self.mean = mean[None, :, None, None]
        self.std = std[None, :, None, None]

    def forward(self, x):
        return (x - self.mean) / self.std


def data_loaders(dataset, data_folder='./data', inet=True,
                 distributed=False, classify=False, size=64, **kwargs):

    global inet_normalize, inet_unnormalize, unnorm

    print("Do imagenet normalization: ", inet)
    if inet:
        inet_normalize.mean = [0.485, 0.456, 0.406]
        inet_normalize.std = [0.229, 0.224, 0.225]
        inet_unnormalize.mean = [-0.485/0.229, -0.456/0.224, -0.406/0.225]
        inet_unnormalize.std = [1./0.229, 1./0.224, 1./0.225]
    else:
        inet_normalize.mean = [0.5, 0.5, 0.5]
        inet_normalize.std = [0.5, 0.5, 0.5]
        inet_unnormalize.mean = [-1., -1., -1.]
        inet_unnormalize.std = [2., 2., 2.]

    # normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    train_transform = transforms.Compose([
        # transforms.RandomResizedCrop(224),
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        inet_normalize,
    ])
    test_transform = transforms.Compose([
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        inet_normalize,
    ])

    if dataset == 'mnist':
        inet_normalize = transforms.Normalize((0.5,), (0.5,))
        inet_unnormalize = transforms.Normalize((-1.,), (2.,))

        train_dataset = datasets.MNIST(data_folder,
                                       train=True, download=True,
                                       transform=transforms.Compose([
                                           transforms.Resize(32),
                                           transforms.ToTensor(),
                                           inet_normalize
                                       ]))
        val_dataset = datasets.MNIST(data_folder, train=False,
                                     transform=transforms.Compose([
                                         transforms.Resize(32),
                                         transforms.ToTensor(),
                                         inet_normalize
                                     ]))
        channels = 1
        height = 32
        width = 32
        classes = 10

    elif dataset == 'indoor':
        if classify:
            train_dataset = datasets.ImageFolder(os.path.join(data_folder, 'train'),
                                                 transform=train_transform)
        else:
            train_dataset = datasets.ImageFolder(os.path.join(data_folder, 'allminustest'),
                                                 transform=train_transform)
        val_dataset = datasets.ImageFolder(os.path.join(data_folder, 'test'),
                                           transform=test_transform)

        channels = 3
        height = size
        width = size
        classes = 67

    elif dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(data_folder, train=True, download=True,
                                          transform=transforms.Compose([
                                              transforms.ToTensor(),
                                              inet_normalize
                                          ]))
        val_dataset = datasets.CIFAR100(data_folder, train=False,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            inet_normalize
                                        ]))

        channels = 3
        height = 32
        width = 32
        classes = 100

    elif dataset == 'places205':
        from utils.places import Places205
        train_dataset = Places205(data_folder, 'train', transform=train_transform)
        val_dataset = Places205(data_folder, 'val', transform=test_transform)

        channels = 3
        height = size
        width = size
        classes = 205

    elif dataset == 'imagenet':
        train_dataset = datasets.ImageFolder(os.path.join(data_folder, 'train'),
                                             transform=train_transform)
        val_dataset = datasets.ImageFolder(os.path.join(data_folder, 'val'),
                                           transform=test_transform)

        channels = 3
        height = size
        width = size
        classes = 1000

    elif dataset == 'miniimagenet':

        train_dataset = datasets.ImageFolder(os.path.join(data_folder, 'train'),
                                             transform=train_transform)

        val_dataset = None
        channels = 3
        height = size
        width = size
        classes = len(train_dataset.classes)

    elif dataset == 'lsun':
        train_dataset = datasets.LSUN(data_folder, classes=['bedroom_train'],
                                      transform=train_transform)
        val_dataset = datasets.LSUN(data_folder, classes=['bedroom_val'],
                                    transform=test_transform),
        channels = 3
        height = size
        width = height
        classes = 1

    elif dataset == 'shapes':
        from utils.shapes import ShapesDataset

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        train_dataset = ShapesDataset(height=32, width=32,
                                      channels=3,
                                      min_objects_per_img=1,
                                      max_objects_per_img=4,
                                      count=60000,
                                      seed=0,
                                      transform=transform)

        val_dataset = ShapesDataset(height=28, width=28,
                                    channels=3,
                                    min_objects_per_img=1,
                                    max_objects_per_img=4,
                                    count=10000,
                                    seed=60000,
                                    transform=transform)
        channels = 3
        height = 32
        width = 32
        classes = 3

    elif dataset == 'binshapes':
        from utils.shapes import ShapesDataset

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        train_dataset = ShapesDataset(height=32, width=32,
                                      channels=1,
                                      binary=True,
                                      min_objects_per_img=1,
                                      max_objects_per_img=2,
                                      count=60000,
                                      seed=0,
                                      # classes=['square'],
                                      transform=transform)

        val_dataset = ShapesDataset(height=28, width=28,
                                    channels=1,
                                    binary=True,
                                    min_objects_per_img=1,
                                    max_objects_per_img=2,
                                    count=10000,
                                    seed=60000,
                                    # classes=['square'],
                                    transform=transform)

        channels = 1
        height = 32
        width = 32
        classes = 3

    elif dataset == 'bincircles':
        from utils.shapes import ShapesDataset
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        train_dataset = ShapesDataset(height=32, width=32,
                                      channels=1,
                                      binary=True,
                                      min_objects_per_img=1,
                                      max_objects_per_img=2,
                                      count=60000,
                                      seed=0,
                                      classes=['circle'],
                                      transform=transform)

        val_dataset = ShapesDataset(height=28, width=28,
                                    channels=1,
                                    binary=True,
                                    min_objects_per_img=1,
                                    max_objects_per_img=2,
                                    count=10000,
                                    seed=60000,
                                    classes=['circle'],
                                    transform=transform)
        channels = 1
        height = 32
        width = 32
        classes = 2

    else:
        print('Unknown dataset: %s' % dataset)
        sys.exit(0)

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    if val_dataset is None:
        dataset_size = len(train_dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.2 * dataset_size))
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        # Creating PT data samplers and loaders:
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

        train_loader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler,
                                                   pin_memory=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(train_dataset, sampler=valid_sampler,
                                                 pin_memory=True, **kwargs)
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, shuffle=(train_sampler is None),
            pin_memory=True, sampler=train_sampler, **kwargs)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, shuffle=False, pin_memory=True, **kwargs)

    unnorm = UnNormalize(inet_unnormalize.mean, inet_unnormalize.std)

    return train_loader, val_loader, (channels, height, width), classes, train_sampler


def load_vae_model(input_size, arch='patchy', device='cuda', **kwargs):
    if arch == 'patchy':
        from model import PatchyVAE
        model = PatchyVAE(input_size, **kwargs).to(device)
        bottleneck = model.bottleneck

    elif arch == 'vae':
        from model import VAE
        model = VAE(input_size, **kwargs).to(device)
        bottleneck = model.bottleneck

    else:
        print('Unknown model architecture: %s' % arch)
        sys.exit(0)

    return model, bottleneck


# count number of model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Compute receptive field of a pixel/location in a single layer
def rf_single_layer(out_size, kernel, stride):
    return stride * (out_size - 1) + kernel if stride > 0.5 else ((out_size + (kernel - 2)) / 2) + 1


# Compute receptive field for a single pixel in output layer
def rf(conv_net):
    out_size = 1
    layers = list(conv_net.modules())
    for layer in layers[::-1]:
        if type(layer) == nn.Conv2d:
            out_size = rf_single_layer(out_size, layer.kernel_size[0], layer.stride[0])
    return out_size


# Computes the accuracy over the k top predictions for the specified values of k
def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def load_cls_model(arch, num_classes=10, pretrained=None, freeze=0, scale=8,
                   input_size=(3, 224, 224), **kwargs):

    if arch == 'alexnet':
        model = models.__dict__[arch]()
        bottleneck = 256 * 1 * 1
        # bottleneck = 256 * 6 * 6

    elif arch == 'resnet18':
        # model = models.__dict__[arch]()
        model = nn.Sequential()
        model.add_module('features', make_encoder(input_size[0], -1, arch='resnet18', scale=scale))
        model.features.add_module('avgpool', nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        bottleneck = 512

    elif arch == 'patchy' or arch == 'vae':
        model, bottleneck = load_vae_model(input_size, arch, scale=scale, **kwargs)

    else:
        print("No arch named: %s" % arch)
        sys.exit(0)

    if pretrained is not None:
        print("Loading pretrained model from %s" % pretrained)
        # https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/3
        map_location = 'cpu'
        if torch.cuda.is_available():
            map_location = 'cuda'

        load = torch.load(pretrained, map_location=map_location)
        if type(load) == collections.OrderedDict:
            pretrained_dict = load
        else:
            pretrained_dict = load['vae_dict']

        # pretrained_dict = torch.load(pretrained)
        model_dict = model.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict_filtered = collections.OrderedDict()
        for k in pretrained_dict:
            if k in model_dict:
                pretrained_dict_filtered[k] = pretrained_dict[k]
            elif k.replace('module.', '') in model_dict:
                pretrained_dict_filtered[k.replace('module.', '')] = pretrained_dict[k]

        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict_filtered)
        # 3. load the new state dict
        model.load_state_dict(pretrained_dict_filtered)
    else:
        model.features.apply(init_weights)

    # if arch == 'resnet18':
    #     last = 7 - int(np.log2(scale))
    #     features = nn.Sequential(*list(model.children())[:-last])
    #     model.features = features

    # Add a simple classification layer on top of the encoder
    cls_model = SimpleClassifier(model.features, bottleneck, num_classes=num_classes)
    cls_model.classifier.apply(init_weights)

    if freeze == -1:
        print("Freezing feature layers")
        # https://discuss.pytorch.org/t/how-the-pytorch-freeze-network-in-some-layers-only-the-rest-of-the-training/7088/3
        for param in cls_model.features.parameters():
            param.requires_grad = False
    elif freeze > 0:
        for idx, child in enumerate(cls_model.features.children(), start=1):
            for param in child.parameters():
                param.requires_grad = False
            if idx >= freeze:
                break

    return cls_model


class SimpleClassifier(nn.Module):
    """
    Add a fc layer to conv layers from another model
    self.features - conv layers
    self.classifier - fc layers
    """
    def __init__(self, features, bottleneck, num_classes=10):
        super(SimpleClassifier, self).__init__()
        self.bottleneck = bottleneck
        self.features = features
        self.features.apply(set_bn_fix)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            # nn.Linear(bottleneck, 4096),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            # nn.Linear(4096, 4096),
            # nn.ReLU(inplace=True),
            # nn.Linear(4096, num_classes),
            nn.Linear(bottleneck, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# Initialization of weights
def init_weights(layer):
    if isinstance(layer, nn.Conv2d):
        nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)
    elif isinstance(layer, nn.BatchNorm2d):
        nn.init.constant_(layer.weight, 1)
        nn.init.constant_(layer.bias, 0)
    elif isinstance(layer, nn.Linear):
        nn.init.normal_(layer.weight, 0, 0.01)
        nn.init.constant_(layer.bias, 0)


def set_bn_fix(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        for p in m.parameters():
            p.requires_grad = False


# For weighted mask loss
class EdgeWeights(nn.Module):
    def __init__(self, nc=3, scale=8):
        super(EdgeWeights, self).__init__()
        self.scale = scale
        edge_weights = torch.Tensor([[0., 1., 0.],
                                     [1., -4., 1.],
                                     [0., 1., 0.]])
        self.pad = nn.ReplicationPad2d(1)
        self.edge_kernel = nn.Conv2d(nc, nc, 3, 1, 0, groups=nc, bias=False)
        self.edge_kernel.weight.data = edge_weights.reshape(1, 1, 3, 3).repeat(nc, 1, 1, 1)
        avg_weights = torch.ones(scale, scale)
        self.avg_kernel = nn.Conv2d(nc, nc, scale, scale, 0, groups=nc, bias=False)
        self.avg_kernel.weight.data = avg_weights.reshape(1, 1, scale, scale).repeat(nc, 1, 1, 1)

        if torch.cuda.is_available():
            self.pad, self.edge_kernel, self.avg_kernel = \
                self.pad.to('cuda'), self.edge_kernel.to('cuda'), self.avg_kernel.to('cuda')

    def forward(self, x):
        n, c, h, w = x.size()

        # Get edges
        edges = torch.abs(self.edge_kernel(self.pad(x)))
        masks = self.avg_kernel(edges)

        # Get grid masks
        masks = interpolate(masks, scale_factor=self.scale, mode='nearest')

        # Normalize
        channel_wise_sum = torch.sum(masks, (2, 3))
        channel_wise_sum = channel_wise_sum[:, :, None, None]
        masks = masks / channel_wise_sum.expand(-1, -1, h, w) * h * w

        return masks.detach()


class Edges(nn.Module):
    def __init__(self, nc=3, scale=8):
        super(Edges, self).__init__()
        self.scale = scale
        edge_weights = torch.Tensor([[0., 1., 0.],
                                     [1., -4., 1.],
                                     [0., 1., 0.]])
        self.pad = nn.ReplicationPad2d(1)
        self.edge_kernel = nn.Conv2d(nc, nc, 3, 1, 0, groups=nc, bias=False)
        self.edge_kernel.weight.data = edge_weights.reshape(1, 1, 3, 3).repeat(nc, 1, 1, 1)

        if torch.cuda.is_available():
            self.pad, self.edge_kernel = self.pad.to('cuda'), self.edge_kernel.to('cuda')

    def forward(self, x):

        # Get edges
        edges = torch.abs(self.edge_kernel(self.pad(x)))
        return edges.detach()


def get_collate_fn(collate_type='edge', scale=8):
    def edge_collate_fn(batch):
        return

    if collate_type == 'edge':
        return edge_collate_fn
    else:
        print('Unknown collate type: %s' % collate_type)
        sys.exit(0)


def get_mask_fn(collate_type='edge', scale=8, nc=3):
    if collate_type == 'edge':
        return EdgeWeights(nc=nc, scale=scale)
    if collate_type == 'pixel':
        return Edges(nc=nc, scale=scale)
    else:
        print('Unknown collate type: %s' % collate_type)
        sys.exit(0)


def make_encoder(input_channels, ngf, arch='resnet', scale=8):
    if arch == 'resnet':
        resnet18 = models.resnet18(pretrained=False)

        # remove the pooling layer and last k residual blocks depending on scale
        # scale can be <4, 8, 16, 32>
        # log_scale will be  <2, 3, 4, 5>
        # last will be <5, 4, 3, 2>
        last = 7 - int(np.log2(scale))

        features = nn.Sequential(*list(resnet18.children())[:-last])

    elif arch == 'alexnet':
        features = models.alexnet(pretrained=False).features

    elif arch == 'resnet18':
        resnet18 = models.resnet18(pretrained=False)

        # remove the pooling layer
        features = nn.Sequential(*list(resnet18.children())[:-2])

    elif arch == 'resnet50':
        resnet50 = models.resnet50(pretrained=False)

        # remove the pooling layer
        features = nn.Sequential(*list(resnet50.children())[:-2])

    elif arch == 'pyramid':
        # input C x H x W
        # output (base_depth * scale / 2) x (H / scale) x (W / scale)

        log_scale = np.log2(scale)

        assert log_scale == round(log_scale), 'scale must be a power of 2'
        assert log_scale > 0, 'must at least half the feature map size'

        log_scale = int(log_scale)

        features = nn.Sequential()
        # input is (nc) x 64 x 64
        features.add_module('input-conv', nn.Conv2d(input_channels, ngf,
                                                    4, 2, 1, bias=False))
        features.add_module('input-relu', nn.LeakyReLU(0.2, inplace=True))

        for i in range(log_scale-1):
            # state size. (ngf) x 32 x 32
            features.add_module('pyramid-{0}-{1}-conv'.format(ngf * 2 ** i, ngf * 2 ** (i + 1)),
                                nn.Conv2d(ngf * 2 ** i, ngf * 2 ** (i + 1), 4, 2, 1, bias=False))
            features.add_module('pyramid-{0}-batchnorm'.format(ngf * 2 ** (i + 1)),
                                nn.BatchNorm2d(ngf * 2 ** (i + 1)))
            features.add_module('pyramid-{0}-relu'.format(ngf * 2 ** (i + 1)),
                                nn.LeakyReLU(0.2, inplace=True))

    else:
        print('Invalid encoder architecture: %s' % arch)
        sys.exit(0)

    return features


def make_decoder(output_channels, ngf, arch='pyramid', groups=1,
                 nz=8, scale=8):

    if arch == 'pyramid':
        log_scale = np.log2(scale)

        assert log_scale == round(log_scale), 'scale must be a power of 2'
        log_scale = int(log_scale)

        decoder = nn.Sequential()
        # input is Z, going into a convolution
        decoder.add_module('input-conv', nn.ConvTranspose2d(nz, ngf * 2 ** (log_scale-1), 1, bias=False, groups=groups))
        decoder.add_module('input-batchnorm', nn.BatchNorm2d(ngf * 2 ** (log_scale-1)))
        decoder.add_module('input-relu', nn.LeakyReLU(0.2, inplace=True))

        for i in range(log_scale-1, 0, -1):
            in_c = int(ngf * 2 ** i)
            out_c = int(ngf * 2 ** (i - 1))
            decoder.add_module(
                'pyramid-{0}-{1}-conv'.format(in_c, out_c),
                nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False, groups=groups))
            decoder.add_module(
                'pyramid-{0}-batchnorm'.format(out_c), nn.BatchNorm2d(out_c))
            decoder.add_module(
                'pyramid-{0}-relu'.format(out_c), nn.LeakyReLU(0.2, inplace=True))

        decoder.add_module('ouput-conv', nn.ConvTranspose2d(ngf, output_channels, 4, 2, 1, bias=False))
        decoder.add_module('output-tanh', nn.Tanh())

    else:
        print('Invalid decoder architecture: %s' % arch)
        sys.exit(0)

    return decoder


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Statistics(object):
    def __init__(self, names):
        self.names = names
        self.meters = {}
        for name in names:
            self.meters.update({name: AverageMeter()})

    def update(self, n, **kwargs):
        info = ''
        for key in kwargs:
            self.meters[key].update(kwargs[key], n)
            info += '{key}={loss.val:.4f}, avg {key}={loss.avg:.4f}, '.format(key=key, loss=self.meters[key])
        return info[:-2]

    def summary(self):
        info = ''
        for name in self.names:
            info += 'avg {key}={loss:.4f}, '.format(key=name, loss=self.meters[name].avg)
        return info[:-2]


class Logger(object):
    def __init__(self, path):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')

        fh = logging.FileHandler(os.path.join(path, 'debug.log'))
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

    def log(self, info):
        self.logger.info(info)


if __name__ == "__main__":

    import matplotlib
    font = {'family': 'normal',
            # 'weight': 'bold',
            'size': 8}

    matplotlib.rc('font', **font)

    parser = argparse.ArgumentParser(description='Shapes dataset')

    # Arguments
    parser.add_argument('--dataset', type=str, default='lsun',
                        help='dataset (default: lsun)')
    parser.add_argument('--data-folder', type=str, default='./data',
                        help='name of the data folder (default: ./data)')
    parser.add_argument('--segment', type=str, default=None)
    parser.add_argument('--scale', type=int, default=8)
    parser.add_argument('--save', default=None)

    parser.add_argument('--batch-size', type=int, default=8,
                        help='batch size (default: 8)')
    parser.add_argument('--seed', type=int, default=0,
                        help='seed (default: 0)')

    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    num_fig = 1
    loader_kwargs = {
        'data_folder': args.data_folder,
        'batch_size': args.batch_size,
        'num_workers': 2,
    }

    train_loader, test_loader, (channels, height, width), classes, _ = \
        data_loaders(args.dataset, inet=False, **loader_kwargs)

    if args.segment:
        # loader_kwargs['collate_fn'] = get_collate_fn(args.segment, args.scale)
        mask_nn = get_mask_fn(args.segment, args.scale, channels)

        edge_nn = get_mask_fn('pixel', args.scale, channels)
        num_fig = 4

    fig = plt.figure(figsize=(8, 6.5))

    for batch_idx, (x, y) in enumerate(train_loader):
        grid_img = tvutils.make_grid(unnorm(x), pad_value=1, nrow=8)
        tvutils.save_image(unnorm(x), 'original.png', pad_value=1, nrow=8)
        fig.add_subplot(num_fig, 1, 1)
        plt.imshow(grid_img.permute(1, 2, 0))
        plt.title('Random images from %s dataset' % args.dataset)
        plt.axis('off')

        if args.segment:
            edge = edge_nn(x)
            grid_img = tvutils.make_grid(edge, nrow=8, pad_value=1, normalize=True)
            tvutils.save_image(edge, 'edges.png', pad_value=1, nrow=8, normalize=True)
            fig.add_subplot(num_fig, 1, 2)
            plt.imshow(grid_img.permute(1, 2, 0))
            plt.title('Image laplacians')
            plt.axis('off')

            mask = mask_nn(x)
            grid_img = tvutils.make_grid(mask, nrow=8, pad_value=1, normalize=True)
            tvutils.save_image(mask, 'masks.png', pad_value=1, nrow=8, normalize=True)
            fig.add_subplot(num_fig, 1, 3)
            plt.imshow(grid_img.permute(1, 2, 0))
            plt.title('Weight masks for computing reconstruction loss')
            plt.axis('off')

            img = mask * unnorm(x)
            grid_img = tvutils.make_grid(img, nrow=8, pad_value=1, normalize=True)
            tvutils.save_image(img, 'product.png', pad_value=1, nrow=8, normalize=True)
            fig.add_subplot(num_fig, 1, 4)
            plt.imshow(grid_img.permute(1, 2, 0))
            plt.title('Product of image with weight mask')
            plt.axis('off')

        if args.save:
            print('Saving at %s' % args.save)
            plt.savefig(args.save)
        else:
            plt.show()
        break
