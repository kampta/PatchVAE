
import math
import numpy as np
import cv2
from skimage import transform
import torch
from torchvision import transforms
from torch.utils.data import Dataset


class ShapesDataset(Dataset):
    """Synthetic shapes dataset
    Generates images with 1-3 of the following shapes
    ['circle', 'triangle', 'square']
    """

    classes = ['circle', 'triangle', 'square']

    def __init__(self, height=32, width=32, channels=3,
                 min_objects_per_img=1,
                 max_objects_per_img=4,
                 count=100000,
                 seed=0,
                 binary=False,
                 classes=None,
                 transform=None):
        self.height = height
        self.width = width
        self.channels = channels
        self.binary = binary
        self.min_objects_per_img = min_objects_per_img
        self.max_objects_per_img = max_objects_per_img
        self.count = count
        self.transform = transform
        self.seed = seed
        if classes is not None:
            self.classes = classes

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        np.random.seed(idx + self.seed)

        # generate number of shapes (between 1 and 3) to put inside the image
        count = np.random.randint(self.min_objects_per_img,
                                  self.max_objects_per_img)

        # generate background and foreground color for the image
        if self.binary:
            bg_color = 0
            fg_color = 255

        else:
            bg_color = self.random_color()
            fg_color = self.random_color()

        # generate random specifications of each shape (size and location)
        specs = [self.random_shape() for _ in range(count)]

        # generate an image with above specifications
        img = self.create_image(specs, bg_color=bg_color, fg_color=fg_color)

        if self.transform:
            img = self.transform(img)

        one_hot = np.zeros(len(self.classes))
        for spec in specs:
            one_hot[self.classes.index(spec[0])] = 1

        return img, one_hot

    def random_color(self):

        return [np.random.randint(0, 255) for _ in range(self.channels)]

    def random_shape(self):
        # Shape
        shape = np.random.choice(self.classes)

        # Center x, y
        buffer = 2
        y = np.random.randint(buffer, self.height - buffer - 1)
        x = np.random.randint(buffer, self.width - buffer - 1)

        # Size
        s = np.random.randint(self.height // 4, self.height // 2)
        return shape, (x, y, s)

    def create_image(self, specs, bg_color=None, fg_color=None):
        if bg_color is None:
            bg_color = self.random_color()
        if fg_color is None:
            fg_color = self.random_color()

        # create image with background
        img = np.ones([self.height, self.width, self.channels], dtype=np.uint8)
        img = img * np.uint8(bg_color)

        # add objects to it
        for shape, (x, y, s) in specs:
            self.draw_shape(img, shape, fg_color, (x, y), s)

        return img

    @staticmethod
    def draw_shape(img, shape, color, location, size):
        x, y = location

        # Circle
        if shape == 'circle':
            cv2.circle(img, (x, y), size, color, -1)

        # Triangle
        elif shape == 'triangle':
            points = np.array([[(x, y-size),
                                (x-size/math.sin(math.radians(60)), y+size),
                                (x+size/math.sin(math.radians(60)), y+size),
                                ]], dtype=np.int32)
            cv2.fillPoly(img, points, color)

        # Square
        elif shape == 'square':
            cv2.rectangle(img, (x - size, y - size), (x + size, y + size), color, -1)

        return img


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        # image, landmarks = sample['image'], sample['landmarks']

        h, w = sample.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(sample, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        # landmarks = landmarks * [new_w / w, new_h / h]

        # return {'image': img, 'landmarks': landmarks}
        return img


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        # image, landmarks = sample['image'], sample['landmarks']
        # image = sample[0]

        h, w = sample.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = sample[top: top + new_h,
                       left: left + new_w]

        # landmarks = landmarks - [left, top]

        # return {'image': image, 'landmarks': landmarks}
        return image


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # image, landmarks = sample['image'], sample['landmarks']
        # image = sample[0]
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = sample.transpose((2, 0, 1))
        # return {'image': torch.from_numpy(image),
        #        'landmarks': torch.from_numpy(landmarks)}
        return torch.from_numpy(image)


if __name__ == "__main__":

    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description='Shapes dataset')

    # Arguments
    parser.add_argument('--height', type=int, default=32,
                        help='image height (default: 32)')
    parser.add_argument('--width', type=int, default=32,
                        help='image width (default: 32)')
    parser.add_argument('--channels', type=int, default=3,
                        help='image channels (default: 3)')
    parser.add_argument('--min', type=int, default=1,
                        help='min shapes per image (default: 1)')
    parser.add_argument('--max', type=int, default=4,
                        help='max shapes per image (default: 4)')
    parser.add_argument('--binary', action='store_true', default=False,
                        help='binary images (default: False)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='batch size (default: 8)')
    parser.add_argument('--seed', type=int, default=0,
                        help='seed (default: 0)')

    args = parser.parse_args()
    shapes_dataset = ShapesDataset(height=args.height, width=args.width,
                                   channels=args.channels,
                                   binary=args.binary,
                                   min_objects_per_img=args.min,
                                   max_objects_per_img=args.max,
                                   seed=args.seed,
                                   transform=transforms.Compose([
                                       # Rescale(256),
                                       # RandomCrop(224),
                                       transforms.ToTensor()
                                   ])
                                   )

    # kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    data_loader = torch.utils.data.DataLoader(shapes_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              # **kwargs
                                              )

    for batch_idx, (x, y) in enumerate(data_loader):
        fig = plt.figure()
        fig.suptitle('{}'.format(shapes_dataset.classes))
        cmap = 'gray' if args.channels == 1 else None

        samples = x.permute(0, 2, 3, 1).numpy()
        y = y.numpy()
        for i, (sample, one_hot) in enumerate(zip(samples, y)):
            print(i, sample.shape)
            ax = plt.subplot(2, args.batch_size // 2, i + 1)
            ax.set_title('{}'.format(one_hot))
            ax.axis('off')
            ax.imshow(sample.squeeze(), cmap=cmap)

        plt.show()

        break
