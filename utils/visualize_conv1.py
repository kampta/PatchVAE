import sys
import torch
import argparse
import matplotlib.pyplot as plt
import collections

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualze conv1 layer')

    # Arguments
    parser.add_argument('--model', required=True,
                        help='model path')
    parser.add_argument('--channels', default=1, type=int,
                        help='model path')
    parser.add_argument('--save', default=None)
    args = parser.parse_args()

    # Model Weights
    map_location = 'cpu'
    if torch.cuda.is_available():
        map_location = 'cuda'
    load = torch.load(args.model, map_location=map_location)
    if type(load) == collections.OrderedDict:
        state_dict = load
    elif 'state_dict' in load:
        state_dict = load['state_dict']
    elif 'vae_dict' in load:
        state_dict = load['vae_dict']
    else:
        print('state_dict keys: %s' % load.keys())
        print('unable to handle')
        sys.exit(0)

    # Get first conv filter
    for k in state_dict:
        if 'weight' in k:
            print('getting %s weight' % k)
            break
    conv1 = state_dict[k].permute(0, 2, 3, 1).data.numpy()
    conv1 = (conv1 - conv1.min()) / (conv1.max() - conv1.min())
    N, H, W, C = conv1.shape

    fig = plt.figure(figsize=(8, 8))
    fig.suptitle('conv1 weights'.format())
    cmap = 'gray' if C == 1 else None

    for i, filters in enumerate(conv1):
        ax = plt.subplot(8, N // 8, i + 1)
        # ax.set_title('{}'.format(label))
        ax.axis('off')
        ax.imshow(filters.squeeze(), cmap=cmap)

    if args.save:
        print('Saving at %s' % args.save)
        plt.savefig(args.save)
    else:
        plt.show()

    if args.channels > 1:
        for c in range(C):
            fig = plt.figure()
            fig.suptitle('conv1, channel: {}'.format(c))

            for i, filters in enumerate(conv1):
                ax = plt.subplot(8, N // 8, i + 1)
                # ax.set_title('{}'.format(label))
                ax.axis('off')
                ax.imshow(filters[:, :, c].squeeze(), cmap='hot')
            if args.save:
                print('Saving at %s' % args.save.replace('.png', '_%d.png' % c))
                plt.savefig(args.save.replace('.png', '_%d.png' % c))
            else:
                plt.show()
