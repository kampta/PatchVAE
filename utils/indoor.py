import os
import argparse


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--image-dir', required=True,
                    help='image directory')
parser.add_argument('--split-file', required=True,
                    help='file containing the path of images to be copies')
parser.add_argument('--target-dir', required=True,
                    help='target directory')
parser.add_argument('--dry', action='store_true', default=False,
                    help='dry run')

args = parser.parse_args()


# Create target dir
if not os.path.exists(args.target_dir):
    os.makedirs(args.target_dir)

# Create all label dirs
for d in os.listdir(args.image_dir):
    src_dir = os.path.join(args.image_dir, d)
    if os.path.isdir(src_dir):
        dst_dir = os.path.join(args.target_dir, d)
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)

# Move files
with open(args.split_file) as f:
    for line in f:
        file_path = line.strip()
        src_path = os.path.join(args.image_dir, file_path)
        dst_path = os.path.join(args.target_dir, file_path)

        print("%s => %s" % (src_path, dst_path))
        if not args.dry and os.path.exists(src_path) and not os.path.exists(dst_path):
            os.rename(src_path, dst_path)