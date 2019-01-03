import argparse

parser = argparse.ArgumentParser(description='training of the deep info model')

parser.add_argument('--z_dim', type=int,  default=256, help='hidden vector size')
parser.add_argument('--batch_size', type=int,  default=64, help='batch size')
parser.add_argument('--alpha', type=float,  default=0.5, help='global mutual information')
parser.add_argument('--beta', type=float,  default=1.5, help='local mutual information')
parser.add_argument('--gamma', type=float,  default=0.01, help='prior distribution loss')
parser.add_argument('--epochs', type=int,  default=20, help='prior distribution loss')
args = parser.parse_args()
