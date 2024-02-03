import time
import argparse
from examples.mnist import mnist
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.nn.layers.activations import tanh
from mlx.utils import tree_flatten
import numpy as np


def batch_iterate(batch_size, X, y):
    perm = mx.array(np.random.permutation(y.size))
    for s in range(0, y.size, batch_size):
        ids = perm[s: s + batch_size]
        yield X[ids].reshape(-1, 28, 28, 1), y[ids]


class LeNet(nn.Module):
    '''LeNet inspired module'''

    def __init__(self, in_dims=1, num_classes=10):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=in_dims, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=0, stride=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=256, kernel_size=5, stride=1, padding=0)
        self.dense0 = nn.Linear(input_dims=256, output_dims=120)
        self.dense1 = nn.Linear(input_dims=120, output_dims=num_classes)

    def num_params(self):
        nparams = sum(x.size for k, x in tree_flatten(self.parameters()))
        return nparams

    def avg_pool_2d(self, x, stride: int = 2):
        '''https://github.com/ml-explore/mlx/issues/25'''
        B, W, H, C = x.shape
        x = x.reshape(B, W // stride, stride, H // stride, stride, C).mean((2, 4))
        return x

    def __call__(self, x):
        x = tanh(self.conv0(x))
        x = tanh(self.avg_pool_2d(x))
        x = tanh(self.conv1(x))
        x = tanh(self.avg_pool_2d(x))
        x = tanh(self.conv2(x))
        x = x.reshape(x.shape[0], -1)
        x = tanh(self.dense0(x))
        x = self.dense1(x)
        return x


def loss_fn(model, X, y):
    return mx.mean(nn.losses.cross_entropy(model(X), y))


def eval_fn(model, X, y):
    return mx.mean(mx.argmax(model(X), axis=1) == y)


def main():
    model = LeNet()
    print('Number of params: {:0.04f} M'.format(model.num_params() / 1e6))
    train_images, train_labels, test_images, test_labels = map(mx.array, mnist.mnist())
    optimizer = optim.SGD(learning_rate=0.1)
    loss_and_grad = nn.value_and_grad(model=model, fn=loss_fn)
    losses = []
    for e in range(20):
        tic = time.perf_counter()
        for X, y in batch_iterate(batch_size=256, X=train_images, y=train_labels):
            loss, gradients = loss_and_grad(model, X, y)
            optimizer.update(model, gradients)
            mx.eval(model.parameters(), optimizer.state)
            losses.append(loss.item())
        toc = time.perf_counter()
        accuracy = eval_fn(model, test_images.reshape(-1, 28, 28, 1), test_labels)
        throughput = 60000 / (toc - tic)
        print(
            f"Epoch {e}: Test accuracy {accuracy.item():.6f},"
            f" Time {toc - tic:.3f} (s), ",
            f"Throughput: {throughput:.2f} images/second"
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', help='disable using the GPU', action='store_true')
    args = parser.parse_args()
    if args.cpu:
        mx.set_default_device(mx.cpu)
    main()
