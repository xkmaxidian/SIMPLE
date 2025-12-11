
import argparse

from model import My_Model
from shuffle_data import preprocess_data
import numpy as np
import random

# 参数配置:original_dim, epoch, n_centroid, lr_nn, lr_gmm, decay_n, decay_nn, decay_gmm, alpha, beta, datatype
def config_init(original_dim, dataset, epoch, alpha=0.1, beta=0.1):
    # if dataset == 'UCI6':
    #     return [240, 76, 216, 47, 64, 6], epoch, 10, 0.0001, 0.002, 10, 0.9, 0.9, alpha, beta, 'sigmoid'
    # if dataset == 'Caltech101-7':
    #     return [1984, 48, 40, 254, 512, 928], epoch, 7, 0.0001, 0.005, 10, 0.9, 0.9, alpha, beta, 'linear'
    # if dataset == 'Scene15':
    #     return [59, 40, 20], epoch, 15, 0.0001, 0.005, 10, 0.9, 0.9, alpha, beta, 'linear'
    # if dataset == 'Office31':
    #     return [2048, 2048, 2048], epoch, 31, 0.0001, 0.005, 10, 0.9, 0.9, alpha, beta,'linear'
    # if dataset == 'cub_googlenet_doc2vec_c10':
    #     return [1024, 300], epoch, 10, 0.001, 0.002, 10, 0.9, 0.9, alpha, beta,'linear'
    # if dataset == 'LandUse-21':
    #     return [59, 40, 20], epoch, 21, 0.0001, 0.005, 10, 0.9, 0.9, alpha, beta, 'linear'
    # if dataset =='Hdigit':
    #     return [784, 256], epoch, 10, 0.0001, 0.005, 10, 0.9, 0.9, alpha, beta, 'sigmoid'
    if dataset == 'UCI6':
        return original_dim, epoch, 10, 0.0001, 0.002, 10, 0.9, 0.9, alpha, beta, 'sigmoid'
    if dataset == 'Caltech101-7':
        return original_dim, epoch, 7, 0.0001, 0.005, 10, 0.9, 0.9, alpha, beta, 'linear'
    if dataset == 'Scene15':
        return original_dim, epoch, 15, 0.0001, 0.005, 10, 0.9, 0.9, alpha, beta, 'linear'
    if dataset == 'Office31':
        return original_dim, epoch, 31, 0.0001, 0.005, 10, 0.9, 0.9, alpha, beta,'linear'
    if dataset == 'cub_googlenet_doc2vec_c10':
        return original_dim, epoch, 10, 0.001, 0.002, 10, 0.9, 0.9, alpha, beta,'linear'
    if dataset == 'LandUse-21':
        return original_dim, epoch, 21, 0.0001, 0.005, 10, 0.9, 0.9, alpha, beta, 'linear'
    if dataset =='Hdigit':
        return original_dim, epoch, 10, 0.0001, 0.005, 10, 0.9, 0.9, alpha, beta, 'sigmoid'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', dest='dataset', default='UCI6',
                    choices=['UCI6', 'Caltech101-7', 'Scene15',
                    'cub_googlenet_doc2vec_c10', 'Office31', 'LandUse-21', 'Hdigit'])
parser.add_argument('--batch_size', default=200)
parser.add_argument('--pre_epoch', default=100)
parser.add_argument('--epoch', default=50)
parser.add_argument('--unaligned_rate', default=1)
parser.add_argument('--alpha', default=0.1)
parser.add_argument('--beta', default=0.1)
parser.add_argument('--seed', default=123)
parser.add_argument('--save', default=0) # 是否保存正式训练日志
parser.add_argument('--base', default=0)
args = parser.parse_args()

print("# args: dataset: {}, batch_size: {}, pre_epoch: {}, epoch: {}, unaligned_rate: {}, alpha: {}, beta: {}, seed: {}, base: {}"
      .format(args.dataset, args.batch_size, args.pre_epoch, args.epoch, args.unaligned_rate, args.alpha, args.beta, args.seed, args.base))

# TODO: 随机种子设定
seed = int(args.seed)
print("seed: ",seed)
np.random.seed(seed)
random.seed(seed)

intermediate_dim = [500, 500, 2000]
latent_dim = 10
print("###  shuffle data   ###")
X, Y = preprocess_data(args.dataset, float(args.unaligned_rate), int(args.base))
# 获取每个 X 的列维度
original_dim = [X_i.shape[1] for X_i in X]
print("original_dim: ", original_dim)
num_views = len(X)

pre_weights_path = 'shuffled_pretrain/batch_norm_trick/'
# 正式训练
model = My_Model(int(args.batch_size), num_views, latent_dim, intermediate_dim,
                 config_init(original_dim, args.dataset, int(args.epoch), float(args.alpha), float(args.beta)),
                 pre_weights_path, args.dataset, int(args.save),
                 ispretrain=True)
model.compile(X, Y)
print("###  train model  ###")
model.train(X)

# weights_path = 'shuffled_pretrain/batch_norm_trick_CL/'
# model.my_model.save_weights(weights_path + args.dataset + '.h5')

