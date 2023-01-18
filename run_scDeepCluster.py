from time import time
import math, os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from scDeepCluster import scDeepCluster
from single_cell_tools import *
import numpy as np
import collections
from sklearn import metrics
import h5py
import scanpy as sc
from preprocess import read_dataset, normalize

# for repeatability
torch.manual_seed(42)

if __name__ == "__main__":

    # 设置超参数
    import argparse
    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_clusters', default=0, type=int, 
                        help='number of clusters, 0 means estimating by the Louvain algorithm')
    parser.add_argument('--knn', default=20, type=int, 
                        help='number of nearest neighbors, used by the Louvain algorithm')
    parser.add_argument('--resolution', default=.8, type=float, 
                        help='resolution parameter, used by the Louvain algorithm, larger value for more number of clusters')
    parser.add_argument('--select_genes', default=0, type=int, 
                        help='number of selected genes, 0 means using all genes')
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--data_file', default='data.h5')
    parser.add_argument('--maxiter', default=2000, type=int)
    parser.add_argument('--pretrain_epochs', default=300, type=int)
    parser.add_argument('--gamma', default=1., type=float,
                        help='coefficient of clustering loss')
    parser.add_argument('--sigma', default=2.5, type=float,
                        help='coefficient of random noise')
    parser.add_argument('--update_interval', default=1, type=int)
    parser.add_argument('--tol', default=0.001, type=float,
                        help='tolerance for delta clustering labels to terminate training stage')
    parser.add_argument('--ae_weights', default=None,
                        help='file to pretrained weights, None for a new pretraining')
    parser.add_argument('--save_dir', default='results/scDeepCluster/',
                        help='directory to save model weights during the training stage')
    parser.add_argument('--ae_weight_file', default='AE_weights.pth.tar',
                        help='file name to save model weights after the pretraining stage')
    parser.add_argument('--final_latent_file', default='final_latent_file.txt',
                        help='file name to save final latent representations')
    parser.add_argument('--predict_label_file', default='pred_labels.txt',
                        help='file name to save final clustering labels')
    parser.add_argument('--device', default='cuda')

    # 解析命令行参数
    args = parser.parse_args()
    # 使用h5py打开数据文件
    data_mat = h5py.File(args.data_file, 'r')
    # 读取数据
    x = np.array(data_mat['X'])
    # y是用来评估聚类性能的真实标签，如果不存在，则跳过计算聚类性能指标（例如NMI,ARI）
    if 'Y' in data_mat:
        y = np.array(data_mat['Y'])
    else:
        y = None
    data_mat.close()

    if args.select_genes > 0:
        importantGenes = geneSelection(x, n=args.select_genes, plot=False)
        x = x[:, importantGenes]

    # 预处理scRNA-seq读数矩阵
    adata = sc.AnnData(x)
    if y is not None:
        adata.obs['Group'] = y

    adata = read_dataset(adata,
                     transpose=False,
                     test_split=False,
                     copy=True)

    adata = normalize(adata,
                      size_factors=True,
                      normalize_input=True,
                      logtrans_input=True)

    input_size = adata.n_vars

    print(args)

    print(adata.X.shape)
    if y is not None:
        print(y.shape)
    # 创建 scDeepCluster 模型
    model = scDeepCluster(input_dim=adata.n_vars, z_dim=32, 
                encodeLayer=[256, 64], decodeLayer=[64, 256], sigma=args.sigma, gamma=args.gamma, device=args.device)
    
    print(str(model))

    # 记录当前时间
    t0 = time()
    """
    如果没有给定自编码器的权重，那么就需要对自编码器进行预训练。
    使用 X，X_raw，size_factor 进行预训练，
    使用 batch_size，epochs 和 ae_weight_file 来指定训练的参数。
    如果给定了自编码器的权重，那么就加载这个文件，如果这个文件不存在，就抛出错误。
    """
    if args.ae_weights is None:
        model.pretrain_autoencoder(X=adata.X, X_raw=adata.raw.X, size_factor=adata.obs.size_factors, 
                                batch_size=args.batch_size, epochs=args.pretrain_epochs, ae_weights=args.ae_weight_file)
    else:
        if os.path.isfile(args.ae_weights):
            print("==> loading checkpoint '{}'".format(args.ae_weights))
            checkpoint = torch.load(args.ae_weights)
            model.load_state_dict(checkpoint['ae_state_dict'])
        else:
            print("==> no checkpoint found at '{}'".format(args.ae_weights))
            raise ValueError
    
    print('Pretraining time: %d seconds.' % int(time() - t0))
    # 加载参数，创建文件夹。
    if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
    # 如果需要的话，就调用 model.pretrain_autoencoder() 函数进行预训练。
    if args.n_clusters > 0:
        # 调用 model.fit() 函数进行聚类
        y_pred, _, _, _, _ = model.fit(X=adata.X, X_raw=adata.raw.X, size_factor=adata.obs.size_factors, n_clusters=args.n_clusters, init_centroid=None, 
                    y_pred_init=None, y=y, batch_size=args.batch_size, num_epochs=args.maxiter, update_interval=args.update_interval, tol=args.tol, save_dir=args.save_dir)
    # 如果不需要，则读取预训练的模型。
    else:
        # 使用autoencoder的隐层表示估计聚类数目 by Louvain algorithm 
        pretrain_latent = model.encodeBatch(torch.tensor(adata.X, dtype=torch.float32)).cpu().numpy()
        adata_latent = sc.AnnData(pretrain_latent)
        sc.pp.neighbors(adata_latent, n_neighbors=args.knn, use_rep="X")
        sc.tl.louvain(adata_latent, resolution=args.resolution)
        y_pred_init = np.asarray(adata_latent.obs['louvain'],dtype=int)
        features = pd.DataFrame(adata_latent.X,index=np.arange(0,adata_latent.n_obs))
        Group = pd.Series(y_pred_init,index=np.arange(0,adata_latent.n_obs),name="Group")
        Mergefeature = pd.concat([features,Group],axis=1)
        cluster_centers = np.asarray(Mergefeature.groupby("Group").mean())
        n_clusters = cluster_centers.shape[0]
        print('Estimated number of clusters: ', n_clusters)
        # 调用 model.fit() 函数进行聚类
        y_pred, _, _, _, _ = model.fit(X=adata.X, X_raw=adata.raw.X, size_factor=adata.obs.size_factors, n_clusters=n_clusters, init_centroid=cluster_centers, 
                    y_pred_init=y_pred_init, y=y, batch_size=args.batch_size, num_epochs=args.maxiter, update_interval=args.update_interval, tol=args.tol, save_dir=args.save_dir)


    print('Total time: %d seconds.' % int(time() - t0))

    # 如果 y 不为空，则使用 NMI 和 ARI 指标评估聚类结果的质量。
    if y is not None:
        nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
        ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
        print('Evaluating cells: NMI= %.4f, ARI= %.4f' % (nmi, ari))
    # 将最终的潜在空间保存在 "final_latent_file" 文件中，
    # 并将预测标签保存在 "predict_label_file" 文件中。
    final_latent_file='results/file/' +args.data_file+args.final_latent_file
    predict_label_file='results/file/' +args.data_file+args.predict_label_file
    final_latent = model.encodeBatch(torch.tensor(adata.X, dtype=torch.float32)).cpu().numpy()
    np.savetxt(final_latent_file, final_latent, delimiter=",")
    np.savetxt(predict_label_file, y_pred, delimiter=",", fmt="%i")

