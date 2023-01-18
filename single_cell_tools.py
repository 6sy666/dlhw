import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from layers import ZINBLoss, MeanAct, DispAct
import numpy as np
from sklearn.cluster import KMeans
import math, os
from sklearn import metrics
#from single_cell_tools import cluster_acc

def buildNetwork(layers, type, activation="relu"):
    net = []
    for i in range(1, len(layers)):
        # 添加线性层, 输入为layers[i-1], 输出为layers[i]
        net.append(nn.Linear(layers[i-1], layers[i]))
        if activation=="relu":
            # 添加ReLU激活函数
            net.append(nn.ReLU())
        elif activation=="sigmoid":
            # 添加Sigmoid激活函数
            net.append(nn.Sigmoid())
    # 返回一个序列化的网络模型
    return nn.Sequential(*net)

def euclidean_dist(x, y):
    # 计算欧几里得距离
    return torch.sum(torch.square(x - y), dim=1)

class scDeepCluster(nn.Module):
    def __init__(self, input_dim, z_dim, encodeLayer=[], decodeLayer=[], 
            activation="relu", sigma=1., alpha=1., gamma=1., device="cuda"):
        super(scDeepCluster, self).__init__()
        self.z_dim = z_dim
        self.activation = activation
        self.sigma = sigma
        self.alpha = alpha
        self.gamma = gamma
        self.device = device
        # 构建编码器
        self.encoder = buildNetwork([input_dim]+encodeLayer, type="encode", activation=activation)
        # 构建解码器
        self.decoder = buildNetwork([z_dim]+decodeLayer, type="decode", activation=activation)
        # 构建编码器的均值层
        self._enc_mu = nn.Linear(encodeLayer[-1], z_dim)
        # 构建解码器的均值层
        self._dec_mean = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), MeanAct())
        # 构建解码器的方差层
        self._dec_disp = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), DispAct())
        # 构建解码器的pi层
        self._dec_pi = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), nn.Sigmoid())

        self.zinb_loss = ZINBLoss().to(self.device)
        self.to(device)
    
    def save_model(self, path):
        """
        保存当前模型到给定路径
        """
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        """
        从给定路径加载模型
        """
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)

    def soft_assign(self, z):
        """
        计算数据点对聚类中心的软赋值
        """
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.mu)**2, dim=2) / self.alpha)
        q = q**((self.alpha+1.0)/2.0)
        q = (q.t() / torch.sum(q, dim=1)).t()
        return q
    
    def target_distribution(self, q):
        """
        计算目标分布
        """
        p = q**2 / q.sum(0)
        return (p.t() / p.sum(1)).t()

    def forwardAE(self, x):
        """
        自编码器的前向传播
        """
        # 在输入上添加噪声
        h = self.encoder(x+torch.randn_like(x) * self.sigma)
        # 计算潜在编码
        z = self._enc_mu(h)
        # 解码潜在编码
        h = self.decoder(z)
        # 计算重构的均值
        _mean = self._dec_mean(h)
        # 计算重构的离散程度
        _disp = self._dec_disp(h)
        # 计算重构的dropout概率
        _pi = self._dec_pi(h)

        # 对原始输入进行编码
        h0 = self.encoder(x)
        # 计算原始输入的潜在编码
        z0 = self._enc_mu(h0)
        return z0, _mean, _disp, _pi

    def forward(self, x):
        """
        整个模型的前向传播
        """
        # 在输入上添加噪声
        h = self.encoder(x+torch.randn_like(x) * self.sigma)
        # 计算潜在编码
        z = self._enc_mu(h)
        # 解码潜在编码
        h = self.decoder(z)
        # 计算重构的均值
        _mean = self._dec_mean(h)
        # 计算重构的离散程度
        _disp = self._dec_disp(h)
        # 计算重构的dropout概率
        _pi = self._dec_pi(h)

        # 对原始输入进行编码
        h0 = self.encoder(x)
        # 计算原始输入的潜在编码
        z0 = self._enc_mu(h0)
        # 计算软赋值
        q = self.soft_assign(z0)
        return z0, q, _mean, _disp, _pi

    def encodeBatch(self, X, batch_size=256):
        """
        使用编码器对一个数据批次进行编码
        """
        self.eval()
        encoded = []
        num = X.shape[0]
        num_batch = int(math.ceil(1.0*X.shape[0]/batch_size))
        for batch_idx in range(num_batch):
            xbatch = X[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
            inputs = Variable(xbatch).to(self.device)
            z, _, _, _= self.forwardAE(inputs)
            encoded.append(z.data)

        encoded = torch.cat(encoded, dim=0)
        return encoded.to(self.device)
    
    def cluster_loss(self, p, q):
        """
        计算聚类损失
        """
        def kld(target, pred):
            """
            计算KL divergence
            """
            return torch.mean(torch.sum(target*torch.log(target/(pred+1e-6)), dim=-1))
        # 计算KL divergence
        kldloss = kld(p, q)
        # 返回乘以权重的KL divergence
        return self.gamma*kldloss

    def pretrain_autoencoder(self, X, X_raw, size_factor, batch_size=256, lr=0.001, epochs=400, ae_save=True, ae_weights='AE_weights.pth.tar'):
        """
        预训练自编码器
        """
        self.train()
        # 创建dataset, 数据类型为Tensor
        dataset = TensorDataset(torch.Tensor(X), torch.Tensor(X_raw), torch.Tensor(size_factor))
        # 创建DataLoader，batch_size = 256, 数据打乱
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        # 打印预训练阶段
        print("Pretraining stage")
        # 创建优化器Adam, 只对需要求导的参数进行优化
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, amsgrad=True)
        
        for epoch in range(epochs):
            # 记录损失值
            loss_val = 0
            # 遍历每个batch
            for batch_idx, (x_batch, x_raw_batch, sf_batch) in enumerate(dataloader):
                # 将数据转换为Tensor,并移到设备上
                x_tensor = Variable(x_batch).to(self.device)
                x_raw_tensor = Variable(x_raw_batch).to(self.device)
                sf_tensor = Variable(sf_batch).to(self.device)
                # 前向传播得到_,mean_tensor,disp_tensor, pi_tensor
                _, mean_tensor, disp_tensor, pi_tensor = self.forwardAE(x_tensor)
                # 计算loss
                loss = self.zinb_loss(x=x_raw_tensor, mean=mean_tensor, disp=disp_tensor, pi=pi_tensor, scale_factor=sf_tensor)
                # 参数梯度清零
                optimizer.zero_grad()
                # 反向传播
                loss.backward()
                # 更新参数
                optimizer.step()
                loss_val += loss.item() * len(x_batch)
            # 打印loss
            print('Pretrain epoch %3d, ZINB loss: %.8f' % (epoch+1, loss_val/X.shape[0]))

        # 保存训练好的网络参数
        if ae_save:
            torch.save({'ae_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, ae_weights)

    def save_checkpoint(self, state, index, filename):
        """
        保存checkpoint
        """
        newfilename = os.path.join(filename, 'FTcheckpoint_%d.pth.tar' % index)
        torch.save(state, newfilename)

    def fit(self, X, X_raw, size_factor, n_clusters, init_centroid=None, y=None, y_pred_init=None, lr=1., batch_size=256, 
            num_epochs=10, update_interval=1, tol=1e-3, save_dir=""):
        '''X: tensor data'''
        # 进行聚类训练
        self.train()
        print("Clustering stage")
        # 将X，X_raw，size_factor转换成torch的tensor类型
        X = torch.tensor(X, dtype=torch.float32)
        X_raw = torch.tensor(X_raw, dtype=torch.float32)
        size_factor = torch.tensor(size_factor, dtype=torch.float32)
        # 初始化簇中心
        self.mu = Parameter(torch.Tensor(n_clusters, self.z_dim).to(self.device))
        # 设置优化器
        optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, rho=.95)

        print("Initializing cluster centers with kmeans.")
        if init_centroid is None:
            # 如果 init_centroid 是 None，那么使用 kmeans 方法初始化聚类中心。
            kmeans = KMeans(n_clusters, n_init=20)
            data = self.encodeBatch(X)
            self.y_pred = kmeans.fit_predict(data.data.cpu().numpy())
            self.y_pred_last = self.y_pred
            self.mu.data.copy_(torch.tensor(kmeans.cluster_centers_, dtype=torch.float32))
            # 如果 init_centroid 不是 None，那么将给定的初始聚类中心复制给 self.mu。
        else:
            # self.y_pred 记录当前聚类结果，self.y_pred_last 记录上一次聚类结果。
            self.mu.data.copy_(torch.tensor(init_centroid, dtype=torch.float32))
            self.y_pred = y_pred_init
            self.y_pred_last = self.y_pred
        """
        这段代码首先判断是否有标签y，如果有就会计算K-means聚类初始化的NMI和ARI。
        然后根据样本数量和batch_size计算batch数量，并初始化最终结果。
        """
        if y is not None:
            nmi = np.round(metrics.normalized_mutual_info_score(y, self.y_pred), 5)
            ari = np.round(metrics.adjusted_rand_score(y, self.y_pred), 5)
            print('Initializing k-means: NMI= %.4f, ARI= %.4f' % (nmi, ari))
        # 计算一共有多少个batch
        num = X.shape[0]
        num_batch = int(math.ceil(1.0*X.shape[0]/batch_size))
        # 初始化结果
        final_acc, final_nmi, final_ari, final_epoch = 0, 0, 0, 0


        for epoch in range(num_epochs):
            if epoch%update_interval == 0:
                # 更新目标分布p
                latent = self.encodeBatch(X.to(self.device))
                q = self.soft_assign(latent)
                p = self.target_distribution(q).data

                # 评估聚类性能
                self.y_pred = torch.argmax(q, dim=1).data.cpu().numpy()

                if y is not None:
                    final_nmi = nmi = np.round(metrics.normalized_mutual_info_score(y, self.y_pred), 5)
                    final_epoch = ari = np.round(metrics.adjusted_rand_score(y, self.y_pred), 5)
                    print('Clustering   %d: NMI= %.4f, ARI= %.4f' % (epoch+1, nmi, ari))

                # 保存当前模型
                if (epoch>0 and delta_label < tol) or epoch%10 == 0:
                    self.save_checkpoint({'epoch': epoch+1,
                            'state_dict': self.state_dict(),
                            'mu': self.mu,
                            'y_pred': self.y_pred,
                            'y_pred_last': self.y_pred_last,
                            'y': y
                            }, epoch+1, filename=save_dir)

                # 监测停止标准
                delta_label = np.sum(self.y_pred != self.y_pred_last).astype(np.float32) / num
                self.y_pred_last = self.y_pred
                if epoch>0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print("Reach tolerance threshold. Stopping training.")
                    break


            # 训练1个epoch来优化聚类损失
            train_loss = 0.0  # 初始化训练损失为0
            recon_loss_val = 0.0  # 初始化重建损失为0
            cluster_loss_val = 0.0  # 初始化聚类损失为0
            for batch_idx in range(num_batch):
                xbatch = X[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                xrawbatch = X_raw[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                sfbatch = size_factor[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                pbatch = p[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                optimizer.zero_grad()
                inputs = Variable(xbatch).to(self.device)
                rawinputs = Variable(xrawbatch).to(self.device)
                sfinputs = Variable(sfbatch).to(self.device)
                target = Variable(pbatch).to(self.device)

                zbatch, qbatch, meanbatch, dispbatch, pibatch = self.forward(inputs)

                cluster_loss = self.cluster_loss(target, qbatch)
                recon_loss = self.zinb_loss(rawinputs, meanbatch, dispbatch, pibatch, sfinputs)

                loss = cluster_loss*self.gamma + recon_loss
                loss.backward()
                optimizer.step()
                cluster_loss_val += cluster_loss.item() * len(inputs)
                recon_loss_val += recon_loss.item() * len(inputs)
                train_loss += loss.item() * len(inputs)

            print("Epoch %3d: Total: %.8f Clustering Loss: %.8f ZINB Loss: %.8f" % (
                epoch + 1, train_loss / num, cluster_loss_val / num, recon_loss_val / num))

        return self.y_pred, final_acc, final_nmi, final_ari, final_epoch

