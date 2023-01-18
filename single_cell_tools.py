import numpy as np
import pylab as plt
import seaborn as sns
import pandas as pd
from scipy import sparse

def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    # 将y_true转换为int64类型
    y_true = y_true.astype(np.int64)
    # 检查y_true和y_pred是否有相同数量的样本
    assert y_pred.size == y_true.size
    # 定义D为y_pred和y_true中最大标签值加1，用于构建计数矩阵w
    D = max(y_pred.max(), y_true.max()) + 1
    # 构建初始为0的计数矩阵w
    w = np.zeros((D, D), dtype=np.int64)
    # 遍历y_pred，并在w矩阵对应位置上计数
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    # 从sklearn.utils.linear_assignment_中导入linear_assignment函数，并调用它
    from sklearn.utils.linear_assignment_ import linear_assignment
    # 计算矩阵w中值最大的部分
    ind = linear_assignment(w.max() - w)
    # 计算计数总和
    count = sum([w[i, j] for i, j in ind])
    # 计算准确度
    acc = count * 1.0 / y_pred.size
    return acc

def geneSelection(data, threshold=0, atleast=10, 
                  yoffset=.02, xoffset=5, decay=1.5, n=None, 
                  plot=True, markers=None, genes=None, figsize=(6,3.5),
                  markeroffsets=None, labelsize=10, alpha=1, verbose=1):
    """
    选择基因
    # 参数
        data: 输入数据, 稀疏矩阵或者普通数组
        threshold: 阈值，低于该阈值的数据将被视为0, 默认为0
        atleast: 限制一个基因至少被检测到的次数, 默认为10
        yoffset: y轴的偏移量, 默认为0.02
        xoffset: x轴的偏移量, 默认为5
        decay: 衰减因子, 默认为1.5
        n: 保留的基因数量，默认为None
        plot: 是否绘制图表, 默认为True
        markers: 用于标记的数组, 默认为None
        genes: 基因名称列表, 默认为None
        figsize: 图表尺寸, 默认为(6,3.5)
        markeroffsets: 标记的偏移量, 默认为None
        labelsize: 标签字体大小, 默认为10
        alpha: 图表透明度, 默认为1
        verbose: 输出信息级别, 默认为1
    # 返回值
        选定的基因索引
    """
    if sparse.issparse(data):
        zeroRate = 1 - np.squeeze(np.array((data>threshold).mean(axis=0)))
        A = data.multiply(data>threshold)
        A.data = np.log2(A.data)
        meanExpr = np.zeros_like(zeroRate) * np.nan
        detected = zeroRate < 1
        meanExpr[detected] = np.squeeze(np.array(A[:,detected].mean(axis=0))) / (1-zeroRate[detected])
    else:
        zeroRate = 1 - np.mean(data>threshold, axis=0)
        meanExpr = np.zeros_like(zeroRate) * np.nan
        detected = zeroRate < 1
        mask = data[:,detected]>threshold
        logs = np.zeros_like(data[:,detected]) * np.nan
        logs[mask] = np.log2(data[:,detected][mask])
        meanExpr[detected] = np.nanmean(logs, axis=0)
    # 找到基因检测次数不足atleast次的基因
    lowDetection = np.array(np.sum(data>threshold, axis=0)).squeeze() < atleast
    zeroRate[lowDetection] = np.nan
    meanExpr[lowDetection] = np.nan
            
    if n is not None:
        up = 10
        low = 0
        for t in range(100):
            nonan = ~np.isnan(zeroRate)
            selected = np.zeros_like(zeroRate).astype(bool)
            # 选定的基因的条件
            selected[nonan] = zeroRate[nonan] > np.exp(-decay*(meanExpr[nonan] - xoffset)) + yoffset
            if np.sum(selected) == n:
                break
            elif np.sum(selected) < n:
                up = xoffset
                xoffset = (xoffset + low)/2
            else:
                low = xoffset
                xoffset = (xoffset + up)/2
        if verbose>0:
            print('Chosen offset: {:.2f}'.format(xoffset))
    else:
        nonan = ~np.isnan(zeroRate)
        selected = np.zeros_like(zeroRate).astype(bool)
        selected[nonan] = zeroRate[nonan] > np.exp(-decay*(meanExpr[nonan] - xoffset)) + yoffset
                
    if plot:
        if figsize is not None:
            plt.figure(figsize=figsize)
        plt.ylim([0, 1])
        if threshold>0:
            plt.xlim([np.log2(threshold), np.ceil(np.nanmax(meanExpr))])
        else:
            plt.xlim([0, np.ceil(np.nanmax(meanExpr))])
        x = np.arange(plt.xlim()[0], plt.xlim()[1]+.1,.1)
        y = np.exp(-decay*(x - xoffset)) + yoffset
        if decay==1:
            plt.text(.4, 0.2, '{} genes selected\ny = exp(-x+{:.2f})+{:.2f}'.format(np.sum(selected),xoffset, yoffset), 
                     color='k', fontsize=labelsize, transform=plt.gca().transAxes)
        else:
            plt.text(.4, 0.2, '{} genes selected\ny = exp(-{:.1f}*(x-{:.2f}))+{:.2f}'.format(np.sum(selected),decay,xoffset, yoffset), 
                     color='k', fontsize=labelsize, transform=plt.gca().transAxes)

        plt.plot(x, y, color=sns.color_palette()[1], linewidth=2)
        xy = np.concatenate((np.concatenate((x[:,None],y[:,None]),axis=1), np.array([[plt.xlim()[1], 1]])))
        t = plt.matplotlib.patches.Polygon(xy, color=sns.color_palette()[1], alpha=.4)
        plt.gca().add_patch(t)
        
        plt.scatter(meanExpr, zeroRate, s=1, alpha=alpha, rasterized=True)
        if threshold==0:
            plt.xlabel('Mean log2 nonzero expression')
            plt.ylabel('Frequency of zero expression')
        else:
            plt.xlabel('Mean log2 nonzero expression')
            plt.ylabel('Frequency of near-zero expression')
        plt.tight_layout()
        
        if markers is not None and genes is not None:
            if markeroffsets is None:
                markeroffsets = [(0, 0) for g in markers]
            for num,g in enumerate(markers):
                i = np.where(genes==g)[0]
                plt.scatter(meanExpr[i], zeroRate[i], s=10, color='k')
                dx, dy = markeroffsets[num]
                plt.text(meanExpr[i]+dx+.1, zeroRate[i]+dy, g, color='k', fontsize=labelsize)
    
    return selected