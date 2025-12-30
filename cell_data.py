import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans,AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.stats import zscore,entropy
from scipy.signal import find_peaks
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

def calculate_purity(data):
    shape=data.shape
    data=data.reshape(shape[0],-1)
    Hs=[]
    Ps=[]
    for slice in range(data.shape[1]):
        # 离散化
        bins = [-np.inf] + list(np.linspace(-1, 1, 6)) + [np.inf]
        hist, _ = np.histogram(data[:,slice].flatten(), bins=bins)
        prob = hist / np.sum(hist)
        prob = prob[prob > 0]  # 移除0概率
        
        # 计算熵
        H = -np.sum(prob * np.log2(prob))
        
        # 选择纯度计算方法
        purity = 1 - H / np.log2(len(bins))
        Hs.append(H)
        Ps.append(purity)
    return np.array(Hs).reshape(shape[1:]), np.array(Ps).reshape(shape[1:])

def peak_aware_sort_indices(signal_matrix, 
                          pos_threshold=0.1, 
                          neg_threshold=-0.1):
    """
    返回按复杂峰值规则排序的索引
    
    参数:
        signal_matrix: (m,n)的numpy数组，每行代表一条信号
        pos_threshold: 正峰阈值（默认0.1）
        neg_threshold: 负峰阈值（默认-0.1）
        
    返回:
        sort_indices: 排序后的行索引数组
        group_labels: 分组标签数组（1=正峰, -1=负峰, 0=无峰）
    """
    # 存储各组的索引和排序依据
    pos_peaks = []  # (first_peak_pos, row_idx, peak_val)
    neg_peaks = []  # (last_peak_pos, row_idx, peak_val) 
    no_peaks = []   # (mean_val, row_idx)
    
    for row_idx, signal in enumerate(signal_matrix):
        # 检测正峰（> pos_threshold）
        pos_pos, _ = find_peaks(signal, height=pos_threshold)
        has_pos = len(pos_pos) > 0
        
        # 检测负峰（< neg_threshold）
        neg_pos, _ = find_peaks(-signal, height=-neg_threshold)
        has_neg = len(neg_pos) > 0
        
        if has_pos:
            # 取第一个正峰的位置和值
            first_pos = np.min(pos_pos)
            peak_val = signal[first_pos]
            pos_peaks.append((first_pos, row_idx, peak_val))
        elif has_neg:
            # 取最后一个负峰的位置和值（反向排序关键）
            last_neg = np.max(neg_pos)
            peak_val = signal[last_neg]
            neg_peaks.append((last_neg, row_idx, peak_val))
        else:
            # 无显著峰，按平均值排序
            mean_val = np.mean(signal)
            no_peaks.append((mean_val, row_idx))
    
    # 分组排序逻辑
    # 1. 正峰组：按首次出现位置升序
    pos_sorted = sorted(pos_peaks, key=lambda x: x[0])
    
    # 2. 负峰组：按最后出现位置降序（实现从后向前）
    neg_sorted = sorted(neg_peaks, key=lambda x: x[0])
    
    # 3. 无峰组：按平均值升序
    no_peaks_sorted = sorted(no_peaks, key=lambda x: -x[0])
    

    sort_indices = (
        [x[1] for x in pos_sorted] +
        [x[1] for x in no_peaks_sorted] +
        [x[1] for x in neg_sorted]
    )


    return np.array(sort_indices)



def cluster_matrix(matrix, n_clusters='auto', max_features=50, random_state=42):
    """
    对矩阵进行聚类并返回排序后的索引
    
    参数:
        matrix: numpy数组, 形状为(n_samples, n_features)
        n_clusters: int或'auto'。如果是'auto'则自动选择最佳聚类数(2-5)
        max_features: 当特征数超过此值时自动启用PCA降维
        random_state: 随机种子
    
    返回:
        clusters: 聚类标签数组, 形状(n_samples,)
        sorted_indices: 按聚类排序后的原矩阵索引
        sorted_matrix: 按聚类排序后的矩阵(可选)
    """
    
    # 2. 自动降维(如果特征太多)
    if matrix.shape[1] > max_features:
        pca = PCA(n_components=max_features, random_state=random_state)
        scaled_data = pca.fit_transform(matrix)
        print(f"自动降维至 {max_features} 个主成分，解释方差比率: {sum(pca.explained_variance_ratio_):.2f}")
    
    # 3. 自动确定聚类数(如果未指定)
    if n_clusters == 'auto':
        best_k = 2
        best_score = -1
        for k in range(2, 6):  # 测试2-5个聚类
            kmeans = KMeans(n_clusters=k, random_state=random_state)
            labels = kmeans.fit_predict(scaled_data)
            score = silhouette_score(scaled_data, labels)
            if score > best_score:
                best_score = score
                best_k = k
        n_clusters = best_k
        print(f"自动选择聚类数: {n_clusters} (轮廓系数: {best_score:.2f})")
    
    # 4. 执行聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    clusters = kmeans.fit_predict(scaled_data)
    best_score=silhouette_score(scaled_data, clusters)
    
    # 5. 生成排序索引(按聚类标签和到质心的距离)
    distances = kmeans.transform(scaled_data)  # 每条数据到各质心的距离
    sorted_indices = np.lexsort((distances[np.arange(len(clusters)), clusters], clusters))
    
    
    return [n_clusters,best_score], sorted_indices,clusters

def cluster_matrix_hierarchical(matrix, n_clusters='auto', metric='euclidean',max_features=50, 
                              random_state=None, linkage_method='ward'):
    from scipy.spatial.distance import pdist
    """
    先构建完整层次树，后确定聚类数量的科学实现
    
    参数:
        matrix: numpy数组, 形状为(n_samples, n_features)
        n_clusters: int或'auto'。'auto'时基于轮廓系数在完整树上选择最优切割
        max_features: PCA降维阈值
        linkage_method: 连接方式 ('ward', 'complete', 'average', 'single')
    
    返回:
        [n_clusters, best_score]: 最优聚类数和轮廓系数
        sorted_indices: 按聚类排序的索引
        clusters: 聚类标签
        Z: 连接矩阵
    """
    # 1. 数据预处理
    scaled_data = matrix
    if matrix.shape[1] > max_features:
        pca = PCA(n_components=max_features, random_state=random_state)
        scaled_data = pca.fit_transform(matrix)
        print(f"降维至 {max_features} 维，解释方差: {sum(pca.explained_variance_ratio_):.2f}")
        
    distance_matrix = pdist(scaled_data,metric=metric )
    # 2. 计算完整的层次树
    Z = linkage(distance_matrix, method=linkage_method)

    scores=[]
    # 4. 后验确定聚类数（基于完整树）
    if n_clusters == 'auto':
        possible_k = range(2, min(9, matrix.shape[0]+1))  # 测试2-8个聚类
        best_k, best_score = 2, -1
        
        for k in possible_k:
            # 从完整树切割出k个聚类
            clusters = fcluster(Z, k, criterion='maxclust')
            if len(np.unique(clusters)) < 2:  # 跳过无效切割
                continue
            score = silhouette_score(scaled_data, clusters)
            scores.append(score)
            if score > best_score:
                best_k, best_score = k, score
        plt.plot(scores)
        n_clusters = best_k
        print(f"后验选择聚类数: {n_clusters} (轮廓系数: {best_score:.2f})")
    
    # 5. 最终切割
    clusters = fcluster(Z, n_clusters, criterion='maxclust')
    best_score = silhouette_score(scaled_data, clusters)
    

    # 返回排序索引（保持原顺序，仅按聚类标签排序）
    return [n_clusters, best_score], np.argsort(clusters), clusters, Z

def align_clusters(cluster_list):
    """
    对齐多个聚类结果的类标签，生成统一索引
    
    参数:
        cluster_list: 多个聚类结果的列表，如 [cluster_1, cluster_2, cluster_3]
    
    返回:
        unified_labels: 统一后的标签数组，形状 (n_samples, n_clusterings)
        parent_classes: 父类到子类的映射字典
    """
    n_samples = len(cluster_list[0])
    n_clusterings = len(cluster_list)
    
    # 初始化统一标签矩阵
    unified_labels = np.zeros((n_samples, n_clusterings), dtype=int)
    
    # 第一个聚类结果作为参考
    unified_labels[:, 0] = cluster_list[0]
    parent_classes = {i: {0: i} for i in np.unique(cluster_list[0])}
    
    # 对齐后续聚类结果
    for i in range(1, n_clusterings):
        ref_cluster = unified_labels[:, i-1]
        curr_cluster = cluster_list[i]
        
        # 计算混淆矩阵
        cm = confusion_matrix(ref_cluster, curr_cluster)
        
        # 匈牙利算法找到最优匹配
        row_ind, col_ind = linear_sum_assignment(-cm)  # 最大化匹配
        
        # 更新父类映射
        for ref_class, curr_class in zip(row_ind, col_ind):
            if ref_class not in parent_classes:
                parent_classes[ref_class] = {}
            parent_classes[ref_class][i] = curr_class
        
        # 生成统一标签
        for sample in range(n_samples):
            for ref_class, mapping in parent_classes.items():
                if (unified_labels[sample, i-1] == ref_class and 
                    curr_cluster[sample] == mapping.get(i, -1)):
                    unified_labels[sample, i] = ref_class
                    break
            else:
                unified_labels[sample, i] = -1  # 未匹配的类
    
    return unified_labels, parent_classes


def map_sorted_clusters(sorted_cluster, parent_classes):
    """
    将排序后的 cluster 列表映射为对齐后的标签版本。
    
    参数:
        sorted_cluster: 与 cluster_list 结构相同的 list，如 [cluster_1, cluster_2, ...]
        parent_classes: align_clusters 返回的映射字典
        
    返回:
        mapped_cluster: 对齐后的标签 list，形状与 sorted_cluster 相同
    """
    n_clusterings = len(sorted_cluster)
    n_samples = len(sorted_cluster[0])
    
    # 构建反向映射：{i: {子类: 父类}}，便于映射
    reverse_mapping = [{} for _ in range(n_clusterings)]
    for parent, mapping in parent_classes.items():
        for i, child in mapping.items():
            reverse_mapping[i][child] = parent

    # 映射 sorted_cluster 到统一标签
    mapped_cluster = []
    for i in range(n_clusterings):
        cluster = sorted_cluster[i]
        mapped = [reverse_mapping[i].get(label, -1) for label in cluster]
        mapped_cluster.append(mapped)

    return mapped_cluster