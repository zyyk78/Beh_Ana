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
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler


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


def cluster_matrix_hierarchical(tot_mix_data, max_features=20, n_clusters=2, metric='correlation', linkage_method='average'):
    from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
    from scipy.spatial.distance import pdist
    """
    对细胞（行）进行聚类
    """
    X = tot_mix_data 
    
    dist_matrix = pdist(X, metric=metric)
    
    Z = linkage(dist_matrix, method=linkage_method)
    
    sorted_idx = leaves_list(Z)
    
    features = (n_clusters, X.shape[0]) 
    
    cluster = fcluster(Z, t=n_clusters, criterion='maxclust')
    
    return features, sorted_idx, cluster, Z

def match_segments(eve_list,signal,gap_threshold=50000):
    mins_events = [item for item in eve_list if item[1] in ['MinS_S', 'MinS_E']]
    mins_events.sort(key=lambda x: x[0])  # 确保按时间戳升序

    segments = []
    current_segment = [mins_events[0]]

    for i in range(1, len(mins_events)):
        prev_ts = mins_events[i-1][0]
        curr_ts = mins_events[i][0]
        
        if curr_ts - prev_ts > gap_threshold:
            # 间隔超过1s，保存当前段，开启新段
            segments.append(current_segment)
            current_segment = [mins_events[i]]
        else:
            # 属于同一段
            current_segment.append(mins_events[i])

    # 别忘了添加最后一段
    segments.append(current_segment)

    # 3. 统计结果
    print(f"检测完成：共发现 {len(segments)} 段记录。\n")
    print(f"{'段落':<10} | {'开始时间':<15} | {'结束时间':<15} | {'持续帧数':<10}")
    print("-" * 60)

    most_like_seg=(None,np.inf)

    for idx, seg in enumerate(segments):
        start_ts = seg[0][0]
        end_ts = seg[-1][0]
        duration = len(seg)
        delta=abs(duration-signal.shape[1])
        if delta<10 and delta<most_like_seg[1]:
            most_like_seg=(idx,delta)
        print(f"Segment {idx+1:<2} | {start_ts:<15} | {end_ts:<15} | {duration:<10}")
    print('Len of MS Record: ',signal.shape[1])
    print('Dropped frams: ',most_like_seg[1])
    print('Most like seq is: ',most_like_seg[0]+1)

    for i in range(len(eve_list)):
        if eve_list[0][0] < segments[most_like_seg[0]][0][0]:
            eve_list.pop(0)
        else:
            break
    for i in range(len(eve_list)):
        if eve_list[-1][0] > segments[most_like_seg[0]][-1][0]:
            eve_list.pop()
        else:
            break
    if most_like_seg[1] >0:
        insert_positions = np.sort(np.unique(np.random.randint(1, signal.shape[1] + 1, size=most_like_seg[1])))
        new_signal = np.zeros((signal.shape[0], signal.shape[1] + most_like_seg[1]))
        current_col = 0
        for i in range(new_signal.shape[1]):
            if i in insert_positions:
                new_signal[:, i] = new_signal[:, i - 1]
            else:
                new_signal[:, i] = signal[:, current_col]
                current_col += 1
        signal=new_signal
        print('已随机插入',most_like_seg[1],'帧用以补全丢帧')
    return eve_list,signal

class GLMDecoder:
    def __init__(self, eve_list, signal):
        self.eve_list = eve_list
        self.signal = signal  # (nNeurons, nFrames)
        self.df = self._prepare_trials()

    def _prepare_trials(self):
        trial_events = ['SlnL', 'SlnR', 'ROmL', 'ROmR']
        ms_frames = np.array([t for t, e in self.eve_list if e in ('MinS_S', 'MinS_E')])

        rows = []
        for t, e in self.eve_list:
            if e in trial_events:
                f_idx = np.argmin(np.abs(ms_frames - t))
                rows.append({
                    'frame': f_idx,
                    'choice': 1 if 'R' in e else 0,
                    'reward': 1 if 'Sln' in e else 0
                })

        df = pd.DataFrame(rows)
        df['prev_choice'] = df['choice'].shift(1)
        df['prev_reward'] = df['reward'].shift(1)
        return df.dropna().reset_index(drop=True)

    def _get_window_feature(self, frames, window):
        """
        window: (start, end), e.g. (-12, 0)
        return: (nTrials, nNeurons)
        """
        start, end = window
        feats = []

        for f in frames:
            f = int(f)
            if f + start < 0 or f + end > self.signal.shape[1]:
                feats.append(np.zeros(self.signal.shape[0]))
                continue

            chunk = self.signal[:, f + start : f + end]
            feats.append(chunk.mean(axis=1))  # 时间平均

        return np.array(feats)

    def fit(self, target_name='choice'):
        targets = self.df[target_name].values
        frames = self.df['frame'].values

        windows = {
            'pre':  (-12, 0),
            'post': (1, 13),
            'full': (-12, 13)
        }

        print(f"\n--- 解码目标: {target_name} ---")

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        results = {}

        for name, win in windows.items():
            X = self._get_window_feature(frames, win)
            X = StandardScaler().fit_transform(X)

            clf = LogisticRegression(
                penalty='l2',
                solver='liblinear',
                max_iter=1000
            )

            acc = cross_val_score(clf, X, targets, cv=cv).mean()
            results[name] = acc
            print(f"{name:5} window | Accuracy = {acc:.3f}")

        return results


def compute_place_fields(signal, pos_segment, chunknum=25, smooth_sigma=1.0, min_occupancy=5):
    """
    计算每个细胞的 place field

    参数:
        signal: 神经活动信号, 形状 (n_neurons, n_frames)
        pos_segment: 位置坐标, 形状 (n_frames, 2), 分别是 x 和 y
        chunknum: 空间分箱数
        smooth_sigma: 高斯平滑 sigma
        min_occupancy: 最小占据次数，只有某位置被访问超过此次数时才计算（排除偶然性放电）

    返回:
        place_fields: dict, 每个神经元的 place field 信息
        place_scores: dict, 每个神经元的综合打分
    """
    from scipy.ndimage import gaussian_filter

    # 1. 空间分箱设置
    x_pos = pos_segment[:, 0]
    y_pos = pos_segment[:, 1]
    x_min, x_max = x_pos.min(), x_pos.max()
    y_min, y_max = y_pos.min(), y_pos.max()

    chunkScale_x = (x_max - x_min) / chunknum
    chunkScale_y = (y_max - y_min) / chunknum

    # 将坐标数字化为 bin 索引
    x_indices = np.floor((x_pos - x_min) / chunkScale_x).astype(int)
    y_indices = np.floor((y_pos - y_min) / chunkScale_y).astype(int)
    x_indices = np.clip(x_indices, 0, chunknum - 1)
    y_indices = np.clip(y_indices, 0, chunknum - 1)

    # 2. 计算占据图 (Occupancy Map)
    occupancy_map = np.zeros((chunknum, chunknum))
    for x, y in zip(x_indices, y_indices):
        occupancy_map[y, x] += 1

    # 3. 计算每个细胞的 place field
    num_neurons = signal.shape[0]
    place_fields = {}
    place_scores = {}

    for neuron_id in range(num_neurons):
        # 获取神经活动信号
        activity_signal = signal[neuron_id, :].copy()

        # 过滤负值
        activity_signal[activity_signal < 0] = 0

        # 计算累积图 (Spike Map)
        spike_map = np.zeros((chunknum, chunknum))
        for i in range(len(activity_signal)):
            spike_map[y_indices[i], x_indices[i]] += activity_signal[i]

        # 计算 Rate Map（排除偶然性放电：只有占据次数超过阈值的bin才计算）
        with np.errstate(divide='ignore', invalid='ignore'):
            # 使用掩码：只有占据次数超过 min_occupancy 的位置才有效
            valid_occupancy = np.where(occupancy_map >= min_occupancy, occupancy_map, np.nan)
            rate_map = spike_map / valid_occupancy
            rate_map[valid_occupancy == np.nan] = np.nan

        # 高斯平滑
        rate_map_filled = rate_map.copy()
        rate_map_filled[np.isnan(rate_map_filled)] = 0
        smoothed_map = gaussian_filter(rate_map_filled, sigma=smooth_sigma)

        # 计算峰值
        peak_rate = np.nanmax(smoothed_map)

        # 计算打分（只考虑有效位置：占据次数 >= min_occupancy）
        rate_map_flat = rate_map.flatten()
        occupancy_flat = occupancy_map.flatten()
        # 使用有效占据次数（>= min_occupancy）来计算概率
        valid_occupancy_flat = np.where(occupancy_flat >= min_occupancy, occupancy_flat, 0)
        total_valid_occupancy = valid_occupancy_flat.sum()

        if total_valid_occupancy > 0 and peak_rate > 0:
            p_i = valid_occupancy_flat / total_valid_occupancy  # 占据概率（只考虑有效位置）
            r_i = rate_map_flat  # firing rate
            r_mean = np.nansum(r_i * p_i)  # 平均 firing rate

            # 过滤掉 NaN、0 和无效位置（占据次数 < min_occupancy）
            valid_mask = ~np.isnan(r_i) & (occupancy_flat >= min_occupancy) & (r_i > 0)
            if r_mean > 0 and valid_mask.sum() > 0:
                # Spatial Information Score (bits/spike)
                spatial_info = np.nansum(p_i[valid_mask] * r_i[valid_mask] *
                                        np.log2(r_i[valid_mask] / r_mean)) / r_mean
                # Sparsity
                sparsity = 1 - (np.nansum(p_i[valid_mask] * r_i[valid_mask])**2 /
                              np.nansum(p_i[valid_mask] * r_i[valid_mask]**2))
            else:
                spatial_info = 0
                sparsity = 0

            # Coherence (空间一致性)
            if smoothed_map.shape[0] > 1 and smoothed_map.shape[1] > 1:
                neighbors = smoothed_map[:, :-1].flatten(), smoothed_map[:, 1:].flatten()
                horizontal_corr = np.corrcoef(neighbors[0], neighbors[1])[0, 1]
                neighbors = smoothed_map[:-1, :].flatten(), smoothed_map[1:, :].flatten()
                vertical_corr = np.corrcoef(neighbors[0], neighbors[1])[0, 1]
                coherence = np.nanmean([horizontal_corr, vertical_corr])
            else:
                coherence = 0
        else:
            spatial_info = 0
            sparsity = 0
            coherence = 0

        # 存储结果
        place_fields[neuron_id] = {
            'rate_map': smoothed_map,
            'peak_rate': peak_rate,
            'spatial_info': spatial_info,
            'sparsity': sparsity,
            'coherence': coherence
        }

        # 综合打分 (可自定义权重)
        combined_score = spatial_info * 0.5 + sparsity * 0.3 + coherence * 0.2
        place_scores[neuron_id] = combined_score

    return place_fields, place_scores