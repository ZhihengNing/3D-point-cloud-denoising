# 对点云进行平滑处理
import torch
from scipy import spatial


def get_mean_displacement(original, predict):
    kdtree = spatial.cKDTree(original, 10)
    nearest_neighbours = torch.tensor(kdtree.query(original, 100)[1])
    # 预测值减去原来的点云
    displacement_vectors = predict - original
    # 得到最近的100个点的索引
    new_displacement = displacement_vectors[nearest_neighbours]
    # 最近100个点的点坐标平均值
    new_displacement = new_displacement.mean(1)
    new_points = predict - new_displacement
    return new_points


# 注意此次模型中 0才是需要的
def acc_precision_recall(predictions, targets):
    tp = torch.sum((predictions == 0) & (targets == 0))
    fp = torch.sum((predictions == 0) & (targets == 1))
    fn = torch.sum((predictions == 1) & (targets == 0))
    tn = torch.sum((predictions == 1) & (targets == 1))
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return accuracy, precision, recall


def fa_score(precision, recall, a):
    return (1 + a ** 2) * (precision * recall) / ((a ** 2 * precision) + recall)
