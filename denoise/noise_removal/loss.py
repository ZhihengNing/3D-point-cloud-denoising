# 损失函数想表达的就是对于某个点云片进行去噪，我们说（找到2-范数即可）
# 这里min shape[1,500] 找的是矩阵每列最小值 max 找的是矩阵每行最大值
# 只需要找到\average (\sigma 0.99*min+0.01*max 即可)
# 极端情况下，忽略最大值，我们函数的优化方向就是寻找最小值的最小值
# 所以整个损失函数也不会小
# predict [64,3] target(干净的点云) [64,500,3]
import torch


def noise_loss(one_predict, one_target):
    # m1 shape[64,500,3]
    m1 = one_predict.expand(one_target.shape[1], -1, -1).transpose(0, 1)
    # m2 shape[64,500,3]
    m2 = one_target
    # 这里得到的是shape[64,500]的矩阵，其中500的意思就是500个点每个点与预测点的欧拉距离的平方，
    temp_loss = (m1 - m2).pow(2).sum(2)
    alpha = 0.99
    # 如果alpha=1,那么显然是最小值的平均值，最小值也就意味着在64行里面每一行的最小值（也就是500个点中存在一个距离最近的点)
    # 但若是这样，就很有可能只拟合其中任意的一个点这样效果不好，
    loss = torch.mean(alpha * torch.min(temp_loss, dim=1)[0] + (1 - alpha) * torch.max(temp_loss, dim=1)[0])
    return loss


def noise_loss2(one_predict, one_target):
    # m1 shape[64,500,3]
    m1 = one_predict.expand(one_target.shape[1], -1, -1).transpose(0, 1)
    # m2 shape[64,500,3]
    m2 = one_target
    # 这里得到的是shape[64,500]的矩阵，其中500的意思就是500个点每个点与预测点的欧拉距离的平方，
    temp_loss = (m1 - m2).pow(2).sum(2)
    loss = torch.mean(temp_loss)
    return loss
