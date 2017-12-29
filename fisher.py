import numpy as np
from scipy.linalg import solve


# LDA线性分类器
class FisherLDA(object):
    
    def __init__(self):
        self.feature_size = 0
        self.pos_value = 0
        self.neg_value = 0
        self.pos_average_value = 0
        self.neg_average_value = 0
        self.pos_num = 0
        self.neg_num = 0
    

    def read_data(self, positive_dataset, negative_dataset):
        
        if isinstance(positive_dataset, list):
            self.pos_value = np.array(positive_dataset)
        
        if isinstance(negative_dataset, list):
            self.neg_value = np.array(negative_dataset)
            
        # 计算出数据的数目
        self.pos_num = len(self.pos_value)
        self.neg_num = len(self.neg_value)

        # 计算出每个数据的特征维度
        self.feature_size = self.pos_value.shape[1]

        # 计算出数据每个特征的平均值
        self.pos_average_value = np.sum(self.pos_value, axis=0) / self.pos_num
        self.neg_average_value = np.sum(self.neg_value, axis=0) / self.neg_num

        # 计算出离差矩阵
        pos_diff = self.pos_value - self.pos_average_value
        neg_diff = self.neg_value - self.neg_average_value
        
        # 计算出偏差矩阵
        S = np.matmul(pos_diff.T, pos_diff) + np.matmul(neg_diff.T, neg_diff)
        print(S)

        # 计算出weight
        W = solve(S, np.array(self.pos_average_value-self.neg_average_value))

        # 计算出bias
        Ahat = np.matmul(W, self.pos_average_value.T)
        Bhat = np.matmul(W, self.neg_average_value.T)
        b = (Ahat*self.pos_num + Bhat*self.neg_num) / (self.pos_num + self.neg_num)

        return(W, b)
 #       return result

if __name__ == '__main__':
    
    model = FisherLDA()

    # Fake data
    pos_data = [[0, 0], [1, 0]]
    neg_data = [[2, 0], [0, 2]]

    W, b = model.read_data(pos_data, neg_data)
    print(W, b)

