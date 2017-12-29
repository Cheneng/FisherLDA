import numpy as np

# LDA线性分类器
class FisherLDA(object):
    
    def __init__(self):
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
            

if __name__ == '__main__':
    model = FisherLDA()

    # Fake data
    pos_data = [[0, 0], [1, 0]]
    neg_data = [[2, 0], [0, 2]]

    
        