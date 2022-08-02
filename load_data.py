from torch.utils.data.dataset import Dataset
from scipy.io import loadmat, savemat
from torch.utils.data import DataLoader
import numpy as np

class CustomDataSet(Dataset):
    def __init__(self,mod1,mod2,labels):
        self.mod1 = mod1
        self.mod2 = mod2
        self.labels = labels

    def __getitem__(self, index):
        mod1 = self.mod1[index]
        mod2 = self.mod2[index]
        label = self.labels[index]
        return mod1, mod2, label

    def __len__(self):
        count = len(self.mod1)
        assert len(self.mod1) == len(self.labels)
        return count

def ont_hot(labels):
    # 假设y是np.array数组形式的数组，形状为（row, col)
    # t是类别标签，也为数组，形状为(row,)，比如[0,2,3,5,2,6]
    one_hot_t = np.zeros(shape=(len(labels),3))  # 生成和y形状一样的元素为零的数组
    for i in range(len(labels)):
        one_hot_t[i][labels[i]-1] = 1
    return one_hot_t

def get_ADNIloader(path, batch_size):

    MRI_test = loadmat(path + "test_MRI.mat")['test_MRI']
    PET_test = loadmat(path + "test_PET.mat")['test_PET']
    label_test = np.squeeze(loadmat(path + "test_lb.mat")['test_lb'])
    label_test = ont_hot(label_test)
    dataset = CustomDataSet(mod1=MRI_test, mod2=PET_test, labels=label_test)
    dataloader = DataLoader(dataset, batch_size=batch_size,shuffle=False, num_workers=0)
    return dataloader


def get_loader(path,batch_size):
    path_list = path.split("/")
    print("loading {} dataset!".format(path_list[-2]))
    return get_ADNIloader(path,batch_size)