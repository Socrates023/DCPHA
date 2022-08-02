import torch
from load_data import get_loader
from evaluate import topALL
from test_model import test_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# data parameters
dataset = "ADNI2"
DATA_DIR = './data/{}/'.format(dataset)
batch_size = 20
# print('...Data loading is beginning...')
data_loader = get_loader(DATA_DIR, batch_size)
hashcodes = ["16bits","32bits","64bits","128bits"]
for hashcode in hashcodes:
    print("*" * 50)
    print(hashcode)
    model_path = "./pretrain_models/DCPHA_{}_{}.pth".format(dataset, hashcode)
    model_ft = torch.load(model_path,map_location=device)
    view1_feature, view2_feature, label = test_model(model_ft,data_loader)
    MRI_PET = topALL(view1_feature, view2_feature, label)
    PET_MRI = topALL(view2_feature, view1_feature, label)
    print('MRI → PET = {:.4f}'.format(MRI_PET))
    print('PET → MRI = {:.4f}'.format(PET_MRI))
    print(' Aver mAP = {:.4f}'.format(((MRI_PET + PET_MRI) / 2.)))