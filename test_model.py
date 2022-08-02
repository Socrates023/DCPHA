import torch
import numpy as np

def test_model(model_backbone,data_loaders):
    MRI_imgs, PET_imgs, t_labels = [], [], []
    with torch.no_grad():
        for MRIs, PETs, labels in data_loaders:
            if torch.cuda.is_available():
                MRIs = MRIs.cuda()
                PETs = PETs.cuda()
                labels = labels.cuda()
            mod1_hashcode,_ , _ = model_backbone(MRIs)
            mod2_hashcode,_ , _= model_backbone(PETs)
            MRI_imgs.append(mod1_hashcode.sign().cpu().numpy())
            PET_imgs.append(mod2_hashcode.sign().cpu().numpy())
            t_labels.append(labels.cpu().numpy())
    MRI_imgs = np.concatenate(MRI_imgs)
    PET_imgs = np.concatenate(PET_imgs)
    t_labels = np.concatenate(t_labels).argmax(1)
    return MRI_imgs,PET_imgs,t_labels