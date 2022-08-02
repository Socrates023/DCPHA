import torch
import torch.nn as nn
from torchvision import models

class FeatureDecoder(nn.Module):
    def __init__(self):
        super(FeatureDecoder, self).__init__()

        backbone = models.alexnet(pretrained=False)
        self.features = backbone.features
        self.fc = nn.Sequential(*list(backbone.classifier.children())[:-2])
        self.SAM = nn.Linear(4096, 3)


    def forward(self, x):
        x = torch.transpose(x,1,-1).float()
        features = self.features(x).view(x.shape[0], -1)
        features = self.fc(features)
        predict = self.SAM(features)
        return features, predict


class HashDecoder(nn.Module):
    def __init__(self, hash_len):
        super(HashDecoder, self).__init__()
        self.HashLayer = nn.Linear(4096, hash_len)

        self.SAM = nn.Linear(hash_len, 3)

    def forward(self, x):
        hash_values = self.HashLayer(x)
        hash_labels = self.SAM(hash_values)
        return hash_values, hash_labels


class DCPHA(nn.Module):

    def __init__(self, hash_len=16):
        super(DCPHA, self).__init__()
        self.FeatureLearningSubmodel = FeatureDecoder()
        self.HashcodeLearningSubmodel = HashDecoder(hash_len)

    def forward(self, inputs):
        features, feat_predict = self.FeatureLearningSubmodel(inputs)
        hash_values, hash_labels = self.HashcodeLearningSubmodel(features)
        return hash_values, feat_predict, hash_labels

if __name__ == '__main__':
    backbone = models.alexnet(pretrained=False)
    print(backbone)