import torch
from torch import nn
from torchvision import models


class MyModel(nn.Module):
    def __init__(self, num_classes=6):
        super(MyModel, self).__init__()
        self.inception = models.inception_v3(pretrained=True, aux_logits=True)

        # Замораживаем все параметры модели InceptionV3
        for param in self.inception.parameters():
            param.requires_grad = False

        # Размораживаем последние несколько слоев
        for param in self.inception.Mixed_7c.parameters():
            param.requires_grad = True
        for param in self.inception.fc.parameters():
            param.requires_grad = True

        # Заменяем последний полносвязный слой на новый, адаптированный под задачу
        num_ftrs = self.inception.fc.in_features
        self.inception.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.inception(x)
