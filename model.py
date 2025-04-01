import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import fcn_resnet101


class Model(nn.Module):
    def __init__(self, num_classes: int = 182):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 6, 3)
        self.avg1 = nn.AvgPool2d(2, 2)

        self.conv2 = nn.Conv2d(6, 16, 3)
        self.avg2 = nn.AvgPool2d(2, 2)

        self.conv3 = nn.Conv2d(16, 120, 3)
        self.avg3 = nn.AvgPool2d(2, 2)

        self.fc1 = nn.Linear(120 * 14 * 14, 84)
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.avg1(x)

        x = F.relu(self.conv2(x))
        x = self.avg2(x)

        x = F.relu(self.conv3(x))
        x = self.avg3(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
