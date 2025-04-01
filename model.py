import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, num_classes: int = 182):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.avg1 = nn.AvgPool2d(2, 2) # 63

        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 128, 3)
        self.avg2 = nn.AvgPool2d(2, 2) # 29

        self.conv5 = nn.Conv2d(128, 256, 3)
        self.conv6 = nn.Conv2d(256, 512, 3)
        self.avg3 = nn.AvgPool2d(2, 2) # 12

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # -> 512x1x1

        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.avg1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.avg2(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.avg3(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
