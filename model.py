import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, num_classes: int = 39):
        super().__init__()

        # 28
        self.conv1 = nn.Conv2d(1, 32, 3) # 26
        self.conv2 = nn.Conv2d(32, 64, 3) # 24
        self.pool1 = nn.AvgPool2d(2, 2) # 12

        self.conv3 = nn.Conv2d(64, 128, 3) #10
        self.conv4 = nn.Conv2d(128, 256, 3) #8
        self.pool2 = nn.MaxPool2d(2, 2) # 4

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
