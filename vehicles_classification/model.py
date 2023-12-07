import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class MultiLabelCNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MultiLabelCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * (input_size // 4) * (input_size // 4), 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 128 * (x.size(2) // 4) * (x.size(3) // 4))
        x = F.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


def main():
    in_size = 640
    mlcnn = MultiLabelCNN(input_size=in_size, num_classes=5)

if __name__ == '__main__':
    main()