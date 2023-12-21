import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLabelCNN(nn.Module):
    def __init__(
        self, num_classes, input_size=640, nn_size=1, device=torch.device("cpu")
    ):
        super(MultiLabelCNN, self).__init__()

        self.nn_size = 1

        self.conv_backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.mlp_head = nn.Sequential(
            nn.Linear(nn_size * 64 * (input_size // 8) * (input_size // 8), num_classes * 16),
            nn.ReLU(),
            nn.Linear(num_classes * 16, num_classes * 16),
            nn.ReLU(),
            nn.Linear(num_classes * 16, num_classes),
        )

        self.sigmoid = nn.Sigmoid()

        # self.welcome_conv.to(device)
        # for seq in self.conv_backbone:
        #     seq.to(device)
        # self.mlp_head.to(device)
        # self.sigmoid.to(device)

    def forward(self, x):
        x = self.conv_backbone(x)
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3))
        x = self.mlp_head(x)
        x = self.sigmoid(x)
        return x


def main():
    in_size = 640
    device = torch.device("cuda:0")
    mlcnn = MultiLabelCNN(num_classes=5, input_size=in_size, nn_size=5, device=device)

    ones_shape = torch.ones((1, 3, in_size, in_size), device=device)

    res = mlcnn(ones_shape)
    print(res)


if __name__ == "__main__":
    main()
