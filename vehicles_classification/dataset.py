import os

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


class TrafficDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir

        self.transform = transform
        self.im_dir = os.path.join(data_dir, "images")
        self.images = os.listdir(self.im_dir)
        self.labels = dict.fromkeys(self.images, None)

        self.setUp()

    def setUp(self):
        for im_name in self.labels:
            name = im_name[:-4]
            label_filename = os.path.join(self.data_dir, "labels", name + ".txt")
            with open(label_filename, "r") as file:
                content = set(
                    [int(line.strip().split()[0]) for line in file.readlines()]
                )

            labels = [0, 0, 0, 0, 0]
            for cls in content:
                labels[cls] = 1

            self.labels[im_name] = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.im_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.labels[img_name]

        return image, torch.FloatTensor(label)


def main():
    dataset = TrafficDataset("traffic-detection-dataset/test")
    print(dataset[0])


if __name__ == "__main__":
    main()
