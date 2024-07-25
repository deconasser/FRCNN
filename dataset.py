import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize, ToTensor, Compose
import cv2
import os
import pickle
import numpy as np

class AnimalDataset(Dataset):
    def __init__(self, root, is_Train, transform=None):
        if is_Train:
            data_path = os.path.join(root, "train")
        else:
            data_path = os.path.join(root, "test")
        self.categories = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider",
                           "squirrel"]
        self.image_paths = []
        self.labels = []
        for index, category in enumerate(self.categories):
            subdir_path = os.path.join(data_path, category)
            for file_name in os.listdir(subdir_path):
                self.image_paths.append(os.path.join(subdir_path, file_name))
                self.labels.append(index)

        self.transform = transform
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, item):
        image = cv2.imread(self.image_paths[item])
        if self.transform:
            image = self.transform(image)
        label = self.labels[item]
        return image, label

class MyCifar10(Dataset):
    def __init__(self, root, is_Train):
        if is_Train:
            data_files = [os.path.join(root, "data_batch_{}".format(i)) for i in range(1, 6)]
        else:
            data_files = [os.path.join(root, "test_batch")]
        self.all_images = []
        self.all_labels = []
        for data_file in data_files:
            with open(data_file, "rb") as fo:
                data = pickle.load(fo, encoding="bytes")
                images= data[b'data']
                labels= data[b'labels']
                self.all_images.extend(images)
                self.all_labels.extend(labels)
    def __len__(self):
        return len(self.all_labels)
    def __getitem__(self, item):
        image = self.all_images[item]
        image = np.reshape(image, (3, 32, 32)).astype(np.float32)/255.
        label = self.all_labels[item]
        return image, label


if __name__ == '__main__':
    #train_dataset = AnimalDataset(root="data/animals", is_Train=True)

    # train_dataset = MyCifar10(root="data/cifar-10", is_Train=True)
    # img, label = train_dataset.__getitem__(3)
    # print(img.shape)

    transform = Compose([
        ToTensor(),
        Resize((224, 224))
    ])
    train_dataset = AnimalDataset(root="data/animals", is_Train=True, transform=transform)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=16,
        num_workers=8,
        shuffle=True,
        drop_last=False
    )

    for epoch in range(100):
        for images, labels in train_dataloader:
            print(images.shape)