import torch
from torch.utils.data import DataLoader,Dataset
from torchvision.transforms import ToTensor
from torchvision.datasets import VOCDetection
from pprint import pprint

class VOCDataset(VOCDetection):
    def __init__(self, root, year, image_set, download, transform):
        super().__init__(root, year, image_set, download, transform)
        self.categories = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                        'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                        'train', 'tvmonitor']
    def __getitem__(self, item):
        image, data = super().__getitem__(item)
        self.all_bboxes = []
        self.all_labels = []
        for obj in data["annotation"]["object"]:
            xmin = float(obj["bndbox"]["xmin"])
            ymin = float(obj["bndbox"]["ymin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymax = float(obj["bndbox"]["ymax"])
            self.all_bboxes.append([xmin, ymin, xmax, ymax])
            self.all_labels.append(self.categories.index(obj["name"]))
        self.all_bboxes = torch.FloatTensor(self.all_bboxes)
        self.all_labels = torch.LongTensor(self.all_labels)
        target = {
            "boxes": self.all_bboxes,
            "labels": self.all_labels
        }
        return image, target


if __name__ == '__main__':
    transform = ToTensor()
    dataset = VOCDataset(root="my_voc_dataset", year="2012", image_set="train", download=False, transform=transform)
    image, label = dataset[1000]
    print(image)
    print("-------------------------")
    print(label)