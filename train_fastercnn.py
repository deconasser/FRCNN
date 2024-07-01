from voc_dataset import VOCDataset
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights

def collate_fn(batch):
    images, labels = zip(*batch)
    return list(images), list(labels)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = ToTensor()
    train_dataset = VOCDataset("my_voc_dataset", year="2012", image_set="train", download=False, transform=transform)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=8,
        num_workers=4,
        shuffle=True,
        collate_fn=collate_fn
    )

    model = fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT).to(device)
    model.train()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-3, momentum=0.9)
    for images, labels in train_dataloader:
        images = [image.to(device) for image in images]
        labels = [{"boxes": target["boxes"].to(device), "labels": target["labels"].to(device)} for target in labels]

        # Forward
        losses = model(images, labels)
        final_losses = sum([loss for loss in losses.values()])

        # Backward
        optimizer.zero_grad()
        final_losses.backward()
        optimizer.step()

        print(final_losses.item())

if __name__ == '__main__':
    train()
