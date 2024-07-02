from voc_dataset import VOCDataset
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm.autonotebook import tqdm
import argparse
import os
from torch.utils.tensorboard import SummaryWriter
import numpy as np
def collate_fn(batch):
    images, labels = zip(*batch)
    return list(images), list(labels)
def get_args():
    parser = argparse.ArgumentParser(description="Train faster rcnn model")
    parser.add_argument("--data_path", "-d", type=str, default="my_voc_dataset", help="Path to dataset")
    parser.add_argument("--year", "-y", type=str, default="2012")
    parser.add_argument("--num_epochs", "-n", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", "-b", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", "-l", type=float, default=1e-3, help="Learning rate for optimizer")
    parser.add_argument("--momentum", "-m", type=float, default=0.9, help="Momentum for optimizer")
    parser.add_argument("--log_folder", "-p", type=str, default="tensorboard", help="Path to generated tensorboard")
    parser.add_argument("--checkpoint_folder", "-c", type=str, default="trained_models",
                        help="Path to save checkpoint")
    parser.add_argument("--saved_checkpoint", "-o", type=str, default=None, help="Continue from this checkpoint")
    args = parser.parse_args()
    return args

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    num_epochs = args.num_epochs
    batch_size = args.batch_size

    transform = ToTensor()
    train_dataset = VOCDataset(root=args.data_path, year="2012", image_set="train", download=False, transform=transform)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=2,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_dataset = VOCDataset(root=args.data_path, year="2012", image_set="val", download=False, transform=transform)
    val_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=2,
        shuffle=True,
        collate_fn=collate_fn
    )

    model = fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT)
    in_channels = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_channels, num_classes=len(train_dataset.categories))
    model.to(device)

    optimizer = torch.optim.SGD(params=model.parameters(), lr=args.learning_rate, momentum=args.momentum)
    if not os.path.isdir(args.log_folder):
        os.makedirs(args.log_folder)

    writer = SummaryWriter(args.log_folder)


    for epoch in range(num_epochs):
        # TRAINING PHASE
        model.train()
        progress_bar = tqdm(train_dataloader, colour="cyan")
        train_loss = []
        for iter, (images, labels) in enumerate(progress_bar):
            images = [image.to(device) for image in images]
            labels = [{"boxes": target["boxes"].to(device), "labels": target["labels"].to(device)} for target in labels]

            # Forward
            losses = model(images, labels)
            final_losses = sum([loss for loss in losses.values()])

            # Backward
            optimizer.zero_grad()
            final_losses.backward()
            optimizer.step()
            train_loss.append(final_losses)
            mean_loss = np.mean(train_loss)
            progress_bar.set_description("Epoch {}/{}. Loss {:0.4f}".format(epoch+1, num_epochs, mean_loss))
            writer.add_scalar("Train/Loss", mean_loss, epoch*len(train_dataloader) + iter)

        #VALIDATION PHASE
        model.eval()
        progress_bar = tqdm(val_dataloader, colour="cyan")
        val_loss = []
        for iter, (images, labels) in enumerate(progress_bar):
            images = [image.to(device) for image in images]
            with torch.no_grad():
                outputs = model(images)


if __name__ == '__main__':
    train(get_args())
