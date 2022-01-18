import math
import os.path

import torch
from tqdm import tqdm
from argparse import ArgumentParser
from src.sphereface20 import Net
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import SGD
from src import cam_loss

def parseargs():
    parser = ArgumentParser(description='Train the model')

    parser.add_argument("--dataset", type=str,
                        help="String - path to the training set folder or file")
    parser.add_argument("--log", type=str,
                        default="log/",
                        help="String - path to the aligned dataset to be saved")
    parser.add_argument("--batch_size", type=int,
                        default=12,
                        help="Integer - batch size")
    parser.add_argument("--num_workers", type=int,
                        default=1,
                        help="Integer - Number of data loader workers. ")
    parser.add_argument("--feature_dim", type=int,
                        default=256,
                        help="Integer - Dimensionality of the feature space. ")
    parser.add_argument("--epoch_cnt", type=int,
                        default=100,
                        help="Integer - Number of epoch to train. ")
    parser.add_argument("--lr", type=float,
                        default=0.001,
                        help="Floating Point - Learning rate. ")
    parser.add_argument("--c_start", type=float,
                        default=math.pi/2,
                        help="Floating Point - Starting value of the c parameter. ")
    parser.add_argument("--c_stop", type=float,
                        default=math.pi/4,
                        help="Floating Point - Minimum value of the c parameter ")
    parser.add_argument("--c_decay", type=float,
                        default=1e-4,
                        help="Floating Point - Decay for parameter c ")

    return parser.parse_args()
    pass


def train(dataset, log, batch_size, num_workers, feature_dim, epoch_cnt, lr, c_start, c_stop, c_decay):
    """
    Train the network
    """

    # Use GPU when available outhervise use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Mean and Standard Deviation for normalising the dataset.
    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    # Declares the list of standard transforms for the input image.
    augmentations = transforms.Compose([transforms.Resize((112, 96)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=mean, std=std)])

    # Create a pytorch dataset loader.
    train_dataset = ImageFolder(root=dataset, transform=augmentations)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Create the model.
    model = Net(d=feature_dim, classnum=len(train_dataset.classes), c_end=0).to(device)

    # Define the optimizer
    optimiser = SGD(params=model.parameters(), lr=lr)

    # Iterate through epochs
    c = c_start
    for epoch in range(epoch_cnt):
        model.train()
        # Iterate through the training dataset
        p_bar = tqdm(train_loader, total=len(train_loader), desc=f"Training epoch {epoch}")
        for imgs, labels in p_bar:
            # Moves the images and labels to the selected device.
            imgs = imgs.to(device)
            labels = labels.to(device)

            # Resets the gradients in the model.
            optimiser.zero_grad()

            # Performs forward propagation with the model.
            logits = model(imgs)
            # Calculates the loss.
            loss = cam_loss.loss(logits,labels,c=c, num_classes=len(train_dataset.classes))
            # Performs backward propagation.
            loss.backward()

            # Update the weights of the model using the optimiser.
            optimiser.step()
            p_bar.set_postfix({'loss': loss.item(), "c":c})
            c = max(c_stop, c-c_decay)

    # Save model
    os.makedirs(log, exist_ok=True)
    torch.save(model, os.path.join(log, "model.chkp"))


if __name__ == "__main__":
    args = parseargs()
train(**args.__dict__)
