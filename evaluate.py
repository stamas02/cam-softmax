import torch
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from src.image_data import ImageData
from src.utils import run_10_fold_test
from src.dataset import get_lfw


def parseargs():
    parser = ArgumentParser(description='Train the model')

    parser.add_argument("--dataset", type=str,
                        help="String - path to the training set folder or file")
    parser.add_argument("--pair_file", type=str,
                        help="String - LFW style pair file")
    parser.add_argument("--model_file", type=str,
                        help="String - Path to the trained model file")
    parser.add_argument("--log", type=str,
                        default="log/",
                        help="String - path to the aligned dataset to be saved")
    parser.add_argument("--batch_size", type=str,
                        default=12,
                        help="Integer - batch size")
    parser.add_argument("--num_workers", type=str,
                        default=1,
                        help="Integer - Number of data loader workers. ")
    parser.add_argument("--feature_dim", type=str,
                        default=256,
                        help="Integer - Dimensionality of the feature space. ")

    return parser.parse_args()
    pass


def evaluate(dataset, pair_file, model_file, log, batch_size, num_workers, feature_dim):
    """
    Train the network
    """

    files, labels = get_lfw(pair_file, dataset)

    # Use GPU when available outhervise use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Mean and Standard Deviation for normalising the dataset.
    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    # Declares the list of standard transforms for the input image.
    augmentations = transforms.Compose([transforms.Resize((112, 96)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=mean, std=std)])

    # Create a pytorch dataset loader.
    train_dataset = ImageData(files, transform=augmentations, force_rgb=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Create the model.
    model = torch.load(model_file).to(device)
    model.eval()
    # Iterate through the training dataset
    p_bar = tqdm(train_loader, total=len(train_loader), desc=f"Extracting features")
    features = []
    cnt = 0
    for imgs in p_bar:
        # Moves the images and labels to the selected device.
        imgs = imgs.to(device)

        # Performs forward propagation with the model.
        features.append(model(imgs, feature=True).detach().cpu().numpy())

        cnt += 1
    # if cnt == 10: break

    # Concatenate features from batches.
    features = np.concatenate(features, axis=0)
    # run 10-fold cross validation
    acc, TPR, TNR, FPR, FNR = run_10_fold_test(featuresL=features[0::2],
                                               featuresR=features[1::2],
                                               labels=labels[0::2])
    print("Results:-----------------------------\n"
          f"Acc:\t {np.mean(acc)}+-{np.std(acc)}\n"
          f"TPR:\t {np.mean(TPR)}+-{np.std(TPR)}\n"
          f"TNR:\t {np.mean(TNR)}+-{np.std(TNR)}\n"
          f"FPR:\t {np.mean(FPR)}+-{np.std(FPR)}\n"
          f"FNR:\t {np.mean(FNR)}+-{np.std(FNR)}\n"
          "-------------------------------------\n")


if __name__ == "__main__":
    args = parseargs()
evaluate(**args.__dict__)
