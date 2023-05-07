import hydra
import torch
import os
from .dataset import MNISTUCCTestDataset
from .model import UCCModel
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from .visualize import visualize_distributions


def obtain_feature_dist_by_class(model, features, labels):
    feature_dist_by_class = {}
    for feature, label in zip(features, labels):
        label = label.item()
        if label not in feature_dist_by_class:
            feature_dist_by_class[label] = []
        feature_dist_by_class[label].append(feature)
    # convert to tensor
    for label in feature_dist_by_class:
        feature_dist_by_class[label] = torch.stack(
            feature_dist_by_class[label], dim=0
        ).unsqueeze(0)
    # compute distribution
    for label in feature_dist_by_class:
        feature_dist_by_class[label] = model.kde(
            feature_dist_by_class[label], model.num_bins, model.sigma
        )
    return feature_dist_by_class


def extract_features(model, test_loader, device):
    features = []
    for sample, label in tqdm(test_loader):
        sample = sample.to(device).view(1, -1, 1, 28, 28)
        label = label.to(device)
        # forward
        with torch.no_grad():
            feature = model.feature_extractor(sample)
        feature = feature.squeeze(0)
        features.append(feature.cpu())
    return torch.cat(features, dim=0)


def init_dataloader(args):
    test_dataset = MNISTUCCTestDataset()
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    return test_loader


def init_model(model_cfg, checkpoint_path, device):
    model = UCCModel(model_cfg)
    # load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    state_dict = ckpt["model_state_dict"]
    msg = model.load_state_dict(state_dict)
    print("load ckpt msg: ", msg)
    model.to(device)
    return model


@hydra.main(version_base=None, config_path="../configs", config_name="test.yaml")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print("device:", device)
    args = cfg.args
    print("args: ", args)

    # init model
    model = init_model(cfg.model, args.model_path, device)
    test_loader = init_dataloader(args)
    # extract features
    features = extract_features(model, test_loader, device)
    # obtain feature distribution by class
    feature_dist_by_class = obtain_feature_dist_by_class(
        model, features, test_loader.dataset.y_test
    )
    # visualize
    num_classes = len(feature_dist_by_class)
    feature_distribution_arr = np.zeros(
        (num_classes, feature_dist_by_class[0].shape[1])
    )
    for c in feature_dist_by_class:
        feature_distribution_arr[c] = feature_dist_by_class[c].cpu().numpy()
    classes = ["digit" + str(i) for i in range(num_classes)]
    visualize_distributions(feature_distribution_arr, classes, model.num_bins)


if __name__ == "__main__":
    main()
