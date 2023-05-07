import hydra
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

from model import UCCModel
from dataset import MNISTUCCDataset, CamelyonUCCDataset
from omegaconf import DictConfig


# set random seed
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def init_model_and_optimizer(args, model_cfg, device):
    model = UCCModel(model_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    return model, optimizer


def init_dataloader(args):
    assert args.dataset in [
        "mnist",
        "camelyon",
    ], "Mode should be either mnist or camelyon"
    if args.dataset == "mnist":
        train_dataset_len = args.train_num_steps * args.batch_size
        train_dataset = MNISTUCCDataset(
            mode="train",
            num_instances=args.num_instances,
            ucc_start=args.ucc_start,
            ucc_end=args.ucc_end,
            dataset_len=train_dataset_len,
        )
        val_dataset_len = args.val_num_steps * args.batch_size
        val_dataset = MNISTUCCDataset(
            mode="val",
            num_instances=args.num_instances,
            ucc_start=args.ucc_start,
            ucc_end=args.ucc_end,
            dataset_len=val_dataset_len,
        )
    else:
        train_dataset_len = args.train_num_steps * args.batch_size
        train_dataset = CamelyonUCCDataset(
            mode="train",
            num_instances=args.num_instances,
            data_augment=args.data_augment,
            patch_size=args.patch_size,
            dataset_len=train_dataset_len,
        )
        val_dataset_len = args.val_num_steps * args.batch_size
        val_dataset = CamelyonUCCDataset(
            mode="val",
            num_instances=args.num_instances,
            data_augment=args.data_augment,
            patch_size=args.patch_size,
            dataset_len=val_dataset_len,
        )
    # create dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    return train_loader, val_loader


def eval(model, val_loader, device):
    model.eval()
    val_loss_list = []
    val_acc_list = []
    with torch.no_grad():
        for batch_samples, batch_labels in val_loader:
            batch_samples = batch_samples.to(device)
            batch_labels = batch_labels.to(device)
            ucc_logits, _ = model(batch_samples)
            ucc_val_loss = F.cross_entropy(ucc_logits, batch_labels)
            # acculate accuracy
            _, ucc_predicts = torch.max(ucc_logits, dim=1)
            acc = torch.sum(ucc_predicts == batch_labels).item() / len(batch_labels)
            val_acc_list.append(acc)
            val_loss_list.append(ucc_val_loss.item())
    return np.mean(val_loss_list), np.mean(val_acc_list)


def train(args, model, optimizer, train_loader, val_loader, device):
    model.train()
    step = 0
    best_eval_acc = 0
    for batch_samples, batch_labels in tqdm(train_loader):
        batch_samples = batch_samples.to(device)
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad()
        loss = model(batch_samples, batch_labels)
        loss.backward()
        optimizer.step()
        step += 1

        if step % args.save_interval == 0:
            eval_loss, eval_acc = eval(model, val_loader, device)
            print(f"step: {step}, eval loss: {eval_loss}, eval acc: {eval_acc}")
            # early stop
            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc
                # save model
                save_path = os.path.join(args.model_dir, f"{args.model_name}_best.pth")
                # put eval loss and acc in model state dict
                save_dict = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "eval_loss": eval_loss,
                    "eval_acc": eval_acc,
                    "step": step,
                }
                torch.save(save_dict, save_path)
            # switch to train mode
            model.train()
    print("Training finished!!!")


@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print("device:", device)
    args = cfg.args
    print("args: ", args)
    print("model: ", cfg.model)
    # set random seed
    set_random_seed(args.seed)
    # set model save dir
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    # init model and optimizer
    model, optimizer = init_model_and_optimizer(args, cfg.model, device)
    train_loader, val_loader = init_dataloader(args)
    train(args, model, optimizer, train_loader, val_loader, device)


if __name__ == "__main__":
    main()
