import numpy as np
import os
from itertools import combinations
import torch
from torch.utils.data import Dataset


def find_root(start_dir=os.curdir, target_name="setup.py"):
    """
    Finds the root directory of a Python project.

    Args:
        start_dir (str): The directory to start the search from (default: current directory).
        target_name (str): The name of the file or directory to search for (default: "setup.py").

    Returns:
        str: The path of the root directory.

    Raises:
        FileNotFoundError: If the target file or directory cannot be found.
    """
    # Make sure start_dir is an absolute path
    start_dir = os.path.abspath(start_dir)

    # Walk up the directory tree until we find the target file or directory
    while True:
        target_path = os.path.join(start_dir, target_name)
        if os.path.exists(target_path):
            return start_dir
        else:
            # If we've reached the root directory, raise an exception
            if start_dir == os.path.dirname(start_dir):
                raise FileNotFoundError(f"Could not find {target_name}")
            # Otherwise, move up one directory and try again
            start_dir = os.path.dirname(start_dir)


class MNISTUCCDataset(Dataset):
    """MNIST dataset for Unique Class Count(UCC)"""

    def __init__(
        self, mode="train", num_instances=32, ucc_start=1, ucc_end=10, dataset_len=1000
    ):
        assert mode in ["train", "val"], "Mode should be either 'train' or 'val'"

        self.mode = mode
        self.num_instances = num_instances
        self.ucc_start = ucc_start
        self.ucc_end = ucc_end
        self.dataset_len = dataset_len

        self.num_digits = 10
        self.num_classes = self.ucc_end - self.ucc_start + 1

        root = find_root(target_name=".gitignore")
        splitted_dataset = np.load(
            os.path.join(root, "data", "mnist", "splitted_mnist_dataset.npz")
        )

        x_train = splitted_dataset["x_train"]
        y_train = splitted_dataset["y_train"]
        x_val = splitted_dataset["x_val"]
        y_val = splitted_dataset["y_val"]

        # preprocess
        x_train = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1) / 255
        x_train = (x_train - x_train.mean(dim=(2, 3), keepdim=True)) / x_train.std(
            dim=(2, 3), keepdim=True
        )
        self.x_train = x_train
        self.y_train = torch.tensor(y_train, dtype=torch.int64)

        x_val = torch.tensor(x_val, dtype=torch.float32).unsqueeze(1) / 255
        x_val = (x_val - x_val.mean(dim=(2, 3), keepdim=True)) / x_val.std(
            dim=(2, 3), keepdim=True
        )
        self.x_val = x_val
        self.y_val = torch.tensor(y_val, dtype=torch.int64)

        self.digit_dict = self.get_digit_dict()
        self.class_dict = self.get_class_dict()

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        class_label = index % self.num_classes
        class_key = f"class_{class_label}"

        # generate a random index
        ind = np.random.randint(0, self.class_dict[class_key].shape[0])
        temp_elements = self.class_dict[class_key][ind, :]

        num_elements = temp_elements.shape[0]
        num_instances_per_element = self.num_instances // num_elements
        remainder_size = self.num_instances % num_elements
        num_instances_arr = np.repeat(num_instances_per_element, num_elements)
        num_instances_arr[:remainder_size] += 1

        indices_list = []
        for k in range(num_elements):
            digit_key = f"digit{temp_elements[k]}"
            num_instances = num_instances_arr[k]
            train_size = len(self.digit_dict[digit_key]["train_indices"])
            val_size = len(self.digit_dict[digit_key]["val_indices"])
            # generate random indices
            random_indices = np.random.randint(
                0, train_size if self.mode == "train" else val_size, size=num_instances
            )

            if self.mode == "train":
                indices_list += list(
                    self.digit_dict[digit_key]["train_indices"][random_indices]
                )
            else:
                indices_list += list(
                    self.digit_dict[digit_key]["val_indices"][random_indices]
                )

        if self.mode == "train":
            samples = self.x_train[indices_list]
        else:
            samples = self.x_val[indices_list]

        samples = samples.view(
            self.num_instances, 1, samples.shape[2], samples.shape[3]
        )

        return samples, class_label

    def get_digit_dict(self):
        digit_dict = dict()
        for digit_value in range(self.num_digits):
            digit_key = "digit" + str(digit_value)

            temp_digit_dict = dict()

            temp_digit_dict["value"] = digit_value
            temp_digit_dict["train_indices"] = np.where(self.y_train == digit_value)[0]
            temp_digit_dict["num_train"] = len(temp_digit_dict["train_indices"])
            temp_digit_dict["val_indices"] = np.where(self.y_val == digit_value)[0]
            temp_digit_dict["num_val"] = len(temp_digit_dict["val_indices"])

            digit_dict[digit_key] = temp_digit_dict

        return digit_dict

    def get_class_dict(self):
        elements_arr = np.arange(self.num_digits)
        class_dict = dict()
        for i in range(self.num_classes):
            class_key = "class_" + str(i)

            elements_list = list()
            for j in combinations(elements_arr, i + self.ucc_start):
                elements_list.append(np.array(j))

            class_dict[class_key] = np.array(elements_list)

        return class_dict


class MNISTUCCTestDataset(Dataset):
    """MNIST Test dataset for Unique Class Count(UCC)"""

    def __init__(
        self,
    ):
        root = find_root(target_name=".gitignore")
        splitted_dataset = np.load(
            os.path.join(root, "data", "mnist", "splitted_mnist_dataset.npz")
        )

        x_test = splitted_dataset["x_test"]
        y_test = splitted_dataset["y_test"]

        # preprocess
        x_test = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1) / 255
        x_test = (x_test - x_test.mean(dim=(2, 3), keepdim=True)) / x_test.std(
            dim=(2, 3), keepdim=True
        )
        self.x_test = x_test
        self.y_test = torch.tensor(y_test, dtype=torch.int64)

    def __len__(self):
        return len(self.y_test)

    def __getitem__(self, index):
        sample = self.x_test[index]
        label = self.y_test[index]
        return sample, label


class CamelyonUCCDataset(Dataset):
    def __init__(
        self,
        mode="train",
        num_instances=10,
        data_augment=False,
        patch_size=None,
        dataset_len=1000,
    ):
        self.mode = mode
        self.num_instances = num_instances
        self.data_augment = data_augment
        self.patch_size = patch_size
        self.dataset_len = dataset_len

        # load data
        root = find_root(target_name=".gitignore")
        dataset_dir = os.path.join(root, "data", "camelyon")
        # shape: (num_samples, 512, 512, 3)
        self.x = np.load("{}/{}_img_data.npy".format(dataset_dir, mode))
        self.y = np.load("{}/{}_ucc_label.npy".format(dataset_dir, mode))
        self.y_mask = np.load("{}/{}_mask_data.npy".format(dataset_dir, mode))

        self.image_size = self.x.shape[1]
        self.range_high_lim = self.image_size - self.patch_size + 1

        self.num_samples = self.y.shape[0]

        self.num_classes = self.y.shape[1]

    def __len__(self):
        return self.dataset_len if self.mode != "test" else self.num_samples

    def __getitem__(self, index):
        index = index % self.num_samples
        # shape: (num_instances, patch_size, patch_size, 3)
        sample_data = self.get_sample_data(index)
        # normalize height and width dimension
        sample_data = (
            sample_data
            - np.mean(sample_data, axis=(1, 2))[:, np.newaxis, np.newaxis, :]
        ) / np.std(sample_data, axis=(1, 2))[:, np.newaxis, np.newaxis, :]
        sample_data = np.asarray(sample_data, dtype=np.float32)
        sample_data = torch.from_numpy(sample_data).permute(0, 3, 1, 2)
        label = torch.from_numpy(self.y[index]).long()
        # convert label to class index
        label = torch.argmax(label)
        return sample_data, label

    def augment_image(self, image, augment_ind):
        if augment_ind == 0:
            return image
        elif augment_ind == 1:
            return np.rot90(image)
        elif augment_ind == 2:
            return np.rot90(image, 2)
        elif augment_ind == 3:
            return np.rot90(image, 3)
        elif augment_ind == 4:
            return np.fliplr(image)
        elif augment_ind == 5:
            return np.rot90(np.fliplr(image))
        elif augment_ind == 6:
            return np.rot90(np.fliplr(image), 2)
        elif augment_ind == 7:
            return np.rot90(np.fliplr(image), 3)

    def get_sample_data(self, sample_ind):
        image_arr = self.x[sample_ind]

        instance_list = list()
        for _ in range(self.num_instances):
            r, c = np.random.randint(low=0, high=self.range_high_lim, size=2)

            patch_data = image_arr[r : r + self.patch_size, c : c + self.patch_size, :]

            if self.data_augment:
                augment_id = np.random.randint(8)
                patch_data = self.augment_image(patch_data, augment_id)

            instance_list.append(patch_data)

        instance_arr = np.array(instance_list)

        return instance_arr

    def get_image_patches(self, sample_ind):
        # get all patches from whole image
        image_arr = self.x[sample_ind]
        image_size = image_arr.shape[0]
        patch_list = list()
        for r in range(0, image_size, self.patch_size):
            for c in range(0, image_size, self.patch_size):
                patch_list.append(
                    image_arr[r : r + self.patch_size, c : c + self.patch_size, :]
                )
        patches = np.array(patch_list, dtype=np.float32)
        patches = (
            patches - np.mean(patches, axis=(1, 2))[:, np.newaxis, np.newaxis, :]
        ) / np.std(patches, axis=(1, 2))[:, np.newaxis, np.newaxis, :]
        patches = torch.from_numpy(patches)
        # reshape patches to (num_patches, num_channels, patch_size, patch_size)
        patches = patches.permute(0, 3, 1, 2)
        return patches

    def get_image_patch_labels(self, sample_ind):
        # must convert pixel value to 0-1 not 0-255
        image_mask = self.y_mask[sample_ind] / 255
        image_size = image_mask.shape[0]
        patch_truth_list = list()
        for r in range(0, image_size, self.patch_size):
            for c in range(0, image_size, self.patch_size):
                mask_patch = image_mask[
                    r : r + self.patch_size, c : c + self.patch_size
                ]
                metastasis_ratio = np.sum(mask_patch) / (
                    self.patch_size * self.patch_size
                )

                if metastasis_ratio > 0.5:
                    patch_truth_list.append(1)
                else:
                    patch_truth_list.append(0)
        return np.asarray(patch_truth_list)
