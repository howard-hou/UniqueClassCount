import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualZeroPaddingBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        first_block=False,
        down_sample=False,
        up_sample=False,
    ):
        super(ResidualZeroPaddingBlock, self).__init__()
        self.first_block = first_block
        self.down_sample = down_sample
        self.up_sample = up_sample

        if self.up_sample:
            self.upsampling = nn.Upsample(scale_factor=2, mode="nearest")

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            stride=2 if self.down_sample else 1,
        )
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=2 if self.down_sample else 1,
        )

        # Initialize the weights and biases
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.constant_(self.conv1.bias, 0.1)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.constant_(self.conv2.bias, 0.1)
        nn.init.xavier_uniform_(self.skip_conv.weight)

    def forward(self, x):
        if self.first_block:
            x = F.relu(x)
            if self.up_sample:
                x = self.upsampling(x)
            out = F.relu(self.conv1(x))
            out = self.conv2(out)
            if x.shape != out.shape:
                x = self.skip_conv(x)
        else:
            out = F.relu(self.conv1(x))
            out = F.relu(self.conv2(out))

        return x + out


class WideResidualBlocks(nn.Module):
    def __init__(
        self, in_channels, out_channels, n, down_sample=False, up_sample=False
    ):
        super(WideResidualBlocks, self).__init__()
        self.blocks = nn.Sequential(
            *[
                ResidualZeroPaddingBlock(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    first_block=(i == 0),
                    down_sample=down_sample,
                    up_sample=up_sample,
                )
                for i in range(n)
            ]
        )

    def forward(self, x):
        return self.blocks(x)


class Reshape(nn.Module):
    def __init__(self, *target_shape):
        super(Reshape, self).__init__()
        self.target_shape = target_shape

    def forward(self, x):
        return x.view(x.size(0), *self.target_shape)


class UCCModel(nn.Module):
    def __init__(self, model_cfg):
        super(UCCModel, self).__init__()
        self.num_bins = model_cfg.kde_model.num_bins
        self.sigma = model_cfg.kde_model.sigma
        self.patch_model = nn.Sequential(
            nn.Conv2d(
                model_cfg.patch_model.conv_input_channel,
                model_cfg.patch_model.conv_output_channel,
                kernel_size=3,
                padding=1,
            ),
            WideResidualBlocks(
                model_cfg.patch_model.conv_output_channel,
                model_cfg.patch_model.block1_output_channel,
                model_cfg.patch_model.block1_num_layer,
            ),
            WideResidualBlocks(
                model_cfg.patch_model.block1_output_channel,
                model_cfg.patch_model.block2_output_channel,
                model_cfg.patch_model.block2_num_layer,
                down_sample=True,
            ),
            WideResidualBlocks(
                model_cfg.patch_model.block2_output_channel,
                model_cfg.patch_model.block3_output_channel,
                model_cfg.patch_model.block3_num_layer,
                down_sample=True,
            ),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(
                model_cfg.patch_model.flatten_size,
                model_cfg.patch_model.num_features,
                bias=False,
            ),
            nn.Sigmoid(),
        )

        self.image_generation_model = nn.Sequential(
            nn.Linear(
                model_cfg.patch_model.num_features,
                model_cfg.image_generation_model.linear_size,
            ),
            Reshape(*model_cfg.image_generation_model.reshape_size),
            WideResidualBlocks(
                model_cfg.image_generation_model.block1_output_channel,
                model_cfg.image_generation_model.block1_output_channel,
                model_cfg.image_generation_model.block1_num_layer,
                up_sample=True,
            ),
            WideResidualBlocks(
                model_cfg.image_generation_model.block1_output_channel,
                model_cfg.image_generation_model.block2_output_channel,
                model_cfg.image_generation_model.block2_num_layer,
                up_sample=True,
            ),
            WideResidualBlocks(
                model_cfg.image_generation_model.block2_output_channel,
                model_cfg.image_generation_model.block3_output_channel,
                model_cfg.image_generation_model.block3_num_layer,
            ),
            nn.ReLU(),
            nn.Conv2d(
                model_cfg.image_generation_model.block3_output_channel,
                model_cfg.image_generation_model.output_channel,
                kernel_size=3,
                padding=1,
            ),
        )

        self.autoencoder_model = nn.Sequential(
            self.patch_model, self.image_generation_model
        )

        fc1_input_size = model_cfg.patch_model.num_features * self.num_bins
        # Define the classification layers
        # add dropout layer
        self.dropout = nn.Dropout(p=model_cfg.classification_model.dropout_rate)
        self.fc_relu1 = nn.Linear(
            fc1_input_size,
            model_cfg.classification_model.fc1_output_size,
        )
        self.fc_relu2 = nn.Linear(
            model_cfg.classification_model.fc1_output_size,
            model_cfg.classification_model.fc2_output_size,
        )
        self.fc_softmax = nn.Linear(
            model_cfg.classification_model.fc2_output_size,
            model_cfg.classification_model.num_classes,
        )

        # Initialize the weights and biases
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x, label=None):
        # x shape: (batch_size, num_instances, num_channel, patch_size, patch_size)
        batch_size, num_instances, num_channel, _, _ = x.shape
        # reshape x to (batch_size * num_instances, 1, patch_size, patch_size)
        x = x.view(-1, num_channel, x.shape[-2], x.shape[-1])
        feature = self.patch_model(x)
        ae_output = self.image_generation_model(feature)
        # reshape output to (batch_size, num_instances, num_features)
        feature = feature.view(batch_size, num_instances, feature.shape[-1])
        # use kernel density estimation to estimate the distribution of the features
        # output of kde is concatenated features distribution
        feature_distribution = self.kde(feature, self.num_bins, self.sigma)

        out = self.mlp(feature_distribution)
        if label is not None:
            ucc_loss = F.cross_entropy(out, label)
            # autoencoder loss
            ae_loss = F.mse_loss(ae_output, x)
            return 0.5 * ucc_loss + 0.5 * ae_loss

        return out, ae_output

    def feature_extractor(self, x):
        batch_size, num_instances, _, _, _ = x.shape
        x = x.view(-1, 1, x.shape[-2], x.shape[-1])
        feature = self.patch_model(x)
        feature = feature.view(batch_size, num_instances, feature.shape[-1])
        return feature

    def feature_distribution(self, x):
        feature = self.feature_extractor(x)
        feature_distribution = self.kde(feature, self.num_bins, self.sigma)
        return feature_distribution

    def mlp(self, y):
        y1 = F.relu(self.fc_relu1(self.dropout(y)))
        y1 = F.relu(self.fc_relu2(self.dropout(y1)))
        y1 = self.dropout(y1)
        out = F.softmax(self.fc_softmax(y1), dim=1)
        return out

    def kde(self, data, num_nodes, sigma):
        device = data.device
        # data shape: (batch_size, num_instances, num_features)
        batch_size, num_instances, num_features = data.shape

        # Create sample points
        k_sample_points = (
            torch.linspace(0, 1, steps=num_nodes)
            .repeat(batch_size, num_instances, 1)
            .to(device)
        )

        # Calculate constants
        k_alpha = 1 / np.sqrt(2 * np.pi * sigma**2)
        k_beta = -1 / (2 * sigma**2)

        # Iterate over features and calculate kernel density estimation for each feature
        out_list = []
        for i in range(num_features):
            one_feature = data[:, :, i : i + 1].repeat(1, 1, num_nodes)
            k_diff_2 = (k_sample_points - one_feature) ** 2
            k_result = k_alpha * torch.exp(k_beta * k_diff_2)
            k_out_unnormalized = k_result.sum(dim=1)
            k_norm_coeff = k_out_unnormalized.sum(dim=1).view(-1, 1)
            k_out = k_out_unnormalized / k_norm_coeff.repeat(
                1, k_out_unnormalized.size(1)
            )
            out_list.append(k_out)

        # Concatenate the results
        concat_out = torch.cat(out_list, dim=-1)
        return concat_out

    def generate_image_from_feature(self, feature):
        # feature shape: (batch_size, num_features)
        if len(feature.shape) == 3:
            # feature shape: (batch_size, num_instances, num_features)
            batch_size, num_instances, num_features = feature.shape
            feature = feature.view(-1, num_features)
        # generate image
        generated_image = self.image_generation_model(feature)
        if len(feature.shape) == 3:
            # reshape image to (batch_size, num_instances, 1, patch_size, patch_size)
            generated_image = generated_image.view(
                batch_size,
                num_instances,
                1,
                generated_image.shape[-2],
                generated_image.shape[-1],
            )
        return generated_image
