import torch.nn as nn
from torch import Tensor
import torch
import numpy as np
from models.text_encoder import load_clip_to_cpu, TextEncoder, PromptLearner
import torch.nn.functional as F
from utils.label_mapping import euclidean_distance

def conv3x3(in_planes, out_planes, stride = 1, groups = 1, dilation = 1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride = 1, downsample = None, norm_layer = None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNetPrompt(nn.Module):
    def __init__(self, classnames, block, layers, mode='train'):
        super().__init__()
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        self.base_width = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, len(classnames))



        '''
        Prompt Learning
        '''
        self.softmax_temp = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # Text encoder
        self.clip_model = load_clip_to_cpu()
        self.text_encoder = TextEncoder(self.clip_model)
        self.prompt_learner = PromptLearner(classnames, self.clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        # The dimensions of the hidden layer are not documented in the paper, so we use the values from the LICO repo.
        self.mlp = nn.Sequential(
            nn.Linear(512, 512),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512, 49)
        )

        self.mode = mode

    def _make_layer(self, block, planes, blocks, stride = 1,
        dilate: bool = False):
        norm_layer = self._norm_layer
        downsample = None
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes, stride), norm_layer(planes))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, targets=None, w_distance=None, mode='train'):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        feature_maps = x.view(x.shape[0], x.shape[1], -1)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        emb = x
        x = self.fc(x)

        # Calculate similarity matrix as the softmax of negative distances. Equation (1) in the paper.
        softmax_temp = self.softmax_temp
        image_distance_matrix = euclidean_distance(emb)
        AF = F.softmax(-image_distance_matrix / softmax_temp, dim=1)
        # Note: AF stands for A^F in the paper. This is the similarity matrix of the feature maps.

        prompts = self.prompt_learner()  # (num_classes, 77, 512)
        text_features = self.text_encoder(prompts, self.tokenized_prompts)  # (num_classes, prompt_length, 512)
        # Note: prompt_length here is <BOS> + n_ctx + (tokens for class: "aquarium fish" is 2 tokens) + '.' + <EOS>
        text_features = text_features[targets]  # (batch_size, prompt_length, 512)

        if mode == 'train' and self.mode == 'train':
            # prompt learning
            text_features_w = self.mlp(text_features)  # (batch_size, prompt_length, 64). This is G_i in the paper.
            feature_maps = F.normalize(feature_maps, dim=-1)  # (batch_size, num_channels, 64)
            text_features_w = F.normalize(text_features_w, dim=-1)  # (batch_size, num_context, 64)
            P, C = w_distance(feature_maps, text_features_w)
            w_loss = torch.sum(P * C, dim=(-2, -1)).mean()

            text_distance_matrix = euclidean_distance(text_features)
            AG = F.softmax(-text_distance_matrix / softmax_temp, dim=1)
            # Note: AG stands for A^G in the paper. This is the similarity matrix of the embedded prompts.
            return x, AF, w_loss, AG

        elif mode == 'test' or self.mode == 'test':
            return x