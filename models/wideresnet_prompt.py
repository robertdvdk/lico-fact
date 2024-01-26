import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.label_mapping import euclidean_distance
from models.text_encoder import load_clip_to_cpu, TextEncoder, PromptLearner

# The code in the LICO repo was probably copied from here:
# https://github.com/xternalz/WideResNet-pytorch/blob/master/wideresnet.py.
# So we use that code directly here, as there is nothing in the paper that indicates they made any changes to a
# "base" WRN implementation.

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=True)
        self.drop_rate = drop_rate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, drop_rate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes,
                                out_planes,
                                i == 0 and stride or 1, drop_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNetPrompt(nn.Module):
    def __init__(self, classnames, depth=28, widen_factor=2, drop_rate=0.0, fixed_temperature=False):
        super(WideResNetPrompt, self).__init__()
        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1,
                               padding=1, bias=True)
        # 1st block
        self.block1 = NetworkBlock(
            n, channels[0], channels[1], block, 1, drop_rate)
        # 2nd block
        self.block2 = NetworkBlock(
            n, channels[1], channels[2], block, 2, drop_rate)
        # 3rd block
        self.block3 = NetworkBlock(
            n, channels[2], channels[3], block, 2, drop_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(channels[3], momentum=0.001, eps=0.001)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(channels[3], len(classnames))
        self.channels = channels[3]


        # Weights initialization
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        #     elif isinstance(m, nn.Linear):
        #         m.bias.data.zero_()

        # This is a learned parameter, as noted just after Equation (1). We initialise it to log(1/0.07) as this is
        # the value used in the CLIP paper.
        if not fixed_temperature:
            self.softmax_temp = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        else:
            self.softmax_temp = torch.ones([]) * np.log(1 / 0.07)

        '''
        Prompt Learning
        '''

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
            nn.Linear(512, 64)
            )

    def forward(self, x, language=True, targets = None, w_distance = None, mode = 'train'):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)  
        out = self.block3(out)
        out = self.relu(self.bn1(out))  # (batch_size, channels, 8, 8)

        feature_maps = out.view(out.shape[0], out.shape[1], -1)  # (batch_size, channels, 64). This is F_i in the paper.
        out = F.adaptive_avg_pool2d(out, 1)  # (batch_size, channels, 1, 1)
        out = out.view(-1, self.channels)
        emb = out  # (batch_size, channels)
        out = self.fc(out)

        # Calculate similarity matrix as the softmax of negative distances. Equation (1) in the paper.
        softmax_temp = self.softmax_temp
        image_distance_matrix = euclidean_distance(emb)
        AF = F.softmax(-image_distance_matrix / softmax_temp, dim=1)
        # Note: AF stands for A^F in the paper. This is the similarity matrix of the feature maps.

        prompts = self.prompt_learner()  # (num_classes, 77, 512)
        text_features = self.text_encoder(prompts, self.tokenized_prompts)  # (num_classes, prompt_length, 512)
        # Note: prompt_length here is <BOS> + n_ctx + (tokens for class: "aquarium fish" is 2 tokens) + '.' + <EOS>
        text_features = text_features[targets]  # (batch_size, prompt_length, 512)

        if language and mode == 'train':
            # prompt learning
            text_features_w = self.mlp(text_features)  # (batch_size, prompt_length, 64). This is G_i in the paper.

            # This is from the LICO repo. We leave this as is (why?).
            feature_maps = F.normalize(feature_maps, dim=-1)  # (batch_size, num_channels, 64)
            text_features_w = F.normalize(text_features_w, dim=-1)  # (batch_size, num_context, 64)
            P, C = w_distance(feature_maps, text_features_w)
            w_loss = torch.sum(P * C, dim=(-2, -1)).mean()

            text_distance_matrix = euclidean_distance(text_features)
            AG = F.softmax(-text_distance_matrix / softmax_temp, dim=1)
            # Note: AG stands for A^G in the paper. This is the similarity matrix of the embedded prompts.
            return out, AF, w_loss, AG
        
        elif mode == 'test':
            text_distance_matrix = euclidean_distance(text_features)
            AG = F.softmax(-text_distance_matrix / softmax_temp, dim=1)
            return out, AF, AG
        
    def get_logits(self, x):

        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)  
        out = self.block3(out)
        out = self.relu(self.bn1(out))

        out = F.adaptive_avg_pool2d(out, 1)
        
        out = out.view(-1, self.channels)

        out = self.fc(out)

        return out

class build_WideResNet:
    def __init__(self, depth=28, widen_factor=2, drop_rate=0.0, fixed_temperature=False):
        self.depth = depth
        self.widen_factor = widen_factor
        self.drop_rate = drop_rate
        self.fixed_temperature=fixed_temperature

    def build(self, classnames):
        return WideResNetPrompt(
            depth=self.depth,
            classnames=classnames,
            widen_factor=self.widen_factor,
            drop_rate=self.drop_rate,
            fixed_temperature=self.fixed_temperature
        )


if __name__ == '__main__':
    pass
