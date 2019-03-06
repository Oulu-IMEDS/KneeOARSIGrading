from torch import nn
import torch.nn.functional as F
from oarsigrading.training.model_zoo import ResNet
from typing import Tuple
import torch
from termcolor import colored


class GlobalWeightedAveragePooling(nn.Module):
    """
    "Global Weighted Average Pooling Bridges Pixel-level Localization and Image-level Classiﬁcation".

    Class-agnostic version.

    """

    def __init__(self, n_feats, use_hidden=False):
        super().__init__()
        if use_hidden:
            self.conv = nn.Sequential(
                nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm2d(n_feats),
                nn.ReLU(True),
                nn.Dropout2d(0.5),
                nn.Conv2d(n_feats, 1, kernel_size=1, bias=True)
            )
        else:
            self.conv = nn.Conv2d(n_feats, 1, kernel_size=1, bias=True)

    def fscore(self, x: torch.Tensor):
        m = self.conv(x)
        m = m.sigmoid().exp()
        return m

    def norm(self, x: torch.Tensor):
        return x / x.sum(dim=[2, 3], keepdim=True)

    def forward(self, x):
        input_x = x
        x = self.fscore(x)
        x = self.norm(x)
        x = x * input_x
        x = x.sum(dim=[2, 3])
        return x


class ClassificationHead(nn.Module):
    def __init__(self, n_features, n_cls, use_bnorm=True, drop=0.5, use_gwap=False,
                 use_gwap_hidden=False, no_pool=False):
        super(ClassificationHead, self).__init__()

        clf_layers = []
        if use_bnorm:
            clf_layers.append(nn.BatchNorm1d(n_features))

        if drop > 0:
            clf_layers.append(nn.Dropout(drop))

        clf_layers.append(nn.Linear(n_features, n_cls))

        self.classifier = nn.Sequential(*clf_layers)
        self.no_pool = no_pool

        if use_gwap:
            self.gwap = GlobalWeightedAveragePooling(n_features, use_hidden=use_gwap_hidden)

    def forward(self, o):
        if not self.no_pool:
            if not hasattr(self, 'gwap'):
                avgp = F.adaptive_avg_pool2d(o, 1).view(o.size(0), -1)
            else:
                avgp = self.gwap(o).view(o.size(0), -1)
        else:
            avgp = o
        clf_result = self.classifier(avgp)
        return clf_result


class MultiTaskHead(nn.Module):
    def __init__(self, n_feats, n_tasks, n_cls: int or Tuple[int], clf_bnorm, dropout, use_gwap=False,
                 use_gwap_hidden=False, no_pool=False):

        super(MultiTaskHead, self).__init__()

        if isinstance(n_cls, int):
            n_cls = (n_cls, )

        if isinstance(n_tasks, int):
            n_tasks = (n_tasks,)

        assert len(n_cls) == len(n_tasks)

        self.n_tasks = n_tasks
        self.n_cls = n_cls

        for task_type_idx, (n_tasks, task_n_cls) in enumerate(zip(self.n_tasks, self.n_cls)):
            for task_idx in range(n_tasks):
                self.__dict__['_modules'][f'head_{task_type_idx+task_idx}'] = ClassificationHead(n_feats,
                                                                                                 task_n_cls,
                                                                                                 clf_bnorm,
                                                                                                 dropout,
                                                                                                 use_gwap,
                                                                                                 use_gwap_hidden,
                                                                                                 no_pool)

    def forward(self, features):
        res = []
        for j in range(sum(self.n_tasks)):
            res.append(self.__dict__['_modules'][f'head_{j}'](features))
        return res


class OARSIGradingNet(nn.Module):
    def __init__(self, bb_depth=50, dropout=0.5, cls_bnorm=False, se=False, dw=False,
                 use_gwap=False, use_gwap_hidden=False):

        super(OARSIGradingNet, self).__init__()
        backbone = ResNet(se, dw, bb_depth, 0, 1)
        self.encoder = backbone.encoder[:-1]
        n_feats = backbone.classifier[-1].in_features
        if use_gwap:
            print(colored('====> ', 'green') + f'Task-specific weighted pooling will be used')
            if use_gwap_hidden:
                print(colored('====> ', 'green') + f'GWAP will have a hidden layer')
        self.classifier = MultiTaskHead(n_feats, (1, 6), (5, 4), cls_bnorm, dropout, use_gwap, use_gwap_hidden)

    def forward(self, x):
        features = self.encoder(x)
        return self.classifier(features)


def conv_block3(inp, out, stride, pad):
    """
    3x3 ConvNet building block with different activations support.

    Aleksei Tiulpin, Unversity of Oulu, 2017 (c).
    """
    return nn.Sequential(
        nn.Conv2d(inp, out, kernel_size=3, stride=stride, padding=pad),
        nn.BatchNorm2d(out, eps=1e-3),
        nn.ReLU(inplace=True)
    )


def weights_init_uniform(m):
    """
    Initializes the weights using kaiming method.
    """
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform(m.weight.data)
        m.bias.data.fill_(0)

    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform(m.weight.data)
        m.bias.data.fill_(0)


class Branch(nn.Module):
    def __init__(self, bw):
        super().__init__()
        """
        self.block1 = nn.Sequential(conv_block3(3, bw, 2, 0),
                                    conv_block3(bw, bw, 1, 0),
                                    conv_block3(bw, bw, 1, 0),
                                    )

        self.block2 = nn.Sequential(conv_block3(bw, bw * 2, 1, 0),
                                    conv_block3(bw * 2, bw * 2, 1, 0),
                                    )

        self.block3 = conv_block3(bw * 2, bw * 4, 1, 0)

        self.block4 = conv_block3(bw * 4, bw * 4, 1, 0)
        """
        self.block1 = nn.Sequential(conv_block3(3, bw, 2, 0),
                                    conv_block3(bw, bw, 1, 0),
                                    )

        self.block2 = nn.Sequential(conv_block3(bw, bw * 2, 1, 1),
                                    conv_block3(bw * 2, bw * 4, 1, 1),
                                    )

        self.block3 = nn.Sequential(conv_block3(bw * 4, bw * 4, 1, 1),
                                    conv_block3(bw * 4, bw * 4, 1, 1))

        self.block4 = nn.Sequential(conv_block3(bw * 4, bw * 8, 1, 1),
                                    conv_block3(bw * 8, bw * 8, 1, 1))

    def forward(self, x):
        o1 = F.max_pool2d(self.block1(x), 2)
        o2 = F.max_pool2d(self.block2(o1), 2)
        o3 = F.max_pool2d(self.block3(o2), 2)

        return F.adaptive_avg_pool2d(self.block4(o3), 1).view(x.size(0), -1)


class OARSIGradingNetSiamese(nn.Module):
    def __init__(self, backbone='lext', dropout=0.5):
        super(OARSIGradingNetSiamese, self).__init__()

        if backbone == 'lext':
            self.encoder = Branch(64)
            n_feats = 64*8*2
            self.classifier = MultiTaskHead(n_feats, (1, 6), (5, 4), False, dropout, False, False, True)
            self.apply(weights_init_uniform)
        elif backbone == 'resnet-18':
            backbone = ResNet(False, False, 18, 0, 1)
            self.encoder = backbone.encoder[:-1]
            n_feats = backbone.classifier[-1].in_features*2
            self.classifier = MultiTaskHead(n_feats, (1, 6), (5, 4), False, dropout, False, False, False)

    def forward(self, med, lat):
        f_med = self.encoder(med)
        f_lat = self.encoder(lat)

        features = torch.cat([f_lat, f_med], 1)
        return self.classifier(features)


class MultiTaskClassificationLoss(nn.Module):
    def __init__(self):
        super(MultiTaskClassificationLoss, self).__init__()
        self.cls_loss = nn.CrossEntropyLoss()

    def forward(self, pred, target_cls):
        loss = 0
        n_tasks = len(pred)

        for task_id in range(n_tasks):
            loss += self.cls_loss(pred[task_id], target_cls[:, task_id])

        loss /= n_tasks

        return loss
