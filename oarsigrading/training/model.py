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
    def __init__(self, n_features, n_cls, use_bnorm=True, drop=0.5, use_gwap=False, use_gwap_hidden=False):
        super(ClassificationHead, self).__init__()

        clf_layers = []
        if use_bnorm:
            clf_layers.append(nn.BatchNorm1d(n_features))

        if drop > 0:
            clf_layers.append(nn.Dropout(drop))

        clf_layers.append(nn.Linear(n_features, n_cls))

        self.classifier = nn.Sequential(*clf_layers)

        if use_gwap:
            self.gwap = GlobalWeightedAveragePooling(n_features, use_hidden=use_gwap_hidden)

    def forward(self, o):
        if not hasattr(self, 'gwap'):
            avgp = F.adaptive_avg_pool2d(o, 1).view(o.size(0), -1)
        else:
            avgp = self.gwap(o).view(o.size(0), -1)

        clf_result = self.classifier(avgp)
        return clf_result


class MultiTaskHead(nn.Module):
    def __init__(self, n_feats, n_tasks, n_cls: int or Tuple[int], clf_bnorm, dropout, use_gwap=False,
                 use_gwap_hidden=False):

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
                                                                                                 use_gwap_hidden)

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
        self.classifier = MultiTaskHead(n_feats, (1, 6), (5, 4), cls_bnorm, dropout, use_gwap, use_gwap_hidden)
        clf_layers = []
        if dropout > 0:
            clf_layers.append(nn.Dropout(dropout))

        clf_layers.append(nn.Linear(n_feats, 5))

    def forward(self, x):
        features = self.encoder(x)
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
