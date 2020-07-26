from torch import nn
import timm
from typing import Tuple
import torch


class GlobalWeightedAveragePooling(nn.Module):
    """
    "Global Weighted Average Pooling Bridges Pixel-level Localization and Image-level Classiﬁcation".

    Class-agnostic version.

    """

    def __init__(self, n_feats):
        super(GlobalWeightedAveragePooling, self).__init__()
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
    def __init__(self, n_features, n_cls, use_bnorm=True, dropout=0.5, use_gwap=False):
        super(ClassificationHead, self).__init__()

        clf_layers = []
        if use_bnorm:
            clf_layers.append(nn.BatchNorm1d(n_features))

        if dropout > 0:
            clf_layers.append(nn.Dropout(dropout))

        clf_layers.append(nn.Linear(n_features, n_cls))

        self.classifier = nn.Sequential(*clf_layers)
        if use_gwap:
            self.pool = GlobalWeightedAveragePooling(n_features)
        else:
            self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, o):
        pooled = self.pool(o).view(o.size(0), -1)
        clf_result = self.classifier(pooled)
        return clf_result


class MultiTaskHead(nn.Module):
    def __init__(self, n_feats, n_tasks, n_cls: int or Tuple[int], clf_bnorm, dropout, use_gwap=False):
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
                                                                                                 use_gwap)

    def forward(self, features):
        res = []
        for j in range(sum(self.n_tasks)):
            res.append(self.__dict__['_modules'][f'head_{j}'](features))
        return res


def gen_model_parts(cfg):
    backbone = timm.create_model(cfg.model.backbone, pretrained=cfg.model.pretrained)
    feature_dim = backbone.fc.in_features
    features = nn.Sequential(
        backbone.conv1,
        backbone.bn1,
        backbone.act1,
        backbone.maxpool,
        backbone.layer1,
        backbone.layer2,
        backbone.layer3,
        backbone.layer4,
    )

    if cfg.training.no_kl:
        classifier = MultiTaskHead(feature_dim,
                                        n_tasks=(6,), n_cls=(4,),
                                        clf_bnorm=cfg.model.cls_bnorm,
                                        dropout=cfg.model.dropout,
                                        use_gwap=cfg.model.pooling == 'gwap')
    else:
        classifier = MultiTaskHead(feature_dim,
                                        n_tasks=(1, 6),
                                        n_cls=(5, 4),
                                        clf_bnorm=cfg.model.cls_bnorm,
                                        dropout=cfg.model.dropout,
                                        use_gwap=cfg.model.pooling == 'gwap')
    return features, classifier
