from torch import nn
import torch
import torch.nn.functional as F
from oarsigrading.training.model_zoo import SeResNet
from typing import Tuple


class ClassificationHead(nn.Module):
    """Size invariant classifier module with weighted
    superpixel pooling
    """
    def __init__(self, n_features, n_cls, use_bnorm=True, drop=0.5):
        super(ClassificationHead, self).__init__()

        clf_layers = []
        #if use_bnorm:
        #    clf_layers.append(nn.BatchNorm1d(n_features))

        if drop > 0:
            clf_layers.append(nn.Dropout(drop))

        #clf_layers.append(nn.Linear(n_features*2, 512))
        #clf_layers.append(nn.BatchNorm1d(512))
        #clf_layers.append(nn.ReLU(True))
        #if drop > 0:
        #    clf_layers.append(nn.Dropout(drop))
        clf_layers.append(nn.Linear(n_features, n_cls))

        self.classifier = nn.Sequential(*clf_layers)

    def forward(self, o):
        avgp = F.adaptive_avg_pool2d(o, 1).view(o.size(0), -1)
        feats = avgp
        #mxp = F.adaptive_max_pool2d(o, 1).view(o.size(0), -1)
        #feats = torch.cat([avgp, mxp], 1)
        clf_result = self.classifier(feats)
        return clf_result


class MultiTaskHead(nn.Module):
    def __init__(self, n_feats, n_tasks, n_cls: int or Tuple[int], clf_bnorm, dropout):
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
                                                                                                 dropout)

    def forward(self, features):
        res = []
        for j in range(sum(self.n_tasks)):
            res.append(self.__dict__['_modules'][f'head_{j}'](features))
        return res


class OARSIGradingNet(nn.Module):
    def __init__(self, bb_width=50, dropout=0.5, cls_bnorm=False):
        super(OARSIGradingNet, self).__init__()
        backbone = SeResNet(bb_width, 0, 1)
        self.encoder = backbone.encoder[:-1]
        n_feats = backbone.classifier[-1].in_features
        #self.classifier = MultiTaskHead(n_feats, (1, 6), (5, 4), cls_bnorm, dropout)
        clf_layers = []
        if dropout > 0:
            clf_layers.append(nn.Dropout(dropout))

        clf_layers.append(nn.Linear(n_feats, 5))

        self.classifier = nn.Sequential(*clf_layers)

    def forward(self, x):
        features = self.encoder(x)
        features = F.adaptive_avg_pool2d(features, 1).view(x.size(0), -1)
        return self.classifier(features)


class MultiTaskClassificationLoss(nn.Module):
    def __init__(self):
        super(MultiTaskClassificationLoss, self).__init__()
        #self.cls_loss = nn.CrossEntropyLoss()

    def forward(self, pred, target_cls):
        #loss = 0
        #n_tasks = 1#len(pred)

        #for task_id in range(n_tasks):
            #loss += self.cls_loss(pred[task_id], target_cls[:, task_id])

        #loss /= n_tasks

        return F.cross_entropy(pred, target_cls[:, 0])
