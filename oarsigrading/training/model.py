from torch import nn
import torch
import torch.nn.functional as F
from oarsigrading.training.model_zoo import SeResNet


class ClassificationHead(nn.Module):
    """Size invariant classifier module with weighted
    superpixel pooling
    """
    def __init__(self, n_features, n_cls, use_bnorm=True, drop=0.5):
        super(ClassificationHead, self).__init__()

        clf_layers = []
        if use_bnorm:
            clf_layers.append(nn.BatchNorm1d(n_features))
        if drop > 0:
            clf_layers.append(nn.Dropout(drop))

        clf_layers.append(nn.Linear(n_features, n_cls))

        self.classifier = nn.Sequential(*clf_layers)

    def forward(self, o):
        clf_result = self.classifier(F.adaptive_avg_pool2d(o, 1).view(o.size(0), -1))
        return clf_result


class MultiTaskHead(nn.Module):
    def __init__(self, n_feats, n_tasks, n_cls, clf_bnorm, dropout):
        super(MultiTaskHead, self).__init__()
        self.n_tasks = n_tasks
        for j in range(n_tasks):
            self.__dict__['_modules'][f'head_{j}'] = ClassificationHead(n_feats, n_cls, clf_bnorm, dropout)

    def forward(self, features):
        res = []
        for j in range(self.n_tasks):
            res.append(self.__dict__['_modules'][f'head_{j}'](features))
        return res


class OARSIGradingNet(nn.Module):
    def __init__(self, bb_width=50, n_tasks=6, n_cls=4, dropout=0.5, cls_bnorm=False):
        super(OARSIGradingNet, self).__init__()
        backbone = SeResNet(bb_width, 0, 1)
        self.encoder = backbone.encoder[:-1]
        n_feats = backbone.classifier[-1].in_features
        self.classifier = MultiTaskHead(n_feats, n_tasks, n_cls, cls_bnorm, dropout)

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
