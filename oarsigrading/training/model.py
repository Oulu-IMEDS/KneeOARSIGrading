from torch import nn
import torch
import torch.nn.functional as F
from oarsigrading.training.model_zoo import SeResNet


class AttentionHead(nn.Module):
    """Size invariant classifier module with weighted
    superpixel pooling
    """
    def __init__(self, n_features, n_cls, weight_gen_hidden=128, use_bnorm=True, drop=0.5):
        super(AttentionHead, self).__init__()
        self.weight_generator = nn.Sequential(nn.BatchNorm2d(n_features),
                                              nn.Conv2d(n_features, weight_gen_hidden, kernel_size=3, padding=1),
                                              nn.ReLU(True),
                                              nn.Conv2d(weight_gen_hidden, 1, kernel_size=1))

        clf_layers = []
        if use_bnorm:
            clf_layers.append(nn.BatchNorm1d(n_features))
        if drop > 0:
            clf_layers.append(nn.Dropout(drop))

        clf_layers.append(nn.Linear(n_features, n_cls))

        self.classifier = nn.Sequential(*clf_layers)

    def forward(self, o):
        att_weights = self.weight_generator(o)
        att_weights_norm = torch.sigmoid(att_weights)
        clf_result = self.classifier(torch.sum(o*att_weights_norm, dim=[2, 3]))
        # Weights are not normalized and should go to BCE loss with logits
        return clf_result, att_weights


class MultiTaskAttentionHead(nn.Module):
    def __init__(self, n_feats, n_tasks, n_cls, att_h_size, att_bnorm, dropout):
        super(MultiTaskAttentionHead, self).__init__()
        self.n_tasks = n_tasks
        for j in range(n_tasks):
            self.__dict__['_modules'][f'head_{j}'] = AttentionHead(n_feats, n_cls, att_h_size, att_bnorm, dropout)

    def forward(self, features):
        res = []
        for j in range(self.n_tasks):
            res.append(self.__dict__['_modules'][f'head_{j}'](features))
        return res


class OARSIGradingNet(nn.Module):
    def __init__(self, bb_width=50, n_tasks=6, n_cls=4, dropout=0.5, att_bnorm=False, att_h_size=128):
        super(OARSIGradingNet, self).__init__()
        backbone = SeResNet(bb_width, 0, 1)
        self.encoder = backbone.encoder[:-1]
        n_feats = backbone.classifier[-1].in_features
        self.classifier = MultiTaskAttentionHead(n_feats, n_tasks, n_cls, att_h_size, att_bnorm, dropout)

    def forward(self, x):
        features = self.encoder(x)
        return self.classifier(features)


class MultiTaskAttentionLoss(nn.Module):
    def __init__(self, w_ratio=0.5):
        super(MultiTaskAttentionLoss, self).__init__()
        self.cls_loss = nn.CrossEntropyLoss()
        self.att_loss = nn.BCEWithLogitsLoss()
        self.w_ratio = w_ratio

    def forward(self, pred, target_cls, target_att):
        loss = 0
        n_tasks = len(pred)

        for task_id in range(n_tasks):
            loss += self.cls_loss(pred[task_id][0], target_cls[:, task_id]).mul(self.w_ratio)
            loss += self.att_loss(pred[task_id][1].squeeze(), target_att[:, task_id]).mul(1 - self.w_ratio)

        loss /= n_tasks

        return loss
