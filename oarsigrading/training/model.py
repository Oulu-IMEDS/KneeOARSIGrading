from torch import nn
import torch
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
            clf_layers.append(nn.BatchNorm2d(n_features))
        if drop > 0:
            clf_layers.append(nn.Dropout(drop))

        clf_layers.append(nn.Conv2d(n_features, n_cls, kernel_size=1))

        self.classifier = nn.Sequential(*clf_layers)

    def forward(self, o):
        att_weights = self.weight_generator(o)
        att_weights_norm = torch.sigmoid(att_weights)
        clf_result = self.classifier(o*att_weights_norm)
        # Weights are not normalized and should go to BCE loss with logits
        return clf_result.view(o.size(0), -1), att_weights


class OARSIGradingNet(nn.Module):
    def __init__(self, bb_width=50, n_tasks=6, n_cls=4, dropout=0.5, att_bnorm=False, att_h_size=128):
        super(OARSIGradingNet, self).__init__()
        backbone = SeResNet(bb_width, 0, 1)
        self.encoder = backbone.encoder[:-1]
        n_feats = backbone.classifier[-1].in_features
        self.n_tasks = n_tasks
        for j in range(n_tasks):
            self.__dict__['_modules'][f'head_{j}'] = AttentionHead(n_feats, n_cls, att_h_size, att_bnorm, dropout)

    def forward(self, x):
        features = self.encoder(x)
        return [self.__dict__['_modules'][f'head_{j}'](features) for j in range(self.n_tasks)]
