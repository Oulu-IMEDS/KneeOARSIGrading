import torch.nn as nn
import pretrainedmodels
from termcolor import colored


class ViewerFC(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def backbone_name(layers, se, dw):
    if layers == 18:
        bb_name = 'resnet18'
    elif layers == 34:
        bb_name = 'resnet34'
    elif layers == 50:
        if not se and not dw:
            bb_name = 'resnet50'
        elif se and not dw:
            bb_name = 'se_resnet50'
        else:
            bb_name = 'se_resnext50_32x4d'
    elif layers == 101:
        if not se and not dw:
            bb_name = 'resnet101'
        elif se and not dw:
            bb_name = 'se_resnet101'
        else:
            bb_name = 'se_resnext101_32x4d'
    else:
        raise NotImplementedError
    return bb_name


class ResNet(nn.Module):
    def __init__(self, se, dw, layers, drop, ncls):
        super(ResNet, self).__init__()

        bb_name = backbone_name(layers, se, dw)
        print(colored('====> ', 'green') + f'Pre-trained {bb_name} is used as backbone')
        model = pretrainedmodels.__dict__[bb_name](num_classes=1000, pretrained='imagenet')
        self.encoder = list(model.children())[:-2]

        self.encoder.append(nn.AdaptiveAvgPool2d(1))
        self.encoder = nn.Sequential(*self.encoder)

        if drop > 0:
            self.classifier = nn.Sequential(ViewerFC(),
                                            nn.Dropout(drop),
                                            nn.Linear(model.last_linear.in_features, ncls))
        else:
            self.classifier = nn.Sequential(
                ViewerFC(),
                nn.Linear(model.last_linear.in_features, ncls)
            )

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x

