import torch.nn as nn
import pretrainedmodels


class ViewerFC(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class SeResNet(nn.Module):
    def __init__(self, layers, drop, ncls):
        super(SeResNet, self).__init__()
        if layers == 18:
            model = pretrainedmodels.__dict__['resnet18'](num_classes=1000, pretrained='imagenet')
        elif layers == 34:
            model = pretrainedmodels.__dict__['resnet34'](num_classes=1000, pretrained='imagenet')
        elif layers == 50:
            model = pretrainedmodels.__dict__['se_resnet50'](num_classes=1000, pretrained='imagenet')
        elif layers == 101:
            model = pretrainedmodels.__dict__['se_resnet101'](num_classes=1000, pretrained='imagenet')
        elif layers == 152:
            model = pretrainedmodels.__dict__['se_resnet152'](num_classes=1000, pretrained='imagenet')
        else:
            raise NotImplementedError

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

