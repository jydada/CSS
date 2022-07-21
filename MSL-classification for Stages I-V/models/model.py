import torchvision
import torch.nn.functional as F 
from torch import nn
from config import config
import pretrainedmodels as pm
from resnet import resnet50
def generate_model():
    class DenseModel(nn.Module):
        def __init__(self, pretrained_model):
            super(DenseModel, self).__init__()
            self.classifier = nn.Linear(pretrained_model.classifier.in_features, config.num_classes)

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    m.bias.data.zero_()

            self.features = pretrained_model.features
            self.layer1 = pretrained_model.features._modules['denseblock1']
            self.layer2 = pretrained_model.features._modules['denseblock2']
            self.layer3 = pretrained_model.features._modules['denseblock3']
            self.layer4 = pretrained_model.features._modules['denseblock4']

        def forward(self, x):
            features = self.features(x)
            out = F.relu(features, inplace=True)
            out = F.avg_pool2d(out, kernel_size=8).view(features.size(0), -1)
            out = F.sigmoid(self.classifier(out))
            return out

    return DenseModel(torchvision.models.densenet169(pretrained=True))

def get_net():
    # #return MyModel(torchvision.models.resnet101(pretrained = True))
    # model = torchvision.models.resnet50(pretrained = True)
    model = pm.se_resnet50(pretrained='imagenet')
    # model = torchvision.models.resnet18(pretrained = True)
    print(model)
    in_features = model.last_linear.in_features
    model.last_linear = nn.Linear(in_features, config.num_classes, bias=True)


    #for param in model.parameters():
    #    param.requires_grad = False
    #model.avgpool = nn.AdaptiveAvgPool2d(1)
    #model.avgpool = nn.AdaptiveMaxPool2d(1)
    #model.fc = nn.Linear(2048,config.num_classes)
    return model
