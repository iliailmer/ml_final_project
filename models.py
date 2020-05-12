from torch import nn

from efficientnet_pytorch import EfficientNet


class Model(nn.Module):
    """ResNet-Based Model."""

    def __init__(self, resnet_model, num_classes=4):
        """Initialize."""
        super().__init__()
        self.features = nn.Sequential(
            resnet_model.conv1,
            resnet_model.bn1,
            resnet_model.relu,
            resnet_model.maxpool,
            resnet_model.layer1,
            resnet_model.layer2,
            resnet_model.layer3,
            resnet_model.layer4,
            resnet_model.avgpool
        )
        self.classifier = nn.Linear(2048, 4)
        nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, x):
        """Run Forward pass."""
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x


class ENet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b0')
        # 1280 is the number of neurons in last layer. is diff for diff. architecture
        self.dense_output = nn.Linear(1280, 4)

    def forward(self, x):
        feat = self.model.extract_features(x)
        feat = nn.functional.avg_pool2d(
            feat, feat.size()[2:]).reshape(-1, 1280)
        return self.dense_output(feat)
