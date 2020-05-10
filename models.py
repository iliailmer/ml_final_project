from torch import nn


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

    def forward(self, x):
        """Run Forward pass."""
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x
