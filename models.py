# TODO: add a Mask-RCNN model and maybe some other segmentation models?
# or object detection models (for the competition)

from torch import nn


class BackBone(nn.Module):
    def __init__(self,
                 net):
        super().__init__()
        # ecnoder

        self.features = nn.Sequential(
            net.conv1,
            net.bn1,
            net.relu,
            net.maxpool,
            net.layer1,
            net.layer2,
            net.layer3,
            net.layer4
        )
        self.out_channels = net.layer4[-1].conv3.out_channels

    def forward(self, x):
        return self.features(x)
