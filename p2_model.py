import torch.nn as nn
import torchvision.models as models
import torch

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.encoder = models.resnet50(pretrained=False)
        self.encoder.load_state_dict(torch.load('backbone.pt')) 
        self.fc = nn.Sequential( 
            nn.Linear(1000, 512),
            nn.Linear(512, 65),
        )

    def forward(self, x):
        feature = self.encoder(x)
        output = self.fc(feature)
        return output
