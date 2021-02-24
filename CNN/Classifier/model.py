import torch
import torch.nn as nn
import torchvision.models as models

resnets = [ 'ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152' ]



class Resnet(nn.Module):
    
    def __init__( self, 
        n_classes, 
        modelname = 'resnet152', 
        freeze = True 
    ):    
        super().__init__()  
        if modelname in resnets :
            self.resnet = getattr(models, modelname)(pretrained = True)
        else : 
            raise AttributeError(f"사용 가능 resnet {resnets}")
        if freeze:
            for param in self.resnet.parameters():
                param.requires_grad = False
                
        n_inputs = self.resnet.fc.out_features # 1000
        
        # 학습시킬 파라미터
        self.fc1 = nn.Linear(n_inputs, 1024)
        self.fc2 = nn.Linear(1024, n_classes)
        
    def forward(self, x):
        x = self.resnet(x)
        x = torch.relu(self.fc1(x))
        return torch.relu(self.fc2(x))
