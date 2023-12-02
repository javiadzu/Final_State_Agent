from torchvision.transforms import functional as TF
import torch
import pystk

class ActionNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.BatchNorm2d(11),
            torch.nn.Conv2d(11, 16, 5, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, 5, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, 5, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, 5, stride=2),
            torch.nn.ReLU()
        )
        self.classifier = torch.nn.Linear(32, 2)
        
    def forward(self, x):
        f = self.network(x)
        return self.classifier(f.mean(dim = [2,3]))
        
class Action:
    def __init__(self, action_net):
        self.action_net = action_net.cpu().eval()
        
    def __call__(self, data, **kwargs):
        output = self.action_net(TF.to_tensor(data)[None])[0]
        action = pystk.Action()
        action.acceleration = output[0]
        action.steer = output[1]
        return action

action_net = ActionNet()
actor = Action(action_net)
    
    
        
        
        