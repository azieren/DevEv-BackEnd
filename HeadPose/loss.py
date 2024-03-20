import torch.nn as nn
import torch

#matrices batch*3*3
#both matrix are orthogonal rotation matrices
#out theta between 0 to 180 degree batch
class GeodesicLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, m1, m2):
        m = torch.bmm(m1, m2.transpose(1,2)) #batch*3*3
        
        cos = (  m[:,0,0] + m[:,1,1] + m[:,2,2] - 1 )/2        
        theta = torch.acos(torch.clamp(cos, -1+self.eps, 1-self.eps))
         
        return torch.mean(theta)
    
    
class HeadPoseLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, m1, m2):
        z = torch.FloatTensor([0.0,0.0,1.0]).to(m1.device)

        m = torch.matmul(m1 - m2, z.view(3,1)) #batch*3*1
        #print(m.size())
        m = torch.norm(m.squeeze(-1), dim = -1)
         
        return torch.mean(m)