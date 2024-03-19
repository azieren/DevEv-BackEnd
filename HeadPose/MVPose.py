import torch
from torch import nn
import torch.nn.functional as F


class MVPoseNet(nn.Module):
    def __init__(self, feature_dim):
        super(MVPoseNet, self).__init__()

        h1, h2 = 2048, 1024
        self.linear1 = nn.Linear(feature_dim, h1)
        self.linear11 = nn.Linear(h1, h1)
        self.self_attention = nn.MultiheadAttention(embed_dim=h1, num_heads=8, batch_first=True, add_bias_kv=True)
        #self.self_attention2 = nn.MultiheadAttention(embed_dim=h3, num_heads=8, batch_first=True, add_bias_kv=True)
        self.linear2 = nn.Linear(h1*8, h2)
        self.linear3 = nn.Linear(h2,3)
        self.norm1 = nn.LayerNorm(h1)
        self.norm2 = nn.LayerNorm(h2)

    def forward(self, x, views_mask = None):
        B, N, V = x.size()
        if views_mask is not None:
            view_mask = (1-views_mask).bool().to(x.device).repeat_interleave(8, dim=0)
            view_mask_s = view_mask.unsqueeze(1).repeat((1,8,1))

        x = torch.relu(self.linear1(x))
        x_, _ = self.self_attention(x, x, x, attn_mask=view_mask_s)
        x =  self.linear11(self.norm1(x+x_))

        x = torch.relu(x).view(B, -1)
        x = torch.relu(self.linear2(x))
        x = self.linear3(self.norm2(x))
        x = F.normalize(x, dim = -1)
        #x = utils.compute_rotation_matrix_from_ortho6d(x)
        return x

class MVPoseNet3(nn.Module):
    def __init__(self, feature_dim):
        super(MVPoseNet3, self).__init__()
        # Error epoch 1: 35.829
        h1, h2, h3 = 2048, 1024, 512
        self.linear1 = nn.Linear(feature_dim, h1)
        self.linear11 = nn.Linear(h1, h1)
        self.linear2 = nn.Linear(h1*8, h2)
        self.self_attention = nn.MultiheadAttention(embed_dim=h2//8, num_heads=8, batch_first=True, add_bias_kv=True)

        self.cross_attention = nn.MultiheadAttention(embed_dim=h3, num_heads=8, batch_first=True, add_bias_kv=True)
        self.linear3 = nn.Linear(h2//8,h3)
        self.linear4 = nn.Linear(h3,3)
      
        self.norm1 = nn.LayerNorm(h2//8)
        self.norm2 = nn.LayerNorm(h3)
        self.relu = nn.ReLU()
        self.token = nn.Parameter(torch.randn((1,1,h3)))
        
    def forward(self, x, views_mask = None):
        B, N, V = x.size()
        if views_mask is not None:
            view_mask = (1-views_mask).bool().to(x.device).repeat_interleave(8, dim=0)
            view_mask_s = view_mask.unsqueeze(1).repeat((1,8,1))
        x = self.linear1(x)
        x = self.relu(x)
        x =  self.linear11(x)
        x = self.relu(x).view(B, -1)
        x = self.linear2(x).view(B, N, -1)
        x_, _ = self.self_attention(x, x, x, attn_mask=view_mask_s)
        x = self.norm1(x+x_)
        x = self.linear3(x)
        x = self.relu(x)
        token = self.token.repeat((B,1,1))
        x, _ = self.cross_attention(token, x, x, attn_mask=view_mask.unsqueeze(1))
        x = self.norm2(token + x).squeeze(1)
        x = self.linear4(x)
        x = F.normalize(x, dim = -1)
        #x = utils.compute_rotation_matrix_from_ortho6d(x)
        return x


class ResidualLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResidualLinear, self).__init__()
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.norm_layer = nn.BatchNorm1d(output_dim)
        self.linear2 = nn.Linear(output_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.linear1(x)
        out = self.norm_layer(out.transpose(1,2)).transpose(1,2)
        #out = self.relu(out)
        out = torch.tanh(out)
        out = out + self.linear2(out)
        return  torch.tanh(out)

class MVPoseNet666(nn.Module):
    def __init__(self, feature_dim):
        super(MVPoseNet666, self).__init__()

        h1, h2, h3, h4 = 1024, 512, 512, 256
        self.linear1 = ResidualLinear(feature_dim, h1)
        self.linear2 = ResidualLinear(h1, h2)
        self.linear3 = ResidualLinear(h2*8,h3)
        self.linear4 = ResidualLinear(h3,h4)
        self.linear = []
        for i in range(5):
            self.linear.append(ResidualLinear(h4,h4))
        self.linear = nn.ModuleList(self.linear)
        self.linear5 = ResidualLinear(h4, 3)
        
    def forward(self, x, views_mask = None):
        B, N, V = x.size()
        x = self.linear1(x)
        x = self.linear2(x)
        x = x.reshape(B, 1, -1)
        x = self.linear3(x)
        x = self.linear4(x)
        for l in self.linear:
            x = l(x)
        x = self.linear5(x)
        return F.normalize(x, dim = -1).squeeze(1)
    
class MVPoseNet0(nn.Module):
    def __init__(self, feature_dim):
        super(MVPoseNet0, self).__init__()

        h1, h2, h3, h4 = 1024, 512, 512, 256
        self.linear1 = nn.Linear(feature_dim, h1)
        self.linear2 = nn.Linear(h1, h2)
        self.linear21 = nn.Linear(h2, h2)
        self.linear22 = nn.Linear(h2, h2)
        self.linear3 = nn.Linear(h2*8,h3)
        self.linear4 = nn.Linear(h3,h4)
        self.linear41 = nn.Linear(h4,h4)
        self.linear5 = nn.Linear(h4, 3)

        self.relu = nn.ReLU()
        
    def forward(self, x, views_mask = None):
        B, N, V = x.size()

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x).view(B, -1)
        x = self.relu(x)
        x = x + self.linear21(x)
        x = x + self.linear22(x)
        
        x = self.linear3(x)
        x = torch.sigmoid(x)
        x = self.linear4(x)
        x = torch.sigmoid(x)
        x = x + self.linear41(x)
        x = self.linear5(x)
        return F.normalize(x, dim = -1)