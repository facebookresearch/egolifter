import torch
import torch.nn as nn
import torch.nn.functional as F

from .deform import get_embedder, Embedder

class DeformNetwork(nn.Module):
    def __init__(
        self, 
        D=8, 
        W=256, 
        input_ch=3, 
        multires=10,
    ):
        super().__init__()
        
        self.D = D
        self.skips = [D // 2]
        self.W = W
        self.input_ch = input_ch
        self.multires = multires
        self.t_multires = 10
        self.prob_logit_offset = -4
        
        self.embed_time_fn, self.time_input_ch = get_embedder(self.t_multires, 1)
        self.embed_fn, self.xyz_input_ch = get_embedder(self.multires, 3)

        # Encoder for xyz embedding
        self.encoder_xyz = nn.Sequential(
            nn.Linear(self.xyz_input_ch, W),
            nn.ReLU(inplace=True),
            nn.Linear(W, W),
            nn.ReLU(inplace=True),
        )
        
        # Decode to a single value for dynamic probability
        self.decoder_prob = nn.Linear(W, 1)
        
        # Encode both the position and time
        self.encoder_1 = nn.Sequential(
            nn.Linear(W + self.time_input_ch, W),
            nn.ReLU(inplace=True),
            nn.Linear(W, W),
            nn.ReLU(inplace=True),
            nn.Linear(W, W),
            nn.ReLU(inplace=True),
        )
        
        # Skip connection and encode again
        self.encoder_2 = nn.Sequential(
            nn.Linear(W + self.time_input_ch + self.xyz_input_ch, W),
            nn.ReLU(inplace=True),
            nn.Linear(W, W),
            nn.ReLU(inplace=True),
            nn.Linear(W, W),
            nn.ReLU(inplace=True),
        )
        
        self.decoder_xyz = nn.Linear(W, 3)
        self.decoder_rot = nn.Linear(W, 4)
        self.decoder_scaling = nn.Linear(W, 3)
        
    def forward(self, x, t):
        t_emb = self.embed_time_fn(t)
        x_emb = self.embed_fn(x)
        
        h = self.encoder_xyz(x_emb)

        # Push down the initial prediction of deforming probability
        prob = torch.sigmoid(self.decoder_prob(h) + self.prob_logit_offset) 
        
        h = torch.cat([h, t_emb], dim=1)
        h = self.encoder_1(h)
        
        h = torch.cat([h, x_emb, t_emb], dim=1)
        h = self.encoder_2(h)
        
        d_xyz = self.decoder_xyz(h)
        d_rot = self.decoder_rot(h)
        d_scale = self.decoder_scaling(h)
        
        return d_xyz, d_rot, d_scale, prob
        