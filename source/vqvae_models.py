import torch
import torch.nn as nn
import source.utils as ut
from einops import rearrange, reduce, repeat
import math

class VQEmbedding(nn.Module):
    '''
    K: the length of the codebook
    D: the latent dimension of the codebook
    '''
    def __init__(self, K, D, beta=0.25):
        super().__init__()
        self.K = K
        self.D = D
        self.beta = beta
        self.embedding = nn.Embedding(K, D)
    
    def forward(self, z_e):
        '''
        z_e: (b, c, h, w), where c is the latent dimension
        return: quantized vector
        '''
        b, c, h, w = z_e.shape
        z_e = rearrange(z_e, 'b c h w -> (b h w) c')
        indices = ut.vq(z_e, self.embedding.weight)  # [(b h w)]
        z_q = self.embedding.weight[indices] # (-1, c)
        res = z_e + (z_q - z_e).detach()
        res = rearrange(res, '(b h w) c -> b c h w', b=b, h=h, w=w).contiguous()
        # calculate the loss in the notebook
        e_latent_loss = torch.mean((z_q.detach() - z_e) ** 2)
        q_latent_loss = torch.mean((z_q - z_e.detach()) ** 2)
        codebook_loss = q_latent_loss + self.beta * e_latent_loss
        return res, codebook_loss, indices


class ResDown(nn.Module):
    def __init__(self, channel_in, channel_out, activation=nn.ELU()):
        super().__init__()
        channel_between = (channel_in + channel_out) // 2
        self.conv1 = nn.Conv2d(channel_in, channel_between, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(channel_between)
        self.conv2 = nn.Conv2d(channel_between, channel_out, 3, 1, 1)
        self.conv3 = nn.Conv2d(channel_in, channel_out, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(channel_out)
        self.activation = activation

    def forward(self, x):
        skip = self.conv3(x)
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        sum = x + skip
        out = self.activation(self.bn2(sum))
        return out


class ResUp(nn.Module):
    '''
    This block uses batchnorm and residual connection so that the model can go deeper
    bn before activation: 
    1. zero out roughtly half of the inputs --> regularization
    2. can potentially mitigate dead activation problem: making the inputs mean 0, variance 1
       so that some values < 0 can be scaled up --> guarantee roughly 50% inputs is 0
    '''
    def __init__(self, channel_in, channel_out, output_padding=0, activation=nn.ELU()):
        super().__init__()
        channel_between = (channel_in + channel_out) // 2
        self.conv1 = nn.ConvTranspose2d(channel_in, channel_between, 4, 2, 1, output_padding)
        self.bn1 = nn.BatchNorm2d(channel_between)
        self.conv2 = nn.ConvTranspose2d(channel_between, channel_out, 3, 1, 1)
        self.conv3 = nn.ConvTranspose2d(channel_in, channel_out, 4, 2, 1, output_padding)
        self.bn2 = nn.BatchNorm2d(channel_out)
        self.activation = activation

    def forward(self, x):
        skip = self.conv3(x)
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        sum = x + skip
        out = self.activation(self.bn2(sum))
        return out
        

class Encoder(nn.Module):
    '''
    (batch, 28, 28)
    '''
    def __init__(self, in_channels, ch, latent_dim, num_res_layers, activation=nn.ELU()):
        super().__init__()
        self.in_channels = in_channels
        self.ch = ch
        self.num_res_layers = num_res_layers
        self.activation = activation
        self.input_conv = nn.Conv2d(in_channels, ch, 7, 1, 3)
        self.ResStack = self.construct_layers()
        self.output_conv = nn.Conv2d(2**(num_res_layers)*self.ch, latent_dim, 3, 1, 1)
    
    def construct_layers(self):
        layers = []
        for i in range(self.num_res_layers):
            layers.append(ResDown(2**i*self.ch, 2**(i+1)*self.ch, self.activation))
        return nn.ModuleList(layers)
    
    def forward(self, x):
        if self.in_channels == 1:
            x = x.unsqueeze(1)  # (batch, 28, 28) --> (batch, 1, 28, 28)
        x = self.activation(self.input_conv(x))
        for layer in self.ResStack:
            x = layer(x)
        x = self.output_conv(x)
        return x
        

class Decoder(nn.Module):
    '''
    (batch, 28, 28)
    '''
    def __init__(self, in_channels, ch, latent_dim, num_res_layers, activation=nn.ELU()):
        super().__init__()
        self.in_channels = in_channels
        self.ch = ch
        self.num_res_layers = num_res_layers
        self.activation = activation
        self.input_conv = nn.ConvTranspose2d(latent_dim, 2**(num_res_layers)*self.ch, 3, 1, 1)
        self.ResStack = self.construct_layers()
        self.output_conv = nn.Conv2d(ch, in_channels, 3, 1, 1)
        self.output_act = nn.Tanh()
    
    def construct_layers(self):
        layers = []
        if self.in_channels == 1:
            layers.append(ResUp(2**3*self.ch, 2**2*self.ch, output_padding=1, activation=self.activation))
            layers.append(ResUp(2**2*self.ch, 2**1*self.ch, output_padding=0, activation=self.activation))
            layers.append(ResUp(2*self.ch, self.ch, output_padding=0, activation=self.activation))
        else:
            for i in range(self.num_res_layers, 0, -1):
                print(i)
                layers.append(ResDown(2**i*self.ch, 2**(i-1)*self.ch, self.activation))
        return nn.ModuleList(layers)
    
    def forward(self, x):
        x = self.activation(self.input_conv(x))
        for layer in self.ResStack:
            x = layer(x)
        out = self.output_act(self.output_conv(x))
        if self.in_channels == 1:
            out = out.squeeze(1)  # (batch, 1, 28, 28) --> (batch, 28, 28)
        return out


class VQVAE(nn.Module):
    def __init__(self, in_channels, ch, latent_dim, num_res_layers, K, alpha, beta, activation=nn.ELU()):
        super().__init__()
        self.K = K
        self.num_res_layers = num_res_layers
        self.in_channels = in_channels
        self.alpha = alpha
        self.encoder = Encoder(in_channels, ch, latent_dim, num_res_layers, activation)
        self.decoder = Decoder(in_channels, ch, latent_dim, num_res_layers, activation)
        self.codebook = VQEmbedding(K, latent_dim, beta)
    
    def encode(self, x):
        '''
        x: (batch, 28, 28) or (batch, 3, 64, 64)
        return: 
        indices: the indices of the codebook, which is used to train the transformer model
        '''
        z_e = self.encoder(x)  # (batch, D, H, W)
        shape = z_e.shape
        _, _, indices = self.codebook(z_e) # [(b h w)]
        return indices, shape

    def decode(self, z_q):
        '''
        input: indices, shape
        return: reconstruction
        '''
        x_hat = self.decoder(z_q)
        return x_hat

    def forward(self, x):
        '''
        Return:
        z_e: encoder output
        z_q: quantized latents
        x_hat: reconstruction 
        '''
        z_e = self.encoder(x)
        z_q, codebook_loss, indices = self.codebook(z_e)
        x_hat = self.decoder(z_q)
        # calculate loss
        rec = torch.mean((x - x_hat) ** 2)
        loss = rec + self.alpha * codebook_loss    
        return x_hat, loss, rec, codebook_loss 