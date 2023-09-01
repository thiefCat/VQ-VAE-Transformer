import torch
import torchvision
import torch.nn as nn
import os
from source import dataset
from source import celeba64
from source import vqvae_models, transformer_models
import source.utils as ut
from pprint import pprint
import argparse
from tqdm import tqdm, trange
from torch.nn import functional as F
import matplotlib.pyplot as plt
import einops

def build_config_from_args(is_jupyter=False):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--d', type=int, default=64, help="dimension of the heads in transformer")
    parser.add_argument('--width', type=int, default=16, help="number of channels of convolution")
    parser.add_argument('--num_layers', type=int, default=3, help="number of stacked transformer blocks")
    parser.add_argument('--num_embeddings', type=int, default=128, help="length of the codebook (embedding)")
    parser.add_argument('--num_heads', type=int, default=5, help="number of heads in attention layer")
    parser.add_argument('--d_ffn', type=int, default=64, help="dimension of the feed forward layer")
    parser.add_argument('--p_drop', type=float, default=0.1, help="dropout probability")
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion', 'celeba'])
    parser.add_argument('--iter_save', type=int, default=50, help="Save running loss every n iterations")
    parser.add_argument('--train', type=int, default=1, help="Flag for training")
    parser.add_argument('--out_dir', type=str, default="VQ-VAE/results", help="Directory of output logging")
    parser.add_argument('--conditional', type=int, default=1, help="Flag for conditional generation")
    parser.add_argument('--alpha', type=float, default=2, help="The hyperparameter controlling the balance between rec loss and codebook loss")
    parser.add_argument('--beta', type=float, default=0.1, help="The hyperparameter controlling the balance between z_latent_loss and q_latent_loss")
    parser.add_argument('--latent_dim', type=int, default=8, help="dimension of the latent space")
    parser.add_argument('--num_res_layers', type=int, default=3, help="number of stacked residual blocks")
    if is_jupyter:
        args = parser.parse_args(args=[])
    else:
        args = parser.parse_args()
    return args

def load_dataset(type, train_batch_size=256):
    if type == 'mnist':
        train_loader, test_loader = dataset.get_mnist_data(train_batch_size, test_batch_size=10)
    elif type == 'celeba':
        train_loader, test_loader = celeba64.get_celeba_data(train_batch_size), None
    else:
        train_loader, test_loader = dataset.get_fashion_mnist_data(train_batch_size, test_batch_size=10)
    return train_loader, test_loader

class TransformerTrainer:
    def __init__(self, train_data, transformer: nn.Module,
                 vq_vae: nn.Module, lr: float, num_epochs, config, save=True):
        self.config = config
        self.train_data = train_data
        self.num_epochs = num_epochs
        self.optimizer = torch.optim.AdamW(transformer.parameters(), lr=lr, weight_decay=1e-2)
        self.save = save
        self.train_loss=[]
        self.test_loss=[]
        self.save = save
        self.vq_vae = vq_vae
        # freeze the vq-vae encoder
        for param in vq_vae.encoder.parameters():
            param.requires_grad = False
        if torch.cuda.is_available():
            self.transformer = transformer.cuda()
        else:
            self.transformer = transformer
    
    def get_sequence(self, data, y, conditional=False):
        '''
        Process the data by vq-vae encoder, and append the sos token
        data: [batch, 28, 28] or [batch, 3, x, x]
        Return: 
        indices: [batch, seq_len + 1], where seq_len = h x w
        labels: ground truth
        '''
        indices, shape = self.vq_vae.encode(data)  # [(b h w)]
        b = shape[0]
        indices = indices.unsqueeze(1) # [(b h w), 1]
        labels = einops.rearrange(indices, '(b hw) 1 -> b (hw)', b=b).contiguous() # [batch, seq_len]
        if conditional:
            y = y + self.config.num_embeddings
            sos_tensor = y.unsqueeze(1)
        else:
            sos_index = 0
            sos_tensor = torch.full((b, 1), sos_index, device=indices.device)
        indices = torch.cat([sos_tensor, labels], dim=1) # [batch, seq_len + 1]
        return indices, labels

    def train(self):
        os.makedirs(self.config.out_dir, exist_ok=True)
        # self.model.train()
        epoch_iterator = trange(self.num_epochs)
        iter_save = self.config.iter_save
        for epoch in epoch_iterator: 
            self.transformer.train()
            running_loss = 0.0
            data_iterator = tqdm(self.train_data)
            for batch_idx, (data, y) in enumerate(data_iterator, 0):
                data = data.to(device)  # (batch, 28, 28)
                y = y.to(device)
                sequences, labels = self.get_sequence(data, y, config.conditional) # [batch, seq_len + 1]
                self.optimizer.zero_grad()
                logits = self.transformer(sequences) # [batch, seq_len+1, K]
                loss = self.get_loss(logits[:,:-1,:], labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                
                if (batch_idx+1) % iter_save == 0:    # recalculate the loss every iter_save batches
                    self.train_loss.append(running_loss / iter_save)
                    data_iterator.set_postfix({'loss': running_loss / iter_save})
                    running_loss = 0.0

        if self.save:
            torch.save(self.transformer.state_dict(), self.config.out_dir + '/transformer.pt')
    
    def get_loss(self, logits, labels):
        '''
        logits: [b, seq_len, K]
        labels: [b, seq_len]
        '''
        criterion = nn.CrossEntropyLoss()
        logits = logits.contiguous()
        logits = logits.view(-1, logits.size(-1))  # [b*seq_len, K]
        labels = labels.view(-1)  # [b*seq_len]
        loss = criterion(logits, labels)
        return loss
        
    def plt_loss(self):
        x1 = range(0, len(self.train_loss))
        y1 = self.train_loss
        plt.plot(x1, y1, '.-')
        plt.title('Train loss vs. batches')
        plt.ylabel('Train loss')
        plt.savefig(self.config.out_dir + '/transformer_train_loss.png')  # specify the full path here
        plt.close()
    
    def inference(self):
        """
        Randomly generate 100 samles, save it in results/generated.png
        """
        self.transformer.eval()
        with torch.no_grad():
            generated_sequences = self.transformer.sample_from_prior(num_samples=100, 
                                                                     temperature=2, 
                                                                     k=8, 
                                                                     device=device)  # [100, seq_len]
            seq_len = generated_sequences.shape[1]
            size = int(seq_len ** 0.5)
            generated_latents = self.vq_vae.codebook.embedding.weight[generated_sequences] # [100, seq_len, d]
            generated_latents = einops.rearrange(generated_latents, 'b (h w) d -> b d h w', h=size, w=size) 
            reconstructions = self.vq_vae.decode(generated_latents) # [b, h, w]
            reconstructions = ut.denormalize(reconstructions)
            if self.vq_vae.in_channels == 1:
                reconstructions = reconstructions.unsqueeze(1)
        torchvision.utils.save_image(
            reconstructions, self.config.out_dir + '/generated.png', nrow=10)
            
    def inference_by_class(self):
        self.transformer.eval()
        generated_sequences = []
        with torch.no_grad():
            for y in range(10):
                generated_sequences.append(self.transformer.sample_from_class(num_samples=10, 
                                                                        temperature=2, 
                                                                        y=y,
                                                                        k=8, 
                                                                        device=device))
            generated_sequences = torch.cat(generated_sequences, dim=0)
            seq_len = generated_sequences.shape[1]
            size = int(seq_len ** 0.5)
            generated_latents = self.vq_vae.codebook.embedding.weight[generated_sequences] # [100, seq_len, d]
            generated_latents = einops.rearrange(generated_latents, 'b (h w) d -> b d h w', h=size, w=size) 
            reconstructions = self.vq_vae.decode(generated_latents) # [b, h, w]
            reconstructions = ut.denormalize(reconstructions)
            if self.vq_vae.in_channels == 1:
                reconstructions = reconstructions.unsqueeze(1)
        torchvision.utils.save_image(
            reconstructions, self.config.out_dir + '/generated.png', nrow=10)

def get_latent_size(input_size, model):
    num_layers = model.num_res_layers
    size = input_size
    for _ in range(num_layers):
        size = size // 2
    return size

def save_image(x, mono=True):
    '''
    expect a single image x with shape [C, H, W] for colored image, and [H, W] for black white image
    '''
    x = ut.denormalize(x)
    if mono:
        x = x.unsqueeze(0)
    torchvision.utils.save_image(x, 'VQ-VAE/results/show.png')

if __name__ == '__main__':
    config = build_config_from_args()
    pprint(vars(config))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    train_loader, test_loader = load_dataset(config.dataset, train_batch_size=128)  # the training data is in range [-1, 1]
    in_dim = None
    if config.dataset in ['mnist', 'fashion']:
        in_dim = 1
        size = 28
    else:
        in_dim = 3
        size = 64
    
    vq_vae = vqvae_models.VQVAE(in_dim, config.width, config.latent_dim, config.num_res_layers,
                          config.num_embeddings, config.alpha, config.beta)
    vq_vae.load_state_dict(torch.load(config.out_dir + '/vqvae.pt'))
    vq_vae.to(device)
    vq_vae.eval()
    latent_size = get_latent_size(size, vq_vae)
    max_seq_len = latent_size ** 2 + 1

    autoregressive_transformer = transformer_models.AutoregressiveTransformer(config.num_layers, 
                                config.d, config.num_embeddings, max_seq_len, config.num_heads,
                                config.d_ffn, config.p_drop, vq_vae)

    # autoregressive_transformer = GPT(config.num_embeddings, max_seq_len, 5, 8, 128)

    trainer = TransformerTrainer(train_loader, autoregressive_transformer, 
                                 vq_vae, lr=0.0015, num_epochs=8, config=config)

    if config.train:
        trainer.train()
        trainer.plt_loss()
    else:
        trainer.transformer.load_state_dict(torch.load(config.out_dir + '/transformer.pt'))
    # trainer.inference()
    trainer.inference_by_class()

    # with torch.no_grad():
    #     data_test = next(iter(train_loader))[0].to(device)
    #     # data_test = data_test[:2]
    #     sequences, labels = trainer.get_sequence(data_test)
    #     # print(sequences)
    #     # print(labels)
    #     predictions = trainer.transformer(sequences)
    #     prediction = predictions[0]
    #     # print(prediction.shape)
    #     print(sequences[0])
    #     values, indices = torch.max(prediction, -1)
    #     print(indices)



        # generated_latents = vq_vae.codebook.embedding.weight[labels] # [100, seq_len, d]
        # size = 3
        # generated_latents = einops.rearrange(generated_latents, 'b (h w) d -> b d h w', h=size, w=size) 
        # reconstructions = vq_vae.decode(generated_latents) # [b, h, w]
        # reconstructions = ut.denormalize(reconstructions)

    #     x_hat, _, _, _ = self.model(data_test)
    #     data_test = ut.denormalize(data_test)
    #     x_hat = ut.denormalize(x_hat)
    #     if mono:
    #         data_test = data_test.unsqueeze(1)
    #         x_hat = x_hat.unsqueeze(1)
    #     stacked = torch.cat([data_test, x_hat], dim=0)
    # torchvision.utils.save_image(
    # stacked, self.config.out_dir + '/reconstruction.png', nrow=10)