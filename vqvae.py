import torch
import torchvision
import torch.nn as nn
import os
from source import dataset
from source import celeba64
from source import vqvae_models
import source.utils as ut
from pprint import pprint
import argparse
from tqdm import tqdm, trange
from torch.nn import functional as F
import matplotlib.pyplot as plt


def build_config_from_args(is_jupyter=False):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--latent_dim', type=int, default=8, help="dimension of the latent space")
    parser.add_argument('--width', type=int, default=16, help="number of channels of convolution")
    parser.add_argument('--num_embeddings', type=int, default=128, help="length of the codebook")
    parser.add_argument('--num_res_layers', type=int, default=3, help="number of stacked residual blocks")
    parser.add_argument('--architecture', type=int, default=2, choices=[1, 2])
    parser.add_argument('--dataset', type=str, default='fashion', choices=['mnist', 'fashion'])
    parser.add_argument('--alpha', type=float, default=2, help="The hyperparameter controlling the balance between rec loss and codebook loss")
    parser.add_argument('--beta', type=float, default=0.1, help="The hyperparameter controlling the balance between z_latent_loss and q_latent_loss")
    parser.add_argument('--iter_save', type=int, default=50, help="Save running loss every n iterations")
    parser.add_argument('--train', type=int, default=1, help="Flag for training")
    parser.add_argument('--out_dir', type=str, default="VQ-VAE/results", help="Flag for output logging")
    if is_jupyter:
        args = parser.parse_args(args=[])
    else:
        args = parser.parse_args()
    return args

def load_dataset(type, train_batch_size=256):
    if type == 'mnist':
        train_loader, test_loader = dataset.get_mnist_data(train_batch_size, test_batch_size=10)
    else:
        train_loader, test_loader = dataset.get_fashion_mnist_data(train_batch_size, test_batch_size=10)
    return train_loader, test_loader


class Experiment:
    def __init__(self, train_data, eval_data, model: nn.Module,
                 lr: float, num_epochs, config, save=True):
        self.config = config
        self.train_data = train_data
        self.eval_data = eval_data
        self.num_epochs = num_epochs
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.save = save
        self.train_loss=[]
        self.test_loss=[]
        self.save = save
        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model

    def train(self):
        os.makedirs(self.config.out_dir, exist_ok=True)
        self.model.train()
        epoch_iterator = trange(self.num_epochs)
        iter_save = self.config.iter_save
        for epoch in epoch_iterator: 
            running_loss = running_rec = running_codebook = 0.0
            data_iterator = tqdm(self.train_data)
            for batch_idx, (data, _) in enumerate(data_iterator, 0):
                data = data.to(device)  # (batch, 28, 28)
                self.optimizer.zero_grad()
                x_hat, loss, rec, codebook_loss = self.model(data)
                loss.backward()
                # print_gradients(self.model)
                self.optimizer.step()
                # print_weight(experiment.model)

                running_loss += loss.item()
                running_rec += rec.item()
                running_codebook += codebook_loss.item()
                
                if (batch_idx+1) % iter_save == 0:    # recalculate the loss every iter_save batches
                    self.train_loss.append(running_loss / iter_save)
                    data_iterator.set_postfix({'loss': running_loss / iter_save,
                                                'rec': running_rec / iter_save,
                                                'codebook_loss': running_codebook / iter_save})
                    running_loss = running_rec = running_codebook = 0.0
            if eval_data != None:
                self.evaluate()

        if self.save:
            torch.save(self.model.state_dict(), self.config.out_dir + '/vqvae.pt')
    
    def evaluate(self):
        with torch.no_grad():
            eval = next(iter(self.eval_data))[0].to(device)
            x_hat, loss, rec, codebook_loss = self.model(eval)
            print("Test Loss: total: {}. Rec: {}. codebook: {}".format(loss, rec, codebook_loss))
        return loss, rec, codebook_loss

    def plt_loss(self):
        x1 = range(0, len(self.train_loss))
        y1 = self.train_loss
        plt.plot(x1, y1, '.-')
        plt.title('Train loss vs. batches')
        plt.ylabel('Train loss')
        plt.savefig(self.config.out_dir + '/vqvae_train_loss.png')  # specify the full path here
        plt.close()

    def vis_reconstruction(self):
        '''
        Visualize 10 reconstructions, the first row is original data, the second
        row is reconstructed data
        mono: if the dataset is black-white, then mono=True
        '''
        with torch.no_grad():
            data_test = next(iter(self.train_data))[0].to(device)
            data_test = data_test[:10]
            x_hat, _, _, _ = self.model(data_test)
            data_test = ut.denormalize(data_test)
            x_hat = ut.denormalize(x_hat)
            data_test = data_test.unsqueeze(1)
            x_hat = x_hat.unsqueeze(1)
            stacked = torch.cat([data_test, x_hat], dim=0)
        torchvision.utils.save_image(
        stacked, self.config.out_dir + '/reconstruction.png', nrow=10)

    
    def save_image(self, x):
        '''
        expect a single image x with shape [C, H, W] for colored image, and [H, W] for black white image
        '''
        x = ut.denormalize(x)
        x = x.unsqueeze(0)
        torchvision.utils.save_image(x, self.config.out_dir + '/show.png')
    
def print_gradients(model):
    for name, param in model.named_parameters():
        if name == 'codebook.embedding.weight': 
            if param.requires_grad:
                if param.grad is not None:
                    print(name, param.grad.data)
                else:
                    print(f"No gradient for {name}")

def print_weight(model):
    for name, param in model.named_parameters():
        if name == 'codebook.embedding.weight': 
                print(name, param.data)

if __name__ == '__main__':
    config = build_config_from_args()
    pprint(vars(config))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    train_loader, test_loader = load_dataset(config.dataset, train_batch_size=256)  # the training data is in range [-1, 1]
    in_dim = 1
    eval_data = dataset.get_eval_data(test_loader)
    model = vqvae_models.VQVAE(config.architecture, in_dim, config.width, config.latent_dim, config.num_res_layers,
                          config.num_embeddings, config.alpha, config.beta)
    experiment = Experiment(train_loader, eval_data, model, lr=0.003, num_epochs=10, config=config)

    if config.train:
        experiment.train()
        experiment.plt_loss()
    else:
        experiment.model.load_state_dict(torch.load(config.out_dir + '/vqvae.pt'))
    experiment.vis_reconstruction()


    # first_batch = next(iter(train_loader))
    # imgs = first_batch[0].to(device)
    # indices, shape = experiment.model.encode(imgs)
    # reconstruction = experiment.model.decode(indices, shape)
    # experiment.save_image(reconstruction[0])
    # print(indices.shape)
    # print(shape)

    # print_weight(experiment.model)
    # print(reconstructions[0])
    # print(z_e-z_q)