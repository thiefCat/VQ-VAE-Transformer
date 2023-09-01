import os
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from . import utils as ut
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
# import cv2


# def imgs_to_npz():
#     npz = []

#     for img in os.listdir("./img_align_celeba"):
#         img_arr = cv2.imread("./img_align_celeba/" + img)
#         img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
#         resized_img = cv2.resize(img_arr, (64, 64), interpolation=cv2.INTER_CUBIC)
#         npz.append(resized_img)

#     output_npz = np.array(npz)
#     np.savez('celeba64_train.npz', output_npz)
#     print(f"{output_npz.shape} size array saved into celeba64_train.npz")  # (202599, 64, 64, 3)


def load_images(path='data/CelebA/celeba64.npz'):
    x = np.load(path)['arr_0']  # (202599, 64, 64, 3)
    return x


class CelebaDataset(Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        '''
        return (3, 64, 64) image
        '''
        x = self.data[index].astype(np.uint8)
        x = self.transform(x)
        return x, 0

    def __len__(self):
        return len(self.data)

def get_celeba_data(train_batch_size, path='data/CelebA/celeba64.npz'):
    '''
    return a dataloader
    '''
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
    ])

    train_images = load_images(path)  # ndarray

    train_dataset = CelebaDataset(train_images, transform=transform)   # pytorch dataset class, indexed by []
    train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True)
    
    return train_loader




if __name__ == '__main__':

    # imgs_to_npz()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
    ])
    images = load_images()
    celeba = CelebaDataset(images, transform)
    x = celeba[567].permute(1, 2, 0)
    x = ut.denormalize(x)
    plt.figure(figsize=(10, 10))
    plt.imshow(x)
    plt.savefig('./imgnet32_samples_4.jpg')
    plt.show()