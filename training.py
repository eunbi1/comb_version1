import os
from sampling import *
from torch.optim import Adam
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST, CIFAR10, CelebA, CIFAR100
import tqdm
import os
from scipy.stats import levy_stable
import matplotlib.pyplot as plt
import time
from model import *
from losses import *
from Diffusion import *
import numpy as np
import torch
from cifar10_model import *
import torchvision.transforms as transforms
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize

from torchvision import datasets, transforms

torch.multiprocessing.set_start_method('spawn')
from torchlevy import LevyStable

levy = LevyStable()

image_size = 28
channels = 1
batch_size = 128


def train(alpha_0=1.8, alpha_1=2, lr=1e-5, batch_size=128, beta_min=0.1, beta_max=7.5,
          n_epochs=15, num_steps=1000, datasets='MNIST', path=None, device='cuda',
          training_clamp=3, resolution=32, ch=128, ch_mult=[1, 2, 2,2], num_res_blocks=2, dropout=0.1, initial_epoch=1370, mode='approximation'):
    sde = VPSDE(alpha_0,alpha_1, beta_min=beta_min, beta_max=beta_max, device=device)

    if device == 'cuda':
        num_workers = 0
    else:
        num_workers = 4

    if datasets == "MNIST":
        image_size = 28
        channels = 1
        transform = Compose([
            Resize(image_size),
            CenterCrop(image_size),
            ToTensor()])
        dataset = MNIST('.', train=True, transform=transform, download=True)
        data_loader = DataLoader(dataset, batch_size=batch_size,
                                 shuffle=True, num_workers=num_workers, generator=torch.Generator(device=device))

        transformer = transforms.Compose([transforms.Resize((28, 28)),
                                          transforms.ToTensor()])
        validation_dataset = MNIST(root='./data', train=False, download=True, transform=transformer)
        validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)

        score_model = Unet(
            dim=image_size,
            channels=channels,
            dim_mults=(1, 2, 4,))

    if datasets == "CelebA":
        image_size = 32
        channels = 3

        transform = transforms.Compose([transforms.Resize((32, 32)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor()])
        dataset = CelebA('/home/eunbiyoon/comb_Levy_motion', transform=transform, download=True)

        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                 generator=torch.Generator(device=device))
        transformer = transforms.Compose(
            [transforms.Resize((32, 32)), transforms.ToTensor()
             ])
        validation_dataset = CelebA(root='/home/eunbiyoon/comb_Levy_motion', download=True, transform=transformer)
        validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)

        score_model = Model(resolution=image_size, in_channels=channels, out_ch=channels,
                            ch=ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks, dropout=dropout)

    if datasets == 'CIFAR100':
        transform = transforms.Compose([
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor()])

        dataset = CIFAR100('/scratch/private/eunbiyoon/data', train=True, transform=transform, download=True)
        data_loader = DataLoader(dataset, batch_size=batch_size,
                                 shuffle=True, num_workers=num_workers, generator=torch.Generator(device=device))
        transformer = transforms.Compose([transforms.Resize((32, 32)),
                                          transforms.ToTensor()
                                          ])
        validation_dataset = CIFAR100(root='/home/eunbiyoon/comb_Levy_motion', train=False, download=True,
                                     transform=transformer)
        validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)

        score_model = Model(ch=ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks, dropout=dropout)



    if datasets == 'CIFAR10':
        transform = transforms.Compose([
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor()])

        dataset = CIFAR10('/home/eunbiyoon/comb_Levy_motion', train=True, transform=transform, download=True)
        data_loader = DataLoader(dataset, batch_size=batch_size,
                                 shuffle=True, num_workers=num_workers, generator=torch.Generator(device=device))
        transformer = transforms.Compose([transforms.Resize((32, 32)),
                                          transforms.ToTensor()
                                          ])
        validation_dataset = CIFAR10(root='/home/eunbiyoon/comb_Levy_motion', train=False, download=True,
                                     transform=transformer)
        validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)

        score_model = Model(ch=ch, ch_mult=ch_mult, resolution=32, num_res_blocks=num_res_blocks, dropout=dropout)

    score_model = score_model.to(device)
    if path:
        ckpt = torch.load(path, map_location=device)
        score_model.load_state_dict(ckpt, strict=False)

    optimizer = Adam(score_model.parameters(), lr=lr)
    score_model.train()

    counter = 0
    t0 = time.time()
    L = []
    L_val = []
    for epoch in range(n_epochs):
        counter += 1
        avg_loss = 0.
        num_items = 0
        i = 0
        for x, y in data_loader:
            x = 2 * x - 1

            x = x.to(device)
            t = torch.sort(torch.rand(x.shape[0]))[0].to(device)+1e-4

            loss = loss_fn(score_model, sde, x, t, num_steps=num_steps, mode=mode, training_clamp=training_clamp)


            optimizer.zero_grad()
            loss.backward()
            print(f'{epoch} th epoch {i} th step loss: {loss}')
            i += 1
            optimizer.step()
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]
            torch.save(score_model.state_dict(), 'ckpt.pth')

        L.append(avg_loss / num_items)

        t1 = time.time()
        name =  mode
        ckpt_name = str(datasets) + str(
            f'batch{batch_size}ch{ch}ch_mult{ch_mult}num_res{num_res_blocks}dropout{dropout}') + str(
            f'clamp{training_clamp}') + str(f'_epoch{epoch+initial_epoch}_') + str(f'{alpha_1}_{beta_min}_{beta_max}.pth')
        dir_path = name+str(datasets) + str(f'batch{batch_size}lr{lr}ch{ch}ch_mult{ch_mult}num_res{num_res_blocks}dropout{dropout}') + str(
            f'clamp{training_clamp}') + str(f'{alpha_1}_{beta_min}_{beta_max}')
        dir_path = os.path.join('/home/eunbiyoon/combine', dir_path)
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
        X = np.arange(epoch + 1)
        plt.plot(X, L, 'r', label='training loss')
        plt.legend()
        name_ = os.path.join(dir_path,  'loss.png')
        plt.savefig(name_)
        plt.cla()

        dir_path2 = os.path.join(dir_path, 'ckpt')
        if not os.path.isdir(dir_path2):
            os.mkdir(dir_path2)

        ckpt_name = os.path.join(dir_path2, ckpt_name)
        torch.save(score_model.state_dict(), ckpt_name)
        name= str(epoch+initial_epoch)+mode
        sample(alpha_0=sde.alpha_0,alpha_1=sde.alpha_1, path=ckpt_name,
               beta_min=beta_min, beta_max=beta_max, sampler='pc_sampler2', batch_size=64, num_steps=num_steps, LM_steps=50,
               Predictor=True, Corrector=False, trajectory=False, clamp=training_clamp, initial_clamp=training_clamp,
               clamp_mode="constant",
               datasets=datasets, name=name,
               dir_path=dir_path, ch=ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks, resolution=resolution)
