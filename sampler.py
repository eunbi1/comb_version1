from torchvision.utils import make_grid
import tqdm
from scipy.stats import levy_stable
import matplotlib.pyplot as plt
from losses import *
import numpy as np
import torch
from Diffusion import *

from torchlevy import LevyStable

levy = LevyStable()

device= 'cuda'
## Sample visualization.

def visualization(samples, sample_batch_size=64):
    samples = samples.clamp(0.0, 1.0)
    sample_grid = make_grid(samples, nrow=int(np.sqrt(sample_batch_size)))
    plt.figure(figsize=(6, 6))
    plt.axis('off')
    plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
    plt.show()


def gamma_func(x):
    return torch.tensor(gamma(x)).to(device)


def get_discrete_time(t, N=1000):
    return N * t


def ddim_score_update2(score_model, sde, x_s, s, t, h=0.01, clamp = 10, device='cuda', mode='approximation'):
    score_s = score_model(x_s, s)
    
    time_step = s-t
    time_step = time_step.to(device)
    beta_step = sde.beta(s)*time_step
    alpha = sde.alpha(t[0]).item()
    x_coeff = 1 + beta_step/alpha
    if sde.alpha(s[0])==2:
        score_coeff2 = torch.pow(beta_step, 2 / 2) * gamma_func(2 + 1)

    else:
        alpha= sde.alpha(s[0]).item()
        score_coeff2 = torch.pow(beta_step, 2/alpha)*torch.pow(time_step, 1-2/alpha)*np.sin(torch.pi/2*(2-alpha))/(2-alpha)*2/torch.pi*gamma_func(alpha+1)
        #score_coeff2 = 2*gamma_func(alpha - 1) / gamma_func(alpha / 2) ** 2 / h ** (alpha - 2) * sde.beta(s) * time_step
    noise_coeff = torch.pow(beta_step, 1 / alpha)

    if mode == "approximation" or "normal":
        e_L = torch.clamp(levy.sample(sde.alpha(t[0]), 0, size=x_s.shape).to(device),-100,100)

    #e_L = torch.clamp(levy.sample(alpha, 0, clamp=3, size=x_s.shape).to(device), -clamp, clamp)

    x_t = x_coeff[:, None, None, None] * x_s + score_coeff2[:, None, None, None] * score_s + noise_coeff[:, None, None,None] * e_L
    #print('score_coee', torch.min(score_coeff), torch.max(score_coeff))
    #print('noise_coeff',torch.min(noise_coeff), torch.max(noise_coeff))
    #print('x_coeff', torch.min(x_coeff), torch.max(x_coeff))

    print('x_s range', torch.min(x_s), torch.max(x_s))
    print('x_t range', torch.min(x_t), torch.max(x_t))
    print('x_s mean', torch.mean(x_s))
    print('x_t mean', torch.mean(x_t))
    print('score range',torch.min(score_s), torch.max(score_s))

    #print('x coeff adding', torch.min(x_coeff[:, None, None, None] * x_s), torch.max(x_coeff[:, None, None, None] * x_s))
    #print('score adding',torch.min(score_coeff[:, None, None, None] * score_s), torch.max(score_coeff[:, None, None, None] * score_s) )
    #print('noise adding', torch.min(noise_coeff[:, None, None,None] * e_L), torch.max(noise_coeff[:, None, None,None] * e_L))
    x_t = torch.clamp(x_t, -clamp, clamp)
    return x_t


def pc_sampler2(score_model,
                sde,
                batch_size,
                num_steps,
                LM_steps=200,
                device='cuda',
                eps=1e-4,
                x_0= None,
                Predictor=True, mode='approximation',
                Corrector=False, trajectory=False,
                clamp = 10,
                initial_clamp =10, final_clamp = 1,
                datasets="MNIST", clamp_mode = 'constant', h=0.9):
    t = torch.ones(batch_size, device=device)
    if datasets =="MNIST":

        if mode == "approximation" or "normal":
            x_s = torch.clamp(levy.sample(sde.alpha(1), 0, size=(batch_size, 1, 28,28)).to(device), -initial_clamp,initial_clamp)
        if mode == "resampling":
            x_s = levy.sample(alpha, 0, clamp=initial_clamp, size=(batch_size, 1, 28,28)).to(device)
        if mode == 'brownian':
            x_s = torch.clamp(levy.sample(1.5, 0, size=(batch_size, 1, 28,28)).to(device), -initial_clamp, initial_clamp)

    elif datasets == "CIFAR10":
        e_L = levy.sample(sde.alpha(1), 0, (batch_size, 3, 32, 32)).to(device)
        x_s = torch.clamp(e_L, -initial_clamp, initial_clamp)


    elif datasets == "CelebA":
        if mode == "approximation" or "normal":
            x_s = torch.clamp(levy.sample(sde.alpha(1), 0, size=(batch_size, 3, 64, 64)).to(device), -initial_clamp,
                              initial_clamp)
        if mode == "resampling":
            x_s = levy.sample(alpha, 0, clamp=initial_clamp, size=(batch_size, 3, 64, 64)).to(device)
        if mode == 'brownian':
            x_s = torch.clamp(levy.sample(1.5, 0, size=(batch_size, 3, 64, 64)).to(device), -initial_clamp,
                              initial_clamp)

    if trajectory:
        samples = []
        samples.append(x_s)
    time_steps = torch.pow(torch.linspace(1.,0, num_steps),1)
    step_size = time_steps[0] - time_steps[1]

    batch_time_step_s = torch.ones(x_s.shape[0]) * time_steps[0]
    batch_time_step_s = batch_time_step_s.to(device)

    with torch.no_grad():
        for t in tqdm.tqdm(time_steps[1:]):
            batch_time_step_t = torch.ones(x_s.shape[0]) * t
            batch_time_step_t = batch_time_step_t.to(device)

            if clamp_mode == "constant":
                linear_clamp = clamp
            if clamp_mode == "linear":
                linear_clamp = batch_time_step_t[0] * (clamp - final_clamp) + final_clamp
            if clamp_mode == "root":
                linear_clamp = torch.pow(batch_time_step_t[0], 1 / 2) * (clamp - final_clamp) + final_clamp
            if clamp_mode == "quad":
                linear_clamp = batch_time_step_t[0] ** 2 * (clamp - final_clamp) + final_clamp

            if Predictor:
                x_s = ddim_score_update2(score_model, sde, x_s, batch_time_step_s, batch_time_step_t,clamp=linear_clamp)
            if trajectory:
                samples.append(x_s)


            if Corrector:
                for j in range(LM_steps):
                    grad = score_model(x_s, batch_time_step_s)*torch.pow(sde.marginal_std(batch_time_step_s) + 1e-8, -1)[:, None, None,None]
                    if datasets == "MNIST":
                        e_L = levy.sample(alpha, 0, (batch_size, 1, 28, 28)).to(device)
                        e_L = torch.clamp(e_L, -final_clamp, final_clamp)
                    elif datasets == "CIFAR10":
                        e_L = levy.sample(alpha, 0, (batch_size, 3, 32, 32)).to(device)
                        e_L = torch.clamp(e_L, -final_clamp, final_clamp)

                    elif datasets == "CelebA":
                        e_L = levy.sample(alpha, 0, (batch_size, 3, 64, 64)).to(device)

                        e_L = torch.clamp(e_L, -final_clamp, final_clamp)

                    x_s = x_s + step_size * gamma_func(sde.alpha - 1) / ( gamma_func(sde.alpha / 2) ** 2) /h**(alpha-2)*grad + torch.pow(step_size, 1 / sde.alpha) * e_L
            batch_time_step_s = batch_time_step_t
    if trajectory:
        return samples
    else:
        return x_s


def ode_score_update(score_model, sde, alpha, x_s, s, t, clamp=3, r=0.01, h=0.9, return_noise=False,order=0):
    if order==0:
     score_s = score_model(x_s, s)*torch.pow(sde.marginal_std(s),-1)[:,None,None,None]
    elif order==1:
        score_s = score_model(x_s, s) * torch.pow(sde.marginal_std(s), -1)[:, None, None, None]
        score_s_1 = score_model(x_s - h, s) * torch.pow(sde.marginal_std(s), -1)[:, None, None, None]
        score_s_2 = score_model(x_s + h, s) * torch.pow(sde.marginal_std(s), -1)[:, None, None, None]
    elif order==2:
        score_s = score_model(x_s, s) * torch.pow(sde.marginal_std(s), -1)[:, None, None, None]
        score_s_1 = score_model(x_s - h, s) * torch.pow(sde.marginal_std(s), -1)[:, None, None, None]
        score_s_2 = score_model(x_s + h, s) * torch.pow(sde.marginal_std(s), -1)[:, None, None, None]
        score_s_3 = score_model(x_s - 2*h, s) * torch.pow(sde.marginal_std(s), -1)[:, None, None, None]
        score_s_4 = score_model(x_s + 2*h, s) * torch.pow(sde.marginal_std(s), -1)[:, None, None, None]

    time_step = s - t
    beta_step = sde.beta(s) * time_step

    #x_coeff = 1 + beta_step / alpha
    x_coeff =sde.diffusion_coeff(t) * torch.pow(sde.diffusion_coeff(s), -1)
    lambda_s = sde.marginal_lambda(s)
    lambda_t = sde.marginal_lambda(t)

    time_step = s - t
    beta_step = sde.beta(s) * time_step

    h_t = lambda_t - lambda_s

    # lambda_s_1 = sde.marginal_lambda(s) + r * h_t
    # h_s_1= lambda_s_1-lambda_s
    # s_1 = sde.inverse_lambda(lambda_s_1)
    # x_coeff_1 = sde.diffusion_coeff(s_1) * torch.pow(sde.diffusion_coeff(s), -1)
    # score_coeff_1 = -gamma_func(alpha - 1) / gamma_func(alpha / 2) ** 2 / h ** (alpha - 2) * alpha * torch.pow( sde.marginal_std(s), alpha - 1) * sde.marginal_std(s_1) * (1 - torch.exp(h_s_1))
    # x_s_1 = x_coeff_1[:, None, None, None] * x_s + score_coeff_1[:, None, None, None] * score_s
    # score_s_1 = score_model(x_s_1, s_1) * torch.pow(sde.marginal_std(s_1), -1)[:, None, None, None]


    if alpha == 2:
        score_coeff = 1 / 2 * sde.diffusion_coeff(t) * torch.pow(sde.diffusion_coeff(s), -1) * beta_step * gamma_func(alpha + 1)

    else:
        #score_coeff = 1 / 2 * torch.pow(beta_step, 2 / alpha) * torch.pow(time_step, 1 - 2 / alpha) * np.sin(torch.pi / 2 * (2 - alpha)) / (2 - alpha) * 2 / torch.pi * gamma_func(alpha + 1)
        #score_coeff = 1 / 2 * sde.diffusion_coeff(t) * torch.pow(sde.diffusion_coeff(s)+1e-8, -1) * torch.pow(beta_step,2 / alpha) * torch.pow(time_step, 1 - 2 / alpha) * np.sin(torch.pi / 2 * (2 - alpha)) / (2 - alpha) * 2 / torch.pi * gamma_func(alpha + 1)
        #score_coeff = sigma_t * torch.pow(sde.beta(s) , alpha - 1) * alpha * torch.expm1(h_t) * gamma_func(alpha - 1) / torch.pow(gamma_func(alpha / 2), 2) / np.power(h, alpha - 2)
        #score_coeff = gamma_func(alpha-1)/gamma_func(alpha/2)**2/h**(alpha-2)*sde.beta(s)*time_step
        #score_coeff = gamma_func(alpha-1)/gamma_func(alpha/2)**2/h**(alpha-2)*alpha*(sde.diffusion_coeff(t) * torch.pow(sde.diffusion_coeff(s)+1e-8, -1)-1)
        if order==0:
            score_coeff=-gamma_func(alpha-1)/gamma_func(alpha/2)**2/h**(alpha-2)*alpha*torch.pow(sde.marginal_std(s),alpha-1)*sde.marginal_std(t)*(1-torch.exp(h_t))
            x_t = x_coeff[:, None, None, None] * x_s + score_coeff[:, None, None, None] * score_s
        if order==1:
            score_coeff = -gamma_func(alpha - 1) / gamma_func(alpha / 2) ** 2 / h ** (alpha - 2) * alpha * torch.pow(sde.marginal_std(s), alpha - 1) * sde.marginal_std(t) * (1 - torch.exp(h_t))
            score_coeff2 = gamma_func(alpha - 1) / gamma_func(alpha / 2 - 1) / gamma_func(alpha / 2 + 1) / h ** ( alpha - 2) * alpha * torch.pow(sde.marginal_std(s), alpha - 1) * sde.marginal_std(t) * (1 - torch.exp(h_t))
            x_t = x_coeff[:, None, None, None] * x_s + score_coeff[:, None, None, None] * score_s + score_coeff2[:, None, None,None] * (score_s_1 * (1 - h * score_s) + score_s_2 * (1 + h * score_s))
        if order==2:
            score_coeff = -gamma_func(alpha - 1) / gamma_func(alpha / 2) ** 2 / h ** (alpha - 2) * alpha * torch.pow( sde.marginal_std(s), alpha - 1) * sde.marginal_std(t) * (1 - torch.exp(h_t))
            score_coeff2 = gamma_func(alpha - 1) / gamma_func(alpha / 2 - 1) / gamma_func(alpha / 2 + 1) / h ** (alpha - 2) * alpha * torch.pow(sde.marginal_std(s), alpha - 1) * sde.marginal_std(t) * ( 1 - torch.exp(h_t))
            score_coeff3 = -gamma_func(alpha - 1) / gamma_func(alpha / 2 - 2) / gamma_func(alpha / 2 + 2) / h ** ( alpha - 2) * alpha * torch.pow(sde.marginal_std(s), alpha - 1) * sde.marginal_std(t) * (1 - torch.exp(h_t))
            x_t = x_coeff[:, None, None, None] * x_s + score_coeff[:, None, None, None] * score_s + score_coeff2[:,None, None,None] * ( score_s_1 * (1 - h * score_s) + score_s_2 * (1 + h * score_s)) + score_coeff3[:, None,None, None] * ( score_s_3 * (1 - 2 * h * score_s) + score_s_4 * (1 + 2 * h * score_s))

        #score_coeff2 = -gamma_func(alpha - 1) / gamma_func(alpha / 2) ** 2 / h ** (alpha - 2) * alpha * torch.pow(sde.marginal_std(s), alpha - 1) * sde.marginal_std(t) * (1 - torch.exp(h_t)-h_t)

    x_t = torch.clamp(x_t, -clamp,clamp)
    # print('x_s range', torch.min(x_s), torch.max(x_s))
    # print('x_t range', torch.min(x_t), torch.max(x_t))
    return x_t


def ode_sampler(score_model,
                sde,
                alpha,
                batch_size,
                num_steps,
                device='cuda',
                eps=1e-6,
                x_0=None,
                Predictor=True,
                Corrector=False, trajectory=False,
                clamp=10,
                initial_clamp=3, final_clamp=1,h=0.9,
                datasets="MNIST", clamp_mode='constant'):
    t = torch.ones(batch_size, device=device)
    if datasets == "MNIST":
        e_L = levy.sample(alpha, 0, (batch_size, 1, 28, 28)).to(device)
        x_s = torch.clamp(e_L, -initial_clamp, initial_clamp)
    elif datasets == "CIFAR10":
        e_L = levy.sample(alpha, 0, (batch_size, 3, 32, 32)).to(device)
        x_s = torch.clamp(e_L, -initial_clamp, initial_clamp)

    elif datasets == "CelebA":
        e_L = levy.sample(alpha, 0, (batch_size, 3, 64, 64)).to(device)
        x_s = torch.clamp(e_L, -initial_clamp, initial_clamp)

    if trajectory:
        samples = []
        samples.append(x_s)
    time_steps = torch.pow(torch.linspace(1., eps, num_steps),1)
   

    step_size = time_steps[0] - time_steps[1]

    batch_time_step_s = torch.ones(x_s.shape[0]) * time_steps[0]
    batch_time_step_s = batch_time_step_s.to(device)

    with torch.no_grad():
        for t in tqdm.tqdm(time_steps[1:]):
            batch_time_step_t = torch.ones(x_s.shape[0]) * t
            batch_time_step_t = batch_time_step_t.to(device)

            if clamp_mode == "constant":
                linear_clamp = clamp
            if clamp_mode == "linear":
                linear_clamp = batch_time_step_t[0] * (clamp - final_clamp) + final_clamp
            if clamp_mode == "root":
                linear_clamp = torch.pow(batch_time_step_t[0], 1 / 2) * (clamp - final_clamp) + final_clamp
            if clamp_mode == "quad":
                linear_clamp = batch_time_step_t[0] ** 2 * (clamp - final_clamp) + final_clamp

            x_s = ode_score_update(score_model, sde, alpha, x_s, batch_time_step_s, batch_time_step_t,h=h,
                                   clamp=linear_clamp)
            if trajectory:
                samples.append(x_s)
            batch_time_step_s = batch_time_step_t

    if trajectory:
        return samples
    else:
        return x_s


