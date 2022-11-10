import torch
import copy
import time
import numpy as np
import tqdm
from scipy.special import gamma
import torchlevy
from torchlevy import LevyStable
import torch.nn.functional as F

levy = LevyStable()

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

def gamma_func(x):
    return torch.tensor(gamma(x))

from torchlevy.approx_score import get_approx_score

from torchlevy.approx_score import get_approx_score




def loss_fn(model, sde,
            x0: torch.Tensor,
            t: torch.LongTensor,
            training_clamp=100,
            num_steps=1000, type="cft", mode='approximation'):

    X=[]
    P=[]


    for i, _ in enumerate(t):
        x_n_1 = x0[i]
        t_n_1 = 0

        score = torch.ones_like(x_n_1)

        for j,_ in enumerate(t[:i]):
            if j==0:
                t_n_1 =0
            t_n = t[j]
            e_L = torch.clamp(levy.sample(sde.alpha(t_n), 0, size=x_n_1.shape).to(device), -training_clamp, training_clamp)
            x_n=sde.diffusion_coeff(t_n_1,t_n)*x_n_1+sde.marginal_std(t_n_1,t_n)*e_L
            score+=get_approx_score(e_L, sde.alpha(t_n).item())/(sde.marginal_std(t_n_1, t_n) +1e-4)

            #score = get_approx_score(e_L, sde.alpha(t_n).item()) / (sde.marginal_std(t_n_1, t_n) + 1e-4)

            x_n_1 = x_n
            t_n_1 = t_n


        t_n = t[i]
        e_L = torch.clamp(levy.sample(sde.alpha(t_n), 0, size=x_n_1.shape).to(device), -training_clamp, training_clamp)
        x_n = sde.diffusion_coeff(t_n_1, t_n) * x_n_1 + sde.marginal_std(t_n_1, t_n) * e_L
        score+=get_approx_score(e_L, sde.alpha(t_n).item())/(sde.marginal_std(t_n_1, t_n) +1e-4)
        X.append(x_n)
        P.append(score)

    x_t = torch.stack(X)
    p_alpha = torch.stack(P)




    output = model(x_t, t)
    # loss = torch.abs(weight).sum(dim=(1,2,3)).mean(dim=0)
    #
    #print('x_t', torch.min(x_t), torch.max(x_t))
    #print('e_L', torch.min(e_L),torch.max(e_L))
    #print('score', torch.min(score), torch.max(score))
    #print('output', torch.min(model(x_t, t)), torch.max(model(x_t, t)))
    #print('output*beta',torch.min(output), torch.max(output))
    #loss = F.smooth_l1_loss(output, score, size_average=False,reduce=True, beta=4.0)
    weight = output-p_alpha
    # print('output', torch.min(output), torch.max(output))
    # print('p_alpha', torch.min(p_alpha), torch.max(p_alpha))
    loss = (weight).square().sum(dim=(1, 2, 3)).mean(dim=0)

    return  loss
