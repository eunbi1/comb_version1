import torch
import numpy as np

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'




class VPSDE:
    def __init__(self, alpha_0, alpha_1, beta_min=0.1, beta_max=20, T=1., device=device):
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.alpha_0 = alpha_0
        self.alpha_1= alpha_1
        self.T = T

    def beta(self, t):
        return (self.beta_1 - self.beta_0) * t + self.beta_0
    def alpha(self,t):
        return  (self.alpha_1 - self.alpha_0) * t + self.alpha_0

    def marginal_log_mean_coeff(self, s,t):
        log_alpha_t = - 1 / (2 * self.alpha(t)) * ((t-s) ** 2) * (self.beta_1 - self.beta_0) - 1 / self.alpha(t) * (t-s) * self.beta_0
        return log_alpha_t

    def diffusion_coeff(self, s,t):
        return torch.exp(self.marginal_log_mean_coeff(s,t))

    def marginal_std(self, s,t):
        sigma = torch.pow(1. - torch.exp(self.alpha(t) * self.marginal_log_mean_coeff(s,t)), 1 / self.alpha(t))
        return sigma

    def marginal_lambda(self, s,t):
        log_mean_coeff = self.marginal_log_mean_coeff(s,t)
        log_sigma = torch.log(torch.pow(1. - torch.exp(self.alpha(t) * log_mean_coeff), 1 / self.alpha(t)))
        return log_mean_coeff - log_sigma
    
    def inverse_lambda(self,l):
        return (-self.beta_0+torch.pow(self.beta_0**2+2*(self.beta_1-self.beta_0)*torch.log(1+torch.exp(-l*self.alpha)),1/2))/(self.beta_1-self.beta_0)
