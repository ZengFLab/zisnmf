import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist


def zip_nll(x, pi, lambda_):
    eps = 1e-10
    pois = dist.Poisson(lambda_)
    log_pois = pois.log_prob(x)
    zero_pois = pois.log_prob(torch.zeros_like(x))
    zero_case = torch.log(pi + torch.exp(zero_pois + torch.log(1.0 - pi + eps)))
    non_zero_case = torch.log(1.0 - pi + eps) + log_pois
    result = torch.where(x < eps, zero_case, non_zero_case)
    return -result.sum()

class NB(nn.Module):
    def __init__(self, n_genes, hidden_dims=[64]):
        super(NB, self).__init__()
        
        # Define the layers
        self.layers = nn.ModuleList()
        input_dim = n_genes
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            input_dim = hidden_dim
        
        self.fc_mu = nn.Linear(input_dim, n_genes)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        mu = torch.exp(self.fc_mu(x))
        return mu

def zinb_nll(x, mu, theta, pi):
    eps = 1e-10
    nb = dist.NegativeBinomial(total_count=theta, logits=torch.log(mu + eps))
    log_nb = nb.log_prob(x)
    zero_nb = nb.log_prob(torch.zeros_like(x))
    zero_case = torch.log(pi + torch.exp(zero_nb + torch.log(1.0 - pi + eps)))
    non_zero_case = torch.log(1.0 - pi + eps) + log_nb
    result = torch.where(x < eps, zero_case, non_zero_case)
    return -result.sum()


def zig_nll(x, pi, mu, sigma):
    """
    Compute the negative log-likelihood of a zero-inflated Gaussian distribution.
    
    Parameters:
    x (Tensor): Observed data.
    pi (float): Probability of zero inflation.
    mu (float): Mean of the Gaussian distribution.
    sigma (float): Standard deviation of the Gaussian distribution.
    
    Returns:
    Tensor: Negative log-likelihood.
    """
    eps = 1e-8
    # Compute Gaussian log-probability
    gauss_log_prob = -0.5 * ((x - mu) / sigma) ** 2 - torch.log(sigma + eps) - 0.5 * torch.log(torch.tensor(2 * torch.pi))
    
    # Compute zero-inflated log-probability
    zero_inflated_log_prob = torch.log(pi + eps) + torch.log(1 - pi + eps) + gauss_log_prob
    #nll = -zero_inflated_log_prob

    # Compute non-zero log-probability for each feature
    non_zero_log_prob = torch.log(1 - pi + eps) + gauss_log_prob
    
    # Combine probabilities for each feature
    nll = torch.where(x == 0, -zero_inflated_log_prob, -non_zero_log_prob)
    
    return nll.sum()


def gaussian_nll(x, mu, sigma):
    """
    Compute the negative log-likelihood of a Gaussian distribution.
    
    Parameters:
    x (Tensor): Observed data.
    mu (float): Mean of the Gaussian distribution.
    sigma (float): Standard deviation of the Gaussian distribution.
    
    Returns:
    Tensor: Negative log-likelihood.
    """
    eps = 1e-8
    # Compute Gaussian log-probability
    gauss_log_prob = -0.5 * ((x - mu) / sigma) ** 2 - torch.log(sigma + eps) - 0.5 * torch.log(torch.tensor(2 * torch.pi))
    
    # Combine probabilities for each feature
    nll = gauss_log_prob
    
    return nll.sum()



class DropoutRate(nn.Module):
    def __init__(self, n_genes, hidden_dims=[64]):
        super(DropoutRate, self).__init__()
        
        # Define the layers
        self.layers = nn.ModuleList()
        input_dim = n_genes
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            input_dim = hidden_dim

        self.fc_pi = nn.Linear(input_dim, n_genes)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        pi = torch.sigmoid(self.fc_pi(x))
        return pi