import torch
import numpy as np

EPSILON = 1e-30

def log_likelihood_bernoulli(x, mu):
	one_mask = (x == 1.).type(torch.float)
	zero_mask = (x == 0.).type(torch.float)
	return torch.sum(torch.log(mu + EPSILON)*one_mask + torch.log(1.-mu + EPSILON)*zero_mask, dim=1).mean()

def log_likelihood_gauss(x, mu, logvar):
	const = np.log(2*np.pi)
	var = torch.exp(logvar)
	return -0.5 * torch.sum(const + logvar + (x - mu)**2 / (var + EPSILON), dim=1).mean()

def kl_divergence_normal(mu, sigma):
	return torch.sum((sigma**2 + mu**2)/2. - torch.log(sigma + EPSILON) - 0.5, dim=1).mean()

def loss_MNIST(output, target):
	mu = output['mu']
	l1 = log_likelihood_bernoulli(target, mu)
	l2 = 0.0
	for m,sigma in zip(output['means'], output['sigmas']):
		l2 += kl_divergence_normal(m, sigma)
	return l2, -l1

def loss_handwriting(output, target):
	out = output['out'] # (T, N, :)
	size = (out.size(2)-1) // 2
	e, mu, logvar = out[:, :, 0], out[:, :, 1:size+1], out[:, :, size+1:]
	l1 = - sum([log_likelihood_gauss(target[t,:,1:], mu[t], logvar[t]) for t in range(out.size(0))])
	l1 += - sum([log_likelihood_bernoulli(target[t,:,0].unsqueeze(1), e[t, :].unsqueeze(1)) for t in range(out.size(0))])
	l2 = 0.0
	for mu_t, sigma_t in zip(output['means'], output['sigmas']):
		for m, sigma in zip(mu_t, sigma_t):
			l2 += kl_divergence_normal(m, sigma)
	return l2, l1