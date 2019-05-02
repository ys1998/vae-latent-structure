import torch

EPSILON = 1e-30

def log_likelihood_bernoulli(x, mu):
	one_mask = (x == 1.).type(torch.float)
	zero_mask = (x == 0.).type(torch.float)
	return torch.sum(torch.log(mu + EPSILON)*one_mask + torch.log(1.-mu + EPSILON)*zero_mask, dim=1).mean()

def kl_divergence_normal(mu, sigma):
	return torch.sum((sigma**2 + mu**2)/2. - torch.log(sigma + EPSILON) - 0.5, dim=1).mean()

def loss_MNIST(output, target):
	mu = output['mu']
	l1 = log_likelihood_bernoulli(target, mu)
	l2 = 0.0
	for m,sigma in zip(output['means'], output['sigmas']):
		l2 += kl_divergence_normal(m, sigma)
	return l2, -l1