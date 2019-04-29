import torch

def log_likelihood_bernoulli(x, mu):
	one_mask = (x == 1.).type(torch.float)
	zero_mask = (x == 0.).type(torch.float)
	return torch.sum(torch.log(mu + 1e-8) * one_mask + torch.log(1-mu + 1e-8)*zero_mask, dim=1).mean()

def kl_divergence_normal(mu, sigma):
	return torch.sum((torch.pow(sigma,2) + torch.pow(mu,2))/2. - torch.log(sigma + 1e-8) - 0.5, dim=1).mean()

def loss_MNIST(output, target):
	mu = output['mu']
	l1 = log_likelihood_bernoulli(target, mu)
	l2 = 0.0
	for m,sigma in zip(output['means'], output['sigmas']):
		l2 += kl_divergence_normal(m, sigma)
	return l2, -l1