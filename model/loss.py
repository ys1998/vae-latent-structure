import torch

def log_likelihood_bernoulli(x, mu):
	one_mask = (x == 1.).type(torch.float)
	zero_mask = (x == 0.).type(torch.float)
	return torch.sum(torch.log(mu) * one_mask + torch.log(1-mu)*zero_mask, dim=1).mean()

def kl_divergence_normal(mean, logvar):
	return torch.sum(0.5 * (torch.exp(logvar) + mean**2 - logvar - 1), dim=1).mean()

def loss_MNIST(output, target):
	mu = output['mu']
	l1 = log_likelihood_bernoulli(target, mu)
	l2 = 0.0
	for m,lv in zip(output['means'], output['logvars']):
		l2 += kl_divergence_normal(m, lv)
	return l2 - l1