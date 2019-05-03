import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from base import BaseModel

EPSILON = 1e-30

class GraphVAE(BaseModel):
    def __init__(self, input_dim, n_nodes, node_dim):
        super(GraphVAE, self).__init__()
        # store parameters
        self.input_dim = input_dim
        self.n_nodes = n_nodes
        self.node_dim = node_dim

        # encoder: x -> h_x
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Linear(256, 128)
        )
        # bottom-up inference: predicts parameters of P(z_i | x)
        self.bottom_up = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 128),
                nn.BatchNorm1d(128),
                nn.ELU(),
                nn.Linear(128, node_dim),
                nn.Linear(node_dim, 2*node_dim) # split into mu and logvar
            )
        for _ in range(n_nodes-1)]) # ignore z_n

        # top-down inference: predicts parameters of P(z_i | Pa(z_i))
        self.top_down = nn.ModuleList([
            nn.Sequential(
                nn.Linear((n_nodes - i - 1)*node_dim, 128), # parents of z_i are z_{i+1} ... z_N
                nn.BatchNorm1d(128),
                nn.ELU(),
                nn.Linear(128, node_dim),
                nn.Linear(node_dim, 2*node_dim) # split into mu and logvar
            )
        for i in range(n_nodes-1)]) # ignore z_n

        # decoder: (z_1, z_2 ... z_n) -> parameters of P(x)
        self.decoder = nn.Sequential(
            nn.Linear(node_dim*n_nodes, 256),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ELU(),
            nn.Linear(512, input_dim)
        )

        # mean of Bernoulli variables c_{i,j} representing edges
        self.gating_params = nn.ParameterList([
            nn.Parameter(torch.empty(n_nodes - i - 1, 1, 1).fill_(0.5), requires_grad=True)
        for i in range(n_nodes-1)]) # ignore z_n

        # distributions for sampling
        self.unit_normal = D.Normal(torch.zeros(self.node_dim), torch.ones(self.node_dim))
        self.gumbel = D.Gumbel(0., 1.)

        # other parameters / distributions
        self.tau = 1.0

    def forward(self, x):
        # x: (batch_size, input_size)
        hx = self.encoder(x)

        # sample z_n from N(0, I)
        z_n = self.unit_normal.sample([x.size(0)]).to(x.device)
        parents = [z_n]
        mu_z = [torch.zeros(x.size(0), self.node_dim).to(x.device)]
        sigma_z = [torch.ones(x.size(0), self.node_dim).to(x.device)]

        for i in reversed(range(self.n_nodes-1)):
            self.gating_params[i].data = self.gating_params[i].data.clamp(0., 1.)
            # compute gating constants c_{i,j}
            mu = self.gating_params[i]
            eps1, eps2 = self.gumbel.sample(mu.size()).to(x.device), self.gumbel.sample(mu.size()).to(x.device)
            num = torch.exp((eps2 - eps1)/self.tau)
            t1 = torch.pow(mu, 1./self.tau)
            t2 = torch.pow((1.-mu), 1./self.tau)*num
            c = t1 / (t1 + t2 + EPSILON)
            # find concatenated parent vector
            parent_vector = (c * torch.stack(parents)).permute(1,0,2).reshape(x.size(0), -1)
            # top-down inference
            td = self.top_down[i](parent_vector)
            mu_td, sigma_td = td[:, :self.node_dim], F.softplus(td[:, self.node_dim:])
            # bottom-up inference
            bu = self.bottom_up[i](hx)
            mu_bu, sigma_bu = bu[:, :self.node_dim], F.softplus(bu[:, self.node_dim:])
            # precision weighted fusion
            mu_zi = (mu_td * sigma_bu**2 + mu_bu * sigma_td**2) / (sigma_td**2 + sigma_bu**2 + EPSILON)
            sigma_zi = (sigma_bu * sigma_td) / (torch.sqrt(sigma_td**2 + sigma_bu**2) + EPSILON)
            # sample z_i from P(z_i | pa(z_i), x)
            z_i = mu_zi + sigma_zi * self.unit_normal.sample([x.size(0)]).to(x.device)
            # store samples and parameters
            parents.append(z_i)
            mu_z.append(mu_zi)
            sigma_z.append(sigma_zi)

        # sample from approximate posterior distribution q(z_1, z_2 ... z_n|x)
        z = torch.cat(parents, dim=1)
        out = torch.sigmoid(self.decoder(z))

        # build output
        output = {}
        output['mu'] = out
        output['means'] = mu_z
        output['sigmas'] = sigma_z
        # output['gate_params'] = self.gating_params.detach()
        return output