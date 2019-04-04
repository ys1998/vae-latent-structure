import torch
import torch.nn as nn
import torch.distributions as D
from base import BaseModel

EPSILON = 1e-8

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
            nn.Parameter(torch.empty(n_nodes - i - 1, 1, 1).uniform_())
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
        z_n = self.unit_normal.sample([x.size(0)])
        parents = [z_n]
        mu_z = [torch.zeros(x.size(0), self.node_dim)]
        sigma_z = [torch.ones(x.size(0), self.node_dim)]

        for i in reversed(range(self.n_nodes-1)):
            # compute gating constants c_{i,j}
            mu = self.gating_params[i]
            eps1, eps2 = self.gumbel.sample(mu.size()), self.gumbel.sample(mu.size())
            c = \
            torch.exp((torch.log(mu + EPSILON) + eps1)/self.tau) / \
                ( torch.exp((torch.log(mu + EPSILON) + eps1)/self.tau) + \
                  torch.exp((torch.log(1.-mu + EPSILON) + eps2)/self.tau))
            # find concatenated parent vector
            parent_vector = (c * torch.stack(parents)).permute(1,0,2).reshape(x.size(0), -1)
            # top-down inference
            td = self.top_down[i](parent_vector)
            mu_td, logvar_td = td[:, :self.node_dim], td[:, self.node_dim:]
            # bottom-up inference
            bu = self.bottom_up[i](hx)
            mu_bu, logvar_bu = bu[:, :self.node_dim], bu[:, self.node_dim:]
            # precision weighted fusion
            mu_zi = (mu_td * torch.exp(logvar_bu) + mu_bu * torch.exp(logvar_td)) / (torch.exp(logvar_td) + torch.exp(logvar_bu) + EPSILON)
            sigma_zi = (torch.exp(.5*logvar_bu) * torch.exp(.5*logvar_td)) / torch.sqrt(torch.exp(logvar_bu) + torch.exp(logvar_td) + EPSILON)
            # sample z_i from P(z_i | pa(z_i), x)
            z_i = mu_zi + sigma_zi * self.unit_normal.sample([x.size(0)])
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
        output['logvars'] = [2*torch.log(v + EPSILON) for v in sigma_z]
        return output