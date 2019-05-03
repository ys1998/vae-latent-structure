import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from base import BaseModel

EPSILON = 1e-30

class RecurrentGraphVAE(BaseModel):
    def __init__(self, input_dim, n_nodes):
        super(RecurrentGraphVAE, self).__init__()
        # store parameters
        self.input_dim = input_dim
        self.n_nodes = n_nodes

        # encoder: x -> h_x
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128)
        )
        # bottom-up inference: predicts parameters of P(z_i | x)
        self.bottom_up = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 128),
                nn.ELU(),
                nn.Linear(128, 2) # split into mu and sigma
            )
        for _ in range(n_nodes-1)]) # ignore z_n

        # top-down inference: predicts parameters of P(z_i | Pa(z_i))
        self.top_down = nn.ModuleList([
            nn.Sequential(
                nn.Linear((n_nodes - i - 1) + n_nodes, 128), # possible parents of z_i are z_{i+1} ... z_N for
                nn.ELU(),                                    # timestep t and z_1, z_2 .... z_N for timestep t-1
                nn.Linear(128, 2) # split into mu and logvar
            )
        for i in range(n_nodes-1)]) # ignore z_n

        # decoder: (z_1, z_2 ... z_n) -> parameters of P(x)
        self.decoder = nn.Sequential(
            nn.Linear(n_nodes, 256),
            nn.ELU(),
            nn.Linear(256, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, 2*(input_dim-1) + 1)
        )

        # mean of Bernoulli variables c_{i,j} representing edges
        # connections between z_i for the same timestep
        self.self_gating_params = nn.ParameterList([
            nn.Parameter(torch.empty(n_nodes - i - 1, 1, 1).fill_(0.5), requires_grad=True)
        for i in range(n_nodes-1)]) # ignore z_n
        # connections between z_i of timesteps t-1 and t
        self.recurrent_gating_params = nn.ParameterList([
            nn.Parameter(torch.empty(1, n_nodes).fill_(0.5), requires_grad=True)
        for i in range(n_nodes-1)]) # ignore z_n

        # distributions for sampling
        self.unit_normal = D.Normal(0., 1.)
        self.gumbel = D.Gumbel(0., 1.)

        # other parameters / distributions
        self.tau = 1.0

    def forward(self, x, init_z):
        # x: (seq_len, batch_size, input_size)
        T, N = x.size(0), x.size(1)
        mu_z_t = []
        sigma_z_t = []
        out_t = []
        prev_z = init_z

        for t in range(T):
            hx = self.encoder(x[t])
            # sample z_n from N(0, I)
            z_n = self.unit_normal.sample([N, 1]).to(x.device)
            parents = [z_n]
            mu_z = [torch.zeros(N, 1).to(x.device)]
            sigma_z = [torch.ones(N, 1).to(x.device)]

            for i in reversed(range(self.n_nodes-1)):
                """ compute gating constants c_{i,j} """
                
                # between z_i of same timestep
                self.self_gating_params[i].data = self.self_gating_params[i].data.clamp(0., 1.)
                mu1 = self.self_gating_params[i]
                eps1, eps2 = self.gumbel.sample(mu1.size()).to(x.device), self.gumbel.sample(mu1.size()).to(x.device)
                num1 = torch.exp(eps2 - eps1)
                t1 = torch.pow(mu1, 1./self.tau)
                t2 = torch.pow((1.-mu1)*num1, 1./self.tau)
                c1 = t1 / (t1 + t2 + EPSILON)
                
                # between z_i of timesteps t-1 and t
                self.recurrent_gating_params[i].data = self.recurrent_gating_params[i].data.clamp(0., 1.)
                mu2 = self.recurrent_gating_params[i]
                eps3, eps4 = self.gumbel.sample(mu2.size()).to(x.device), self.gumbel.sample(mu2.size()).to(x.device)
                num2 = torch.exp(eps4 - eps3)
                t3 = torch.pow(mu2, 1./self.tau)
                t4 = torch.pow((1.-mu2)*num2, 1./self.tau)
                c2 = t3 / (t3 + t4 + EPSILON)

                """ find concatenated parent vector """
                parent_vector = (c1 * torch.stack(parents)).permute(1,0,2).reshape(N, -1)
                parent_vector = torch.cat([parent_vector, c2 * prev_z], dim=1)

                """ top-down inference """
                td = self.top_down[i](parent_vector)
                mu_td, sigma_td = td[:, 0].unsqueeze(1), F.softplus(td[:, 1]).unsqueeze(1)

                """ bottom-up inference """
                bu = self.bottom_up[i](hx)
                mu_bu, sigma_bu = bu[:, 0].unsqueeze(1), F.softplus(bu[:, 1]).unsqueeze(1)
                
                """ precision weighted fusion """
                mu_zi = (mu_td * sigma_bu**2 + mu_bu * sigma_td**2) / (sigma_td**2 + sigma_bu**2 + EPSILON)
                sigma_zi = (sigma_bu * sigma_td) / (torch.sqrt(sigma_td**2 + sigma_bu**2) + EPSILON)
                
                """ sample z_i from P(z_i | pa(z_i), x) """
                z_i = mu_zi + sigma_zi * self.unit_normal.sample([N, 1]).to(x.device)
                
                """ store samples and parameters """
                parents.append(z_i)
                mu_z.append(mu_zi)
                sigma_z.append(sigma_zi)

            """ sample from approximate posterior distribution q(z_1, z_2 ... z_n|x) """
            z = torch.cat(parents, dim=1)
            out = self.decoder(z) # change this as required

            """ store variables """
            prev_z = z
            mu_z_t.append(mu_z)
            sigma_z_t.append(sigma_z)
            out_t.append(out)

        """ build output """
        output = {}
        output['out'] = torch.stack(out_t)
        output['means'] = mu_z_t
        output['sigmas'] = sigma_z_t
        return output, prev_z