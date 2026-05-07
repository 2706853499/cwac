import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Union, Tuple, Optional

import numpy as np
from algos.dist_module import *
 
def huber(x, min_priority=1):
	return torch.where(x < min_priority, 0.5 * x.pow(2), min_priority * x)
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


 
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu_head_1 = nn.Linear(hidden_dim, 1)
        self.log_std_head_1 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.mu_head_2 = nn.Linear(hidden_dim, 1)
        self.log_std_head_2 = nn.Linear(hidden_dim, 1)
        
    
    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1_fea = F.relu(self.l2(q1))
        q1 = self.mu_head_1(q1_fea)
        log_std_1 = self.log_std_head_1(q1_fea) 
        log_std_1 = F.softplus(log_std_1)

        q2 = F.relu(self.l4(sa))
        q2_fea = F.relu(self.l5(q2))
        q2 = self.mu_head_2(q2_fea)
        log_std_2 = self.log_std_head_2(q2_fea) 
        log_std_2 = F.softplus(log_std_2)
        return torch.cat([q1, q2],dim=1),torch.cat([log_std_1  ,log_std_2  ],dim=1)

 
    def get_Q(self, x,action,mu=1.0,determined=False):
        mean, log_std = self(x ,action)
        if mu==0:
            return mean, mean, log_std
        normal = torch.distributions.Normal(torch.zeros_like(mean), mu*torch.ones_like(log_std))
        z = normal.sample()
        z = torch.clamp(abs(z), 0, 3)
        if determined:
            x_t = mean
        else:
            x_t = mean- torch.mul(z, log_std)  # for reparameterization trick (mean + std * N(0,1))
        return mean, x_t, log_std


 
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Union[List[int], Tuple[int]],
        output_dim: Optional[int] = None,
        activation: nn.Module = nn.ReLU,
        dropout_rate: Optional[float] = None
    ) -> None:
        super().__init__()
        hidden_dims = [input_dim] + list(hidden_dims)
        model = []
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            model +=[nn.Linear(in_dim, out_dim), activation()]#  [ELinear(in_dim, out_dim,dropout=dropout_rate), activation()]#[nn.Linear(in_dim, out_dim), activation()]
            if dropout_rate is not None:
                model += [nn.Dropout(p=dropout_rate)]

        self.output_dim = hidden_dims[-1]
        if output_dim is not None:
            model += [nn.Linear(hidden_dims[-1], output_dim)]# [ELinear(hidden_dims[-1], output_dim,dropout=dropout_rate)]#[nn.Linear(hidden_dims[-1], output_dim)]
            self.output_dim = output_dim
        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
 
 
class ActorProb(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        dist_net: nn.Module,
        device: str = "cpu"
    ) -> None:
        super().__init__()

        self.device = torch.device(device)
        self.backbone = backbone.to(device)
        self.dist_net = dist_net.to(device)

    def forward(self, obs: Union[np.ndarray, torch.Tensor]) -> torch.distributions.Normal:
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        logits = self.backbone(obs)
        dist = self.dist_net(logits)
        return dist
 
 
class CWAC(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            discount=0.99,
            lamda=0.005,
            device='cuda:0',
            mu=0.8,
            beta_omega=2.0,
            beta_xi=1.0
    ):
        self.mu = mu
        self.beta_omega = beta_omega
        self.beta_xi = beta_xi
        self.device = device
        actor_backbone = MLP(input_dim=state_dim, hidden_dims=[256, 256])
        dist = TanhDiagGaussian(
            latent_dim=getattr(actor_backbone, "output_dim"),
            output_dim=action_dim,
            unbounded=True,
            conditioned_sigma=True,
            max_mu=1.0
        )
        self.actor = ActorProb(actor_backbone, dist, self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.max_action = max_action
        self.discount = discount
        self.lamda = lamda
 
        
        self.total_it = 0
        self._is_auto_alpha = True
        self._target_entropy  =  -self.action_dim
        self._log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim  = torch.optim.Adam([self._log_alpha], lr=3e-4)
        self._alpha = self._log_alpha.detach().exp()
        self.gamma = 1.0
 
        self.action_dim = action_dim
        self.state_dim = state_dim
 
      
    def actforward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
       
        dist = self.actor(obs)
        if deterministic:
            squashed_action, raw_action = dist.mode()
        else:
            squashed_action, raw_action = dist.rsample()
        log_prob = dist.log_prob(squashed_action, raw_action)
        return squashed_action, log_prob
 
     
    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False
    ) -> np.ndarray:
        with torch.no_grad():
            obs = torch.FloatTensor(obs.reshape(1, -1)).to(self.device)
            action, _ = self.actforward(obs, deterministic)
        return action.cpu().data.numpy().flatten()
    
    
 
  
    def train(self, replay_buffer, batch_size=256):
 
        self.total_it += 1
        state, action, next_state, reward, not_done = replay_buffer.sample()
        with torch.no_grad():
            next_actions, next_log_probs =  self.actforward(next_state,deterministic=False )
            _, target_Qs,_ = self.critic_target.get_Q(next_state, next_actions,self.mu)
            target_Q = torch.min(target_Qs,dim=1,keepdim=True).values  - self._alpha * next_log_probs 
            target_Q = reward + not_done * self.discount * target_Q
         
        # Get current Q estimates
        current_Q ,q_std = self.critic(state, action)
        bias = 0.1
        TD_error = abs(current_Q-target_Q)
        omega = (torch.pow(q_std.mean(axis=0), self.beta_omega) / (torch.pow(q_std, self.beta_omega) + bias)).clamp(min=0.1, max=10)
        target_dist = huber(TD_error)
        xi = (torch.pow(TD_error.mean(dim=0), self.beta_xi ) / (torch.pow(TD_error, self.beta_xi ) + bias)).clamp(0.1, 10.0)
        critic_loss = (omega.detach()*target_dist) .mean()+(xi.detach()*q_std) .mean()  
        

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
 
        
        a,log_probs = self.actforward(state)
        _,qs,q_std = self.critic.get_Q(state, a,1.0)
        actor_loss =- qs.mean()   + self._alpha * log_probs.mean()
        
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self._is_auto_alpha:
            log_probs = log_probs.detach() + self._target_entropy
            alpha_loss = -(self._log_alpha * log_probs).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self._alpha = torch.clamp(self._log_alpha.detach().exp(), 0.0, 1.0)
         
        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.lamda * param.data + (1 - self.lamda) * target_param.data)
       
   
    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        

 