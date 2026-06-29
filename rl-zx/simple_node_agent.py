import torch
import torch.nn as nn
from torch.distributions import Categorical

class SimpleNodeAgent(nn.Module):
    def __init__(self, num_nodes=10):
        super().__init__()
        self.num_nodes = num_nodes  # Hard boundary for your environment graph size (e.g., 10)
        
        # 1. Feature extractor for the ACTOR (expects 17 features)
        self.actor_feature_extractor = nn.Sequential(
            nn.Linear(17, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.actor_head = nn.Linear(16, 2)  # 2 actions: Up, Down per node
        
        # 2. Feature extractor for the CRITIC (expects 12 features)
        self.critic_feature_extractor = nn.Sequential(
            nn.Linear(12, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.critic_head = nn.Linear(16, 1)

    def _unpack_input(self, network_input):
        if isinstance(network_input, tuple):
            base_obj = network_input[0]
            return base_obj.x if hasattr(base_obj, 'x') else base_obj
        elif hasattr(network_input, 'x'):
            return network_input.x
        return network_input

    def actor(self, network_input):
        x = self._unpack_input(network_input)
        node_features = self.actor_feature_extractor(x)
        logits = self.actor_head(node_features)
        
        # SAFETY SLICE: Only take the first 'num_nodes' entries to prevent environment overflow
        valid_logits = logits[:self.num_nodes, :]
        return valid_logits.reshape(-1)

    def get_value(self, network_input, edge_index=None):
        x = self._unpack_input(network_input)
        node_features = self.critic_feature_extractor(x)
        # Only evaluate the value over the true environment graph nodes
        valid_features = node_features[:self.num_nodes, :]
        return self.critic_head(valid_features).sum(dim=0, keepdim=True)

    def get_action_and_value(self, network_input, edge_index=None, action=None):
        x = self._unpack_input(network_input)
        
        if x.shape[-1] == 17:
            node_features = self.actor_feature_extractor(x)
        else:
            node_features = self.critic_feature_extractor(x)
            
        logits = self.actor_head(node_features) 
        
        # SAFETY SLICE: Enforce graph size boundary
        valid_logits = logits[:self.num_nodes, :]
        flat_logits = valid_logits.reshape(-1) 
        
        dist = Categorical(logits=flat_logits)
        if action is None:
            action = dist.sample()
            
        if x.shape[-1] == 12:
            val_features = self.critic_feature_extractor(x)
        else:
            val_features = self.actor_feature_extractor(x) if x.shape[-1] == 17 else node_features
            
        valid_val_features = val_features[:self.num_nodes, :]
        value = self.critic_head(valid_val_features).sum(dim=0, keepdim=True)
        return action, dist.log_prob(action).unsqueeze(0), dist.entropy(), value

    def get_action(self, network_input, device=None):
        x = self._unpack_input(network_input)

        if device is not None:
            x = x.to(device)
            
        node_features = self.actor_feature_extractor(x)
        logits = self.actor_head(node_features)
        
        # SAFETY SLICE: Enforce graph size boundary
        valid_logits = logits[:self.num_nodes, :]
        flat_logits = valid_logits.reshape(-1)
        
        dist = Categorical(logits=flat_logits)
        action = dist.sample()
        return action, action