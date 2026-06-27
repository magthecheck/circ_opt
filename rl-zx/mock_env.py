import random
import torch
import networkx as nx
import gymnasium as gym
from gymnasium.spaces import Discrete, Graph, Box

class MockColorEnv(gym.Env):
    def __init__(self, num_nodes=10, max_episode_len=20):
        super().__init__()
        self.num_nodes = num_nodes
        self.max_episode_len = max_episode_len
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Define standard color states
        self.BLUE = 1
        self.PURPLE = 2
        self.GREEN = 3
        
        # Gym interface requirements
        self.action_space = Discrete(self.num_nodes * 2) # e.g., node_id + direction
        self.observation_space = Graph(
            node_space=Box(low=0, high=1, shape=(3,)), # 3-dimensional One-Hot color vector
            edge_space=Discrete(1)
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_len = 0
        
        # Generate a random initial structural graph
        self.graph = nx.erdos_renyi_graph(n=self.num_nodes, p=0.4, seed=seed)
        
        # Assign random initial colors (1, 2, or 3) to the nodes
        for node in self.graph.nodes:
            self.graph.nodes[node]['color'] = random.choice([self.BLUE, self.PURPLE, self.GREEN])
            
        return self.graph, {"graph_obs": [self.policy_obs(), self.value_obs()]}

    def step(self, action):
        self.episode_len += 1
        
        # Decode the flat action into (target_node, transformation_direction)
        # Direction 0: shift up (Blue->Purple or Purple->Green)
        # Direction 1: shift down (Green->Purple or Purple->Blue)
        node = action // 2
        direction = action % 2
        
        current_color = self.graph.nodes[node]['color']
        new_color = current_color
        is_legal_move = False
        
        # Enforce transition boundaries: 1 <-> 2 <-> 3
        if direction == 0: # Moving Up
            if current_color == self.BLUE:
                new_color = self.PURPLE
                is_legal_move = True
            elif current_color == self.PURPLE:
                new_color = self.GREEN
                is_legal_move = True
        else: # Moving Down
            if current_color == self.GREEN:
                new_color = self.PURPLE
                is_legal_move = True
            elif current_color == self.PURPLE:
                new_color = self.BLUE
                is_legal_move = True
                
        # Apply transition if rules are met
        if is_legal_move:
            self.graph.nodes[node]['color'] = new_color
            reward = 1.0  # Reward for successful strategy execution
        else:
            reward = -0.5 # Penalty for trying an invalid transition
            
        # Determine termination criteria
        done = self.episode_len >= self.max_episode_len
        truncated = False 


        # Calculate terminal bonus if the episode is finished
        if done: 
            # Count how many nodes successfully reached the GREEN state
            green_count = sum(1 for node in self.graph.nodes if self.graph.nodes[node]['color'] == self.GREEN)

            # Weight factor (e.g. +2.0 per green node achieved) 
            terminal_bonus = green_count * 2.0
            reward += terminal_bonus
        
        info = {
            "node_changed": node,
            "color_before": current_color,
            "color_after": new_color,
            "graph_obs": [self.policy_obs(), self.value_obs()]
        }
        
        return self.graph, reward, done, truncated, info

    def policy_obs(self):
        """Generates GNN policy tensors resembling ZXEnv's logic."""
        node_features = []
        for node in sorted(self.graph.nodes):
            color = self.graph.nodes[node]['color']
            # Convert color integer to an explicit One-Hot vector [Blue, Purple, Green]
            oh_feature = [1.0 if color == i else 0.0 for i in [self.BLUE, self.PURPLE, self.GREEN]]
            node_features.append(oh_feature)
            
        # Extract edge list from NetworkX structure
        edge_list = list(self.graph.edges)
        # Convert to bidirectional edges to support spatial Graph Neural Networks
        for u, v in list(edge_list):
            edge_list.append((v, u))
            
        x = torch.tensor(node_features, dtype=torch.float32).to(self.device)
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous().to(self.device)
        
        return x, edge_index

    def value_obs(self):
        """Simplified downstream critic state monitoring."""
        # For a mock setup, value network features can mimic policy features
        return self.policy_obs()