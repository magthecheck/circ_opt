import torch
from torch_geometric.data import Data, Batch
from rl_agent import AgentGNN
from mock_env import MockColorEnv  # Assuming your mock env code is saved as mock_env.py

device = torch.device("cpu")
print(f"Using device {device}")

# 1. Initialize the Environment
# We choose 10 nodes. Our environment will generate structural graphs dynamically.
env = MockColorEnv(num_nodes=10)
graph_state, info = env.reset(seed=42)

# 2. Extract structural metrics from our new environment's info tracker
# Each step or reset provides a dictionary containing raw graph features
raw_policy_obs, raw_value_obs = info["graph_obs"]
policy_x, policy_edge_index = raw_policy_obs
value_x, value_edge_index = raw_value_obs

# 3. Adapt node features to match the Agent's expected GNN dimensionalities
# The Agent expects c_in_p = 17 and c_in_v = 12. 
# Our mock env provides 3 features (one-hot color). We pad the remaining positions with zeros.
padded_policy_x = torch.cat([policy_x, torch.zeros(policy_x.size(0), 17 - policy_x.size(1))], dim=1)
padded_value_x = torch.cat([value_x, torch.zeros(value_x.size(0), 12 - value_x.size(1))], dim=1)

# 4. Generate Edge Attributes & Valid Action Masking (y tensor)
# The agent's actor demands 7 edge features; the critic demands 2.
policy_edges = torch.zeros(policy_edge_index.size(1), 7)
critic_edges = torch.zeros(value_edge_index.size(1), 2)

# Generate pseudo-action mappings for your nodes. 
# y maps each node/action node index to a unique environment action ID.
# Non-action internal nodes are masked out using -1.
action_ids = torch.tensor([i for i in range(env.num_nodes * 2)] + 
                          [-1] * (padded_policy_x.size(0) - (env.num_nodes * 2)), dtype=torch.long)

# 5. Build PyTorch Geometric Graph Structures
policy_graph = Data(x=padded_policy_x, edge_index=policy_edge_index, edge_attr=policy_edges, y=action_ids)
critic_graph = Data(x=padded_value_x, edge_index=value_edge_index, edge_attr=critic_edges)

# 6. Collate single instances into Batches (Expected format for GATv2Conv layers)
policy_batch = Batch.from_data_list([policy_graph]).to(device)
critic_batch = Batch.from_data_list([critic_graph]).to(device)
network_input = (policy_batch, critic_batch)

# 7. Initialize AgentGNN Topology
agent = AgentGNN(envs=None, device=device, c_hidden=32, c_hidden_v=32).to(device)

print("\n--- Testing forward pass with MockColorEnv Data ---")
with torch.no_grad():
    action, action_id = agent.get_action(network_input, device="cpu")

print("Success! The data flowed through the network without errors.")
print(f"Sampled Action Index (0-{env.num_nodes*2-1}): {action.item()}")
print(f"Corresponding Action ID mapped to Environment: {action_id.item()}")