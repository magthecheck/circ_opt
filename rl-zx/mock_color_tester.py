import torch
from torch_geometric.data import Data, Batch
from rl_agent import AgentGNN
from mock_env import MockColorEnv  # Assuming your mock env code is saved as mock_env.py

device = torch.device("cpu")
print(f"Using device {device}")

# 1. Initialize the Environment
env = MockColorEnv(num_nodes=10)
graph_state, info = env.reset(seed=42)



# --- VISUALIZE INITIAL STATE ---
print("\n=== Initial Graph State ===")
print(f"Graph Nodes: {list(env.graph.nodes)}")
print(f"Graph Edges: {list(env.graph.edges)}")
print("Node Colors (Initial):")
color_map = {1: "Blue (1)", 2: "Purple (2)", 3: "Green (3)"}
for node in sorted(env.graph.nodes):
    print(f"  Node {node}: {color_map[env.graph.nodes[node]['color']]}")

# 2. Extract structural metrics from our new environment's info tracker
raw_policy_obs, raw_value_obs = info["graph_obs"]
policy_x, policy_edge_index = raw_policy_obs
value_x, value_edge_index = raw_value_obs

# 3. Align Node Features and Action Masks
# The agent expects the action array y to be perfectly aligned with the number of nodes.
total_actions = env.num_nodes * 2  # 20 actions

# Create a padded node feature matrix where total nodes = total_actions (20)
policy_nodes_aligned = torch.zeros(total_actions, policy_x.size(1))
policy_nodes_aligned[:env.num_nodes, :] = policy_x  # Put actual color features in the first 10 rows

# Now pad the 3 color features up to the 17 features expected by c_in_p
padded_policy_x = torch.cat([policy_nodes_aligned, torch.zeros(total_actions, 17 - policy_x.size(1))], dim=1)

# Do the same for the critic (value) graph to keep it structurally parallel
value_nodes_aligned = torch.zeros(total_actions, value_x.size(1))
value_nodes_aligned[:env.num_nodes, :] = value_x
padded_value_x = torch.cat([value_nodes_aligned, torch.zeros(total_actions, 12 - value_x.size(1))], dim=1)

# 4. Generate Edge Attributes & Valid Action Masking (y tensor)
policy_edges = torch.zeros(policy_edge_index.size(1), 7)
critic_edges = torch.zeros(value_edge_index.size(1), 2)

# Assign action indices (0 to 19) to match rows in padded_policy_x
action_ids = torch.tensor([i for i in range(total_actions)], dtype=torch.long)

# 5. Build PyTorch Geometric Graph Structures
policy_graph = Data(x=padded_policy_x, edge_index=policy_edge_index, edge_attr=policy_edges, y=action_ids)
critic_graph = Data(x=padded_value_x, edge_index=value_edge_index, edge_attr=critic_edges)

# 6. Collate single instances into Batches
policy_batch = Batch.from_data_list([policy_graph]).to(device)
critic_batch = Batch.from_data_list([critic_graph]).to(device)
network_input = (policy_batch, critic_batch)

# 7. Initialize AgentGNN Topology
agent = AgentGNN(envs=None, device=device, c_hidden=32, c_hidden_v=32).to(device)


print("\n--- Testing forward pass with MockColorEnv Data ---")
with torch.no_grad():
    # FIXED: Removed 'testing=True' to match your exact signature get_action(self, x, device, action=None)
    action, action_id = agent.get_action(network_input, device=device)


# 9. Step the Environment using the Agent's Action
# We pass the action integer directly to the environment's step function
next_state, reward, done, truncated, step_info = env.step(chosen_action)

# 10. Output the Environment Results
print("\n=== Environment Execution Summary ===")
print(f"Targeted Node: {step_info['node_changed']}")
print(f"Direction: {'Down' if chosen_action % 2 == 1 else 'Up'}")
print(f"Color Before: {color_map[step_info['color_before']]}")
print(f"Color After: {color_map[step_info['color_after']]}")
print("-" * 30)
print(f">>> Step Reward Received: {reward} <<<")
print("-" * 30)

print("\nNode Colors (Updated State):")
for node in sorted(env.graph.nodes):
    print(f"  Node {node}: {color_map[env.graph.nodes[node]['color']]}")