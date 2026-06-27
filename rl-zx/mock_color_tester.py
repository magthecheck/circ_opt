import torch
from torch_geometric.data import Data, Batch
from rl_agent import AgentGNN
from mock_env import MockColorEnv  # Assuming your mock env code is saved as mock_env.py

def export_to_dot(graph):
    """
    Converts a NetworkX graph with color attributes (1, 2, 3) 
    into a valid Graphviz DOT format string.
    """
    # Color hex mapping to match your exact styling preference
    color_hex_map = {
        1: "#3498DB",  # Blue
        2: "#8E44AD",  # Purple
        3: "#2ECC71"   # Green
    }
    color_name_map = {1: "Blue", 2: "Purple", 3: "Green"}

    dot_lines = []
    dot_lines.append("graph G {")
    dot_lines.append('    node [shape=circle, fontname="Helvetica", style=filled];')
    dot_lines.append("")
    dot_lines.append("    // Node Color Definitions")

    # 1. Process and style nodes
    for node in sorted(graph.nodes):
        color_val = graph.nodes[node].get('color', 1) # Default to 1 if not set
        hex_color = color_hex_map.get(color_val, "#3498DB")
        color_name = color_name_map.get(color_val, "Unknown")
        
        node_line = f'    {node} [fillcolor="{hex_color}", fontcolor=white]; // {color_name}'
        dot_lines.append(node_line)

    dot_lines.append("")
    dot_lines.append("    // Edge Connections")

    # 2. Process edges (ensuring we only list each undirected edge once)
    for u, v in sorted(graph.edges):
        if u <= v:  # Keeps formatting neat and ordered (e.g., 0 -- 2 instead of 2 -- 0)
            dot_lines.append(f"    {u} -- {v};")

    dot_lines.append("}")
    
    return "\n".join(dot_lines)

device = torch.device("cpu")
print(f"Using device {device}")

# 1. Initialize the Environment
env = MockColorEnv(num_nodes=5)
graph_state, info = env.reset(seed=42)



# --- EXPORT AND PRINT INITIAL DOT GRAPH ---
print("\n=== Graphviz DOT Structure (Initial State) ===")
initial_dot_string = export_to_dot(env.graph)
print(initial_dot_string)

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

# Extract the concrete integer value from the PyTorch Tensor
chosen_action = action_id.item()
print(f"Agent sampled Action ID: {chosen_action}")

# 9. Step the Environment using the Agent's Action
next_state, reward, done, truncated, step_info = env.step(chosen_action)

# --- EXPORT AND PRINT POST-STEP DOT GRAPH ---
print("\n=== Graphviz DOT Structure (After Agent Action) ===")
print(f"// Agent executed Action ID {chosen_action}: targeted Node {step_info['node_changed']}")
updated_dot_string = export_to_dot(env.graph)
print(updated_dot_string)

print(f"\nStep Reward: {reward}")





