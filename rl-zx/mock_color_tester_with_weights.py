import os
import torch
from torch_geometric.data import Data, Batch
from rl_agent import AgentGNN
from mock_env import MockColorEnv  

def export_to_dot(graph, step_num, action_desc="None", reward_val=0.0):
    """
    Converts a NetworkX graph with color attributes (1, 2, 3) 
    into a formatted Graphviz DOT layout string.
    """
    color_hex_map = {1: "#3498DB", 2: "#8E44AD", 3: "#2ECC71"}
    color_name_map = {1: "Blue", 2: "Purple", 3: "Green"}

    dot_lines = [
        "graph G {",
        f'    label="Step {step_num} | Last Action: {action_desc} | Reward: {reward_val}";',
        '    labelloc="t";',
        '    node [shape=circle, fontname="Helvetica", style=filled];',
        "",
        "    // Node Color Definitions"
    ]

    for node in sorted(graph.nodes):
        color_val = graph.nodes[node].get('color', 1)
        hex_color = color_hex_map.get(color_val, "#3498DB")
        color_name = color_name_map.get(color_val, "Unknown")
        dot_lines.append(f'    {node} [fillcolor="{hex_color}", fontcolor=white]; // {color_name}')

    # disable edges -> only colours are interesting 
    dot_lines.append("\n    // Edge Connections")
    for u, v in sorted(graph.edges):
        if u <= v:
            dot_lines.append(f"    {u} -- {v};")

    dot_lines.append("}")
    return "\n".join(dot_lines)


# 1. Platform and Env Setup
device = torch.device("cpu")
env = MockColorEnv(num_nodes=5)

# Initialize the GNN Agent Structure
agent = AgentGNN(envs=None, device=device, c_hidden=32, c_hidden_v=32).to(device)

# --- NEW: Load Your Computed Weights Matrix ---
weights_path = "trained_color_gnn.pt"
if os.path.exists(weights_path):
    agent.load_state_dict(torch.load(weights_path, map_location=device))
    print(f"Successfully loaded trained weights from '{weights_path}'!")
    # Lock the network layer behaviors into assessment/inference mode
    agent.eval()
else:
    print(f"Warning: '{weights_path}' not found. Running inference with random weights.")
    agent.eval()

color_map = {1: "Blue", 2: "Purple", 3: "Green"}
total_actions = env.num_nodes * 2 

# Initialize the environment
graph_state, info = env.reset(seed=42)

print("=" * 50)
print("STARTING 10-STEP REINFORCEMENT LEARNING AGENT LOOP")
print("=" * 50)

# Print Initial Unmodified State
initial_dot = export_to_dot(env.graph, step_num=0, action_desc="Reset / Initial")
print("\n=== INITIAL STATE (STEP 0) ===")
print(initial_dot)

# 2. Multi-Step Interaction Loop
for step in range(1, 30):
    # Extract Graph Tensors from the 'info' observation metadata wrapper
    raw_policy_obs, raw_value_obs = info["graph_obs"]
    policy_x, policy_edge_index = raw_policy_obs
    value_x, value_edge_index = raw_value_obs

    # Align Node dimensions to total actions (20) to prevent x[0].batch indexing errors
    policy_nodes_aligned = torch.zeros(total_actions, policy_x.size(1))
    policy_nodes_aligned[:env.num_nodes, :] = policy_x
    padded_policy_x = torch.cat([policy_nodes_aligned, torch.zeros(total_actions, 17 - policy_x.size(1))], dim=1)

    value_nodes_aligned = torch.zeros(total_actions, value_x.size(1))
    value_nodes_aligned[:env.num_nodes, :] = value_x
    padded_value_x = torch.cat([value_nodes_aligned, torch.zeros(total_actions, 12 - value_x.size(1))], dim=1)

    # Edge Features and Mask Construction
    policy_edges = torch.zeros(policy_edge_index.size(1), 7)
    critic_edges = torch.zeros(value_edge_index.size(1), 2)
    action_ids = torch.tensor([i for i in range(total_actions)], dtype=torch.long)

    # Build PyG Graph Objects
    policy_graph = Data(x=padded_policy_x, edge_index=policy_edge_index, edge_attr=policy_edges, y=action_ids)
    critic_graph = Data(x=padded_value_x, edge_index=value_edge_index, edge_attr=critic_edges)

    # Collate Single Graphs into Batches for the Network Forward Pass
    policy_batch = Batch.from_data_list([policy_graph]).to(device)
    critic_batch = Batch.from_data_list([critic_graph]).to(device)
    network_input = (policy_batch, critic_batch)

    # 3. Agent Evaluation (Predict Action using loaded weights)
    with torch.no_grad():
        # Your agent's get_action returns (action, action_id) in inference mode
        action, action_id = agent.get_action(network_input, device=device)
    
    chosen_action = action_id.item()
    
    # Decode the Action for the Console Print Statement
    target_node = chosen_action // 2
    direction_str = "Down" if chosen_action % 2 == 1 else "Up"
    action_description = f"Node {target_node} Shift {direction_str}"

    # 4. Step the Environment using the Agent's Choice
    next_state, reward, done, truncated, info = env.step(chosen_action)

    # 5. Format and Print the Current Step Result
    dot_output = export_to_dot(env.graph, step_num=step, action_desc=action_description, reward_val=reward)
    
    print(f"\n=== EPISODE STEP {step} ===")
    print(f"Executed: {action_description}")
    print(f"Resulting Reward: {reward}")
    print(f"Is Episode Done? {done}")
    print("\n--- Current Graphviz DOT Structure ---")
    print(dot_output)
    print("-" * 50)
    
    # If the environment signals termination (max length hit), break out early
    if done:
        print("Environment signaled completion.")
        break