import torch
from torch_geometric.data import Data, Batch
from rl_agent import AgentGNN

device = torch.device("cpu")
print(f"Using device {device}")

# 2. Initialize your Agent (Passing None for envs since we are building dummy data)
# We set c_hidden to a small number (32) to keep it lightweight
agent = AgentGNN(envs = None, device=device, c_hidden=32, c_hidden_v=32).to(device)

# Dummy graph 0->1, 1->2, 2->3
edge_index = torch.tensor(  [[0,1,2],
                             [1,2,3]], dtype= torch.long)

policy_nodes = torch.randn(4,17) # 4 nodes with 17 features 
policy_edges = torch.randn(3,7) # 3 edges with 7 features each 


action_ids = torch.tensor([100, 99, -1, 105], dtype=torch.long)
# 0,1,2, are valid, 3 is not valid

policy_graph = policy_graph = Data(x=policy_nodes, edge_index=edge_index, edge_attr=policy_edges, y=action_ids)

# Critic Input Data (Needs 12 node features, 2 edge features)
critic_nodes = torch.randn(4, 12)   # 4 nodes, 12 features each
critic_edges = torch.randn(3, 2)    # 3 edges, 2 features each

critic_graph = Data(x=critic_nodes, edge_index=edge_index, edge_attr=critic_edges)


# 4. PyTorch Geometric models expect data to be batched
# Even for a single graph, we turn it into a Batch of size 1
policy_batch = Batch.from_data_list([policy_graph]).to(device)
critic_batch = Batch.from_data_list([critic_graph]).to(device)

# Pack them into the tuple format your agent looks for
network_input = (policy_batch, critic_batch)


print("\n--- Inspecting Actor Weights ---")

# 1. Inspect the very first Graph Attention Layer (GATv2Conv)
# In PyTorch Geometric Sequential, layers can be accessed by index
first_gat_layer = agent.actor_gnn[0]
print("First Actor GAT Layer Weight Shape (Source):", first_gat_layer.lin_r.weight.shape)
print("First Actor GAT Layer Weights (Snippet):\n", first_gat_layer.lin_r.weight[:2, :5]) # Show a 2x5 slice

print("-" * 40)

# 2. Inspect the final Linear layer that outputs the action logits
# Index 10 corresponds to the final nn.Linear(c_hidden, 1) layer in your actor chain
final_linear_layer = agent.actor_gnn[10]
print("Final Actor Linear Layer Weight Shape:", final_linear_layer.weight.shape)
print("Final Actor Linear Layer Weights:\n", final_linear_layer.weight)


# 5. Run the Forward Pass (No optimization, just checking if data flows)
print("\n--- Testing forward pass ---")
with torch.no_grad(): # No optimizations 
    action, action_id = agent.get_action(network_input, device="cpu")


print("Success! The data flowed through the network without errors.")
print(f"Sampled Action Index: {action.item()}")
print(f"Corresponding Action ID from graph: {action_id.item()}")