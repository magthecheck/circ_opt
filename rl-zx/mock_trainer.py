import torch
import torch.optim as optim
from torch_geometric.data import Data, Batch
from rl_agent import AgentGNN

device = torch.device("cpu")
agent = AgentGNN(envs=None, device=device, c_hidden=32, c_hidden_v=32).to(device)

# --- 1. Prepare Your Fixed Input Data ---
edge_index = torch.tensor([[0, 1, 2, 0], [1, 2, 3, 0]], dtype=torch.long)
policy_nodes = torch.randn(4, 17)
policy_edges = torch.randn(4, 7)
action_ids = torch.tensor([100, 101, 102, -1], dtype=torch.long) 

policy_graph = Data(x=policy_nodes, edge_index=edge_index, edge_attr=policy_edges, y=action_ids)
critic_nodes = torch.randn(4, 12)
critic_edges = torch.randn(4, 2)
critic_graph = Data(x=critic_nodes, edge_index=edge_index, edge_attr=critic_edges)

network_input = (Batch.from_data_list([policy_graph]).to(device), 
                 Batch.from_data_list([critic_graph]).to(device))

# --- 2. SETUP FOR LEARNING (NEW) ---
# We use the Adam optimizer to update the network weights
optimizer = optim.Adam(agent.parameters(), lr=0.001)

print("Starting mock training step...")

# --- 3. THE TRAINING CYCLE ---

# Step A: Clear out any old gradients from a previous run
optimizer.zero_grad()

# Step B: Run the full forward pass to get actions, log probabilities, and values
# Note: We use get_action_and_value here because training requires logprob and entropy!
action, logprob, entropy, values, action_logits, action_id = agent.get_action_and_value(
    network_input, device="cpu"
)

# Step C: Create dummy RL targets
# In real RL, these come from your environment's rewards
dummy_advantage = torch.tensor([1.5])  # Pretend the action taken was "better than expected"
dummy_return = torch.tensor([2.0])     # Pretend the total future reward score is 2.0

# Step D: Calculate Actor Loss & Critic Loss
# Actor wants to maximize logprob * advantage. (We minimize negative logprob)
actor_loss = -(logprob * dummy_advantage).mean() - (0.01 * entropy.mean())

# Critic wants its estimated value to match the actual return (Mean Squared Error)
critic_loss = torch.nn.functional.mse_loss(values, dummy_return)

# Total Loss combined
total_loss = actor_loss + critic_loss
print(f"Calculated Total Loss: {total_loss.item():.4f}")

# Step E: BACKPROPAGATION
# This is where PyTorch shines. It automatically calculates how to tweak 
# every weight in your actor_gnn and critic_gnn to reduce the loss.
total_loss.backward()

# Step F: OPTIMIZER STEP
# This actually applies the changes to the weights!
optimizer.step()

print("Success! One full backpropagation training step completed cleanly.")