import torch
import torch.optim as optim
from torch_geometric.data import Data, Batch
from torch.distributions.categorical import Categorical

from mock_env import MockColorEnv  # Your simple color environment
from rl_agent import AgentGNN     # Your custom GNN Agent
from simple_node_agent import SimpleNodeAgent


# For writing in csv file
import csv
import os 

# --- Hyperparameters ---
NUM_EPISODES = 4000       # Total game rounds to train on
EPISODE_LENGTH = 6      # Maximum steps per episode
LR = 5e-4                # Learning rate for Adam Optimizer
GAMMA = 0.99             # Discount factor for long-term rewards
PPO_EPOCHS = 4           # How many times to reuse collected data per update
CLIP_EPS = 0.2           # PPO clipping constraint for safe updates

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running training loop on: {device}")

# 1. Initialize Objects
env = MockColorEnv(num_nodes=10, max_episode_len=EPISODE_LENGTH)
#agent = SimpleNodeAgent(num_nodes = env.num_nodes).to(env.device)
agent = SimpleNodeAgent(num_nodes = env.num_nodes).to(device) 
#agent = AgentGNN(envs=None, device=device, c_hidden=32, c_hidden_v=32).to(device)
optimizer = optim.Adam(agent.parameters(), lr=LR)

total_actions = env.num_nodes * 2

def build_pyg_input(policy_x, policy_edge_index, value_x, value_edge_index):
    """Adapts, aligns, and pads raw environment graph matrices into PyG Batches."""
    # Align rows to match action masking dimensions (20)
    p_aligned = torch.zeros(total_actions, policy_x.size(1))
    p_aligned[:env.num_nodes, :] = policy_x
    padded_p_x = torch.cat([p_aligned, torch.zeros(total_actions, 17 - p_aligned.size(1))], dim=1)

    v_aligned = torch.zeros(total_actions, value_x.size(1))
    v_aligned[:env.num_nodes, :] = value_x
    padded_v_x = torch.cat([v_aligned, torch.zeros(total_actions, 12 - v_aligned.size(1))], dim=1)

    policy_edges = torch.zeros(policy_edge_index.size(1), 7)
    critic_edges = torch.zeros(value_edge_index.size(1), 2)
    action_ids = torch.tensor([i for i in range(total_actions)], dtype=torch.long)

    p_graph = Data(x=padded_p_x, edge_index=policy_edge_index, edge_attr=policy_edges, y=action_ids)
    v_graph = Data(x=padded_v_x, edge_index=value_edge_index, edge_attr=critic_edges)

    return Batch.from_data_list([p_graph]), Batch.from_data_list([v_graph])


print("Starting Training Loop...")


# Define the CVS file path
csv_file_path = "training_logs.csv"

# Write headers only if the file doesn't exist yet
if not os.path.exists(csv_file_path):
    with open(csv_file_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Total Reward", "Final Loss"])



for episode in range(1, NUM_EPISODES + 1):
    
    # --- PHASE 1: COLLECT ROLLOUT TRAJECTORY ---
    _, info = env.reset()
    
    # Temporary storage buffers for this episode
    episode_inputs = []
    episode_actions = []
    episode_log_probs = []
    episode_rewards = []
    episode_values = []
    
    total_episode_reward = 0
    
    for step in range(EPISODE_LENGTH):
        # Package data for the forward pass
        raw_p, raw_v = info["graph_obs"]
        p_batch, v_batch = build_pyg_input(raw_p[0], raw_p[1], raw_v[0], raw_v[1])
        
        # GPU Usage ---
        p_batch = p_batch.to(device)
        v_batch = v_batch.to(device) 
        # GPU Usage --
        
        network_input = (p_batch, v_batch)
        
        # FIXED: Explicitly call get_action, which returns 2 values (action, action_id) in inference mode
        action, action_id = agent.get_action(network_input, device=device)
        
        # FIXED: Extract underlying critic values and log-probabilities explicitly for the PPO buffers
        with torch.no_grad():
            # Use the built-in helper method for the critic value
            value = agent.get_value(network_input[1])
            
            raw_logits = agent.actor(network_input[0])
            logits = raw_logits.view(-1)
            
            # Form a categorical distribution over the actor logits
            probs = torch.softmax(logits, dim=-1)
            dist = Categorical(probs=probs)
            log_prob = dist.log_prob(action_id)
        
        # 3. Track the environment target integer from the action choice
        chosen_action = action_id.item()
        
        # Step the environment
        _, reward, done, _, info = env.step(chosen_action)
        
        # Store records into rollout optimization buffers
        episode_inputs.append(network_input)
        episode_actions.append(action_id)
        episode_log_probs.append(log_prob)
        episode_rewards.append(reward)
        episode_values.append(value)
        
        total_episode_reward += reward
        if done:
            break

    # --- PHASE 2: CALCULATE ADVANTAGES & TARGET RETURNS ---
    discounted_returns = []
    g = 0
    for r in reversed(episode_rewards):
        g = r + GAMMA * g
        discounted_returns.insert(0, g)
        
    returns_tensor = torch.tensor(discounted_returns, dtype=torch.float32)
    values_tensor = torch.cat(episode_values).flatten().to(device)
    
    old_log_probs_tensor = torch.stack(episode_log_probs).detach().to(device)
    # old_log_probs_tensor = torch.cat(episode_log_probs).detach()
    
    advantages_tensor = returns_tensor - values_tensor.detach()

    # --- PHASE 3: OPTIMIZE CRITIC & ACTOR WEIGHTS ---
    for epoch in range(PPO_EPOCHS):
        for i in range(len(episode_inputs)):
            inp = episode_inputs[i]
            old_act = episode_actions[i]
            
            # FIXED: Explicitly call .get_action(). In this code-path (with action provided),
            # your agent naturally executes all the way through to return exactly 4 training variables.
            _, new_log_prob, _, new_val, *_ = agent.get_action_and_value(inp, action=old_act)

            # 1. Actor Loss (PPO Clipped Objective)
            ratio = torch.exp(new_log_prob - old_log_probs_tensor[i])
            surr1 = ratio * advantages_tensor[i]
            surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * advantages_tensor[i]
            actor_loss = -torch.min(surr1, surr2)
            
            # 2. Critic Loss (Mean Squared Error against returns)
            critic_loss = 0.5 * (new_val.flatten() - returns_tensor[i]) ** 2
            
            # 3. Combined Loss Backpropagation
            total_loss = actor_loss + critic_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

    # Console Logging Tracker
    if episode % 10 == 0 or episode == 1:
        print(f"Episode {episode:3d}/{NUM_EPISODES} | Total Reward: {total_episode_reward:5.1f} | Final Loss: {total_loss.item():.4f}")


        # Append the structured data to the CSV file
        with open(csv_file_path, mode='a', newline='') as f: 
            writer = csv.writer(f)
            writer.writerow([
                episode,
                f"{total_episode_reward:.1f}",
                f"{total_loss.item():.4f}"
                ])
        
# 4. Save Trained Policy Weights to Disk
torch.save(agent.state_dict(), "trained_color_gnn_MLP_4000_new.pt")
print("\nSuccess! Meaningful weights generated and saved as 'trained_color_gnn_MLP_4000_new.pt'.")