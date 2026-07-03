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

# for command line parameters
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train RL Agent with GNN/MLP on MockColorEnv")
    parser.add_argument(
        "--episodes", "-e", 
        type=int, 
        default=4000, 
        help="Total game rounds to train on (default: 4000)"
    )
    parser.add_argument(
        "--model-out", "-m", 
        type=str, 
        default="trained_color_gnn.pt", 
        help="Output filename/path for the saved PyTorch model weights (default: trained_color_gnn.pt)"
    )
    parser.add_argument(
        "--csv-out", "-c", 
        type=str, 
        default="training_logs.csv", 
        help="Output filename/path for the CSV training logs (default: training_logs.csv)"
    )

    parser.add_argument(
        "--nodes", "-n",
        type=int,
        default = 10,
        help="Number of nodes in the environment graph (default: 10)"
    )
    parser.add_argument(
        "--gpu", "-g",
        action="store_true",
        help="Enable GPU acceleration modes, vectorization, and batch optimization paths."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # --- Hyperparameters ---
    NUM_EPISODES = args.episodes
    NUM_NODES = args.nodes
    csv_file_path = args.csv_out
    model_file_path = args.model_out
    USE_GPU_OPTIMIZATIONS = args.gpu

    EPISODE_LENGTH = 6      # Maximum steps per episode
    LR = 5e-4                # Learning rate for Adam Optimizer
    GAMMA = 0.99             # Discount factor for long-term rewards
    PPO_EPOCHS = 4           # How many times to reuse collected data per update
    CLIP_EPS = 0.2           # PPO clipping constraint for safe updates

    if USE_GPU_OPTIMIZATIONS and torch.cuda.is_available():
        device = torch.device("cuda")   
    else:
        device = torch.device("cpu")
        if USE_GPU_OPTIMIZATIONS:
            print("Warning: GPU flag requested but CUDA is unavailable. Defaulting to CPU.")

    print(f"Running training loop on: {device}")
    print(f"Graph Size: {NUM_NODES} nodes.")
    print(f"Training for {NUM_EPISODES} episodes.")
    print(f"Logging to: {csv_file_path}")
    print(f"Model will save to: {model_file_path}\n")

    # 1. Initialize Objects
    env = MockColorEnv(num_nodes=NUM_NODES, max_episode_len=EPISODE_LENGTH)
    agent = SimpleNodeAgent(num_nodes=env.num_nodes).to(device) 
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
    episode_p_graphs = []
    episode_v_graphs = []
    episode_local_actions = []
    episode_action_ids = [] 
    episode_log_probs = []
    episode_rewards = []
    episode_values = []
    total_episode_reward = 0
    
    for step in range(EPISODE_LENGTH):
        # Package data for the forward pass
        raw_p, raw_v = info["graph_obs"]

        p_graph, v_graph = build_pyg_input(raw_p[0], raw_p[1], raw_v[0], raw_v[1])
            
        # Temporary batches just for the single inference step forward pass
        p_batch = Batch.from_data_list([p_graph]).to(device)
        v_batch = Batch.from_data_list([v_graph]).to(device) 
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
        
        # === GPU SPEED FIX: Cache unbatched raw graphs to batch them cleanly later ===
        episode_p_graphs.append(p_graph)
        episode_v_graphs.append(v_graph)
        episode_local_actions.append(action.flatten())
        episode_action_ids.append(action_id)
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
        
    returns_tensor = torch.tensor(discounted_returns, dtype=torch.float32).to(device)
    
    values_tensor = torch.cat([v.to(device) for v in episode_values]).flatten()
    
    old_log_probs_tensor = torch.stack(episode_log_probs).detach().to(device)
    # old_log_probs_tensor = torch.cat(episode_log_probs).detach()
    
    advantages_tensor = returns_tensor - values_tensor.detach()

    if USE_GPU_OPTIMIZATIONS:
        # ==========================================
        # GPU OPTIMIZED PIPELINE (BATCH VECTORIZED)
        # ==========================================

        batched_p_input = Batch.from_data_list([g.to(device) for g in episode_p_graphs])
        batched_v_input = Batch.from_data_list([g.to(device) for g in episode_v_graphs])
        batched_actions = torch.cat(episode_local_actions).to(device) 
        batched_network_input = (batched_p_input, batched_v_input)    



        # --- PHASE 3: OPTIMIZE CRITIC & ACTOR WEIGHTS ---
        for epoch in range(PPO_EPOCHS):
            # === GPU SPEED FIX: Removed the sequential 'for i in range(len(episode_inputs)):' loop.
            # Passing the full batch vectors variables directly through the agent.
            _, new_log_prob, _, new_val, *_ = agent.get_action_and_value(
                batched_network_input, 
                action=batched_actions
                )

            # 1. Actor Loss (Vectorized across the entire episode)
            ratio = torch.exp(new_log_prob - old_log_probs_tensor)
            surr1 = ratio * advantages_tensor
            surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * advantages_tensor
            actor_loss = -torch.min(surr1, surr2).mean() # Added .mean() for the full batch vector
                
            # 2. Critic Loss (Vectorized Mean Squared Error against returns)
            critic_loss = 0.5 * ((new_val.flatten() - returns_tensor) ** 2).mean() # Added .mean()
                
            # 3. Combined Loss Backpropagation
            total_loss = actor_loss + critic_loss
                
            optimizer.zero_grad()
            total_loss.backward
            optimizer.step()
    else: 
        # ==========================================
        # CPU OPTIMIZED PIPELINE (SEQUENTIAL ITER)
        # ==========================================
        for epoch in range(PPO_EPOCHS):
            for i in range(len(episode_p_graphs)):
                # Extract single items to compute one step at a time, avoiding PyG collation overhead on CPU
                s_p_batch = Batch.from_data_list([episode_p_graphs[i]]).to(device)
                s_v_batch = Batch.from_data_list([episode_v_graphs[i]]).to(device)
                single_input = (s_p_batch, s_v_batch)
                single_act = episode_local_actions[i].to(device)

                _, new_log_prob, _, new_val, *_ = agent.get_action_and_value(
                    single_input, 
                    action=single_act
                )

                ratio = torch.exp(new_log_prob - old_log_probs_tensor[i])
                surr1 = ratio * advantages_tensor[i]
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * advantages_tensor[i]
                actor_loss = -torch.min(surr1, surr2)
                   
                critic_loss = 0.5 * (new_val.flatten() - returns_tensor[i]) ** 2
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