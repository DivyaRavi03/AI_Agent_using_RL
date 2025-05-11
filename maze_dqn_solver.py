import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
from maze import maze, start, goal
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rows, cols = maze.shape
actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
action_map = [(-1, 0), (1, 0), (0, -1), (0, 1)]
action_size = len(actions)



gamma = 0.99               # Discount factor
epsilon = 1.0              # Initial exploration rate
epsilon_min = 0.05         # Minimum exploration rate
epsilon_decay = 0.995      # Decay rate for exploration
lr = 0.0005                # Learning rate
episodes = 15000           # Training episodes
max_steps = 200            # Max steps per episode
batch_size = 128
memory_size = 10000        # Replay memory size


class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 128),  
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)  
        )

    def forward(self, x):
        return self.fc(x)

def encode_state(state):
    return torch.tensor([state[0] / rows, state[1] / cols], dtype=torch.float32, device=device)


def is_valid(state): # Check if a position is within bounds and not a wall
    r, c = state
    return 0 <= r < rows and 0 <= c < cols and maze[r][c] == 0

def step(state, action):
    dr, dc = action_map[action]
    next_state = (state[0] + dr, state[1] + dc)

    if not is_valid(next_state):
        return state, -20, False  # Penalty for hitting a wall

   
    curr_dist = abs(state[0] - goal[0]) + abs(state[1] - goal[1])  
    next_dist = abs(next_state[0] - goal[0]) + abs(next_state[1] - goal[1])
    dist_reward = 2 if next_dist < curr_dist else -1

    if next_state == goal:
        return next_state, 200, True  # rewards for reaching goal

    return next_state, dist_reward, False



model = QNetwork().to(device)
target_model = QNetwork().to(device)
target_model.load_state_dict(model.state_dict())  # weights
target_model.eval()

optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()
memory = deque(maxlen=memory_size)

rewards_log = []
best_reward = float('-inf')  


# training loop
for ep in range(episodes): 
    state = start
    total_reward = 0

    for _ in range(max_steps):
        state_tensor = encode_state(state).unsqueeze(0)

        # Epsilon-greedy policy: explore or exploit
        if random.random() < epsilon:
            action = random.randint(0, action_size - 1)
        else:
            with torch.no_grad():
                q_values = model(state_tensor)
                action = torch.argmax(q_values).item()

        # Take action and observing result
        next_state, reward, done = step(state, action)
        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        if done:
            break

        # Experience Replay
        if len(memory) >= batch_size:
            batch = random.sample(memory, batch_size)
            states, actions_, rewards_, next_states, dones = zip(*batch)

            states_tensor = torch.tensor([[s[0]/rows, s[1]/cols] for s in states], dtype=torch.float32, device=device)
            actions_tensor = torch.tensor(actions_, dtype=torch.int64, device=device).unsqueeze(1)
            rewards_tensor = torch.tensor(rewards_, dtype=torch.float32, device=device)
            next_states_tensor = torch.tensor([[s[0]/rows, s[1]/cols] for s in next_states], dtype=torch.float32, device=device)
            dones_tensor = torch.tensor(dones, dtype=torch.float32, device=device)

            
            q_values = model(states_tensor).gather(1, actions_tensor).squeeze() # Q-value for current state-action pairs

           # Q-target using target network
            with torch.no_grad():
                next_q_values = target_model(next_states_tensor).max(1)[0]
            target_q = rewards_tensor + (1 - dones_tensor) * gamma * next_q_values

            loss = loss_fn(q_values, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    rewards_log.append(total_reward)
    epsilon = max(epsilon * epsilon_decay, epsilon_min)  # Decay epsilon

    if total_reward > best_reward:
        best_reward = total_reward
        torch.save(model.state_dict(), 'best_dqn_model.pth')

    if ep % 50 == 0: #updating target network 
        target_model.load_state_dict(model.state_dict())

    if (ep + 1) % 100 == 0:
        print(f"Episode {ep+1}, Reward: {total_reward}, Epsilon: {epsilon:.3f}")



plt.plot(rewards_log)
plt.title("DQN Training Progress")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)
plt.show()



model.load_state_dict(torch.load('best_dqn_model.pth'))
print("\nEvaluating DQN Agent over 100 test runs...")

successes = 0
all_steps, all_costs, all_times = [], [], []

def test_agent():
    path = []
    state = start
    steps = 0
    cost = 0
    start_time = time.time()
    visited = {}

    while state != goal and steps < max_steps:
        path.append(state)
        state_tensor = encode_state(state).unsqueeze(0)
        with torch.no_grad():
            q_values = model(state_tensor)
            action = torch.argmax(q_values).item()

        dr, dc = action_map[action]
        next_state = (state[0] + dr, state[1] + dc)

        # Handle invalid moves
        if not is_valid(next_state):
            q_values[0][action] = -float('inf')
            action = torch.argmax(q_values).item()
            dr, dc = action_map[action]
            next_state = (state[0] + dr, state[1] + dc)
            if not is_valid(next_state):
                break

        # limiting values to avoid infinte loops
        visited[next_state] = visited.get(next_state, 0) + 1
        if visited[next_state] > 3:
            break

        state = next_state
        steps += 1
        cost += -1 if state != goal else 100

    if state == goal:
        path.append(goal)

    end_time = time.time()
    return steps, cost, end_time - start_time, state == goal, path

# Run evaluation
all_paths = []
for _ in range(100):
    steps, cost, elapsed, success, path = test_agent()
    if success:
        successes += 1
        all_steps.append(steps)
        all_costs.append(cost)
        all_times.append(elapsed)
    all_paths.append(path)



print("\nDQN Evaluation Results:")
print(f"Success rate: {successes}%")
if all_steps:
    print(f"Avg steps to goal: {sum(all_steps)/len(all_steps):.2f}")
    print(f"Avg path cost: {sum(all_costs)/len(all_costs):.2f}")
    print(f"Avg time taken: {sum(all_times)/len(all_times):.4f} sec")
else:
    print("Agent failed to reach the goal in any test run.")



final_path = max(all_paths, key=lambda p: len(p))  # Longest successful path

def display_path(path):
    display = maze.copy()
    for (x, y) in path:
        display[x][y] = 8
    display[start[0]][start[1]] = 5
    display[goal[0]][goal[1]] = 9

    fig, ax = plt.subplots()
    cmap = plt.get_cmap('Set3')
    ax.imshow(display, cmap=cmap)
    for (x, y) in path:
        if (x, y) == start:
            ax.text(y, x, 'S', ha='center', va='center', weight='bold')
        elif (x, y) == goal:
            ax.text(y, x, 'G', ha='center', va='center', weight='bold')
        else:
            ax.text(y, x, '*', ha='center', va='center', color='gray')
    ax.set_title("Maze Path Learned by DQN")
    plt.show()

display_path(final_path)
