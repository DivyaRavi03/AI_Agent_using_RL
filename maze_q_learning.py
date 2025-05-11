import numpy as np
import random
import time
import matplotlib.pyplot as plt
from maze import maze, start, goal 

rows, cols = maze.shape 
actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']  
action_map = {
    'UP': (-1, 0),
    'DOWN': (1, 0),
    'LEFT': (0, -1),
    'RIGHT': (0, 1)
}

# Initializing Q-table: 3D array to store state-action values
q_table = np.zeros((rows, cols, len(actions)))

# Q-learning Hyperparameters
alpha = 0.1       # Learning rate
gamma = 0.9       # Discount factor: value of future rewards
epsilon = 0.2     # Exploration rate: chance of trying random action
episodes = 500    # Total training episodes


def is_valid(pos): # Check if a move is valid not hitting a wall
    r, c = pos
    return 0 <= r < rows and 0 <= c < cols and maze[r][c] == 0

def choose_action(state): # Decide whether to explore or exploit
    if random.uniform(0, 1) < epsilon:
        return random.choice(actions)  
    else:
        r, c = state
        return actions[np.argmax(q_table[r][c])]  


def step(state, action): # take an action and return the next state and reward
    r, c = state
    dr, dc = action_map[action]
    new_r, new_c = r + dr, c + dc

    if not is_valid((new_r, new_c)):
        return state, -10  # Penalty for hitting a wall

    if (new_r, new_c) == goal:
        return (new_r, new_c), 50  # Reward for reaching goal

    return (new_r, new_c), -1  # Small penalty for each step to find shortest path


episode_rewards = []  # Track total rewards per episode for plotting

for episode in range(episodes):
    state = start
    total_reward = 0

    while state != goal:
        action = choose_action(state)
        next_state, reward = step(state, action)

        # Q-value update rule
        r, c = state
        a = actions.index(action)
        nr, nc = next_state
        q_table[r][c][a] += alpha * (reward + gamma * np.max(q_table[nr][nc]) - q_table[r][c][a])

        state = next_state
        total_reward += reward

    episode_rewards.append(total_reward)

print("Training complete")


path = []
state = start
path_cost = 0
steps = 0
start_time = time.time()
path.append(state)

while state != goal:
    r, c = state
    best_action = actions[np.argmax(q_table[r][c])]
    dr, dc = action_map[best_action]
    next_state = (r + dr, c + dc)

    # Avoid invalid moves or cycles
    if not is_valid(next_state) or next_state in path:
        break

    path_cost += -1 if next_state != goal else 50
    path.append(next_state)
    state = next_state
    steps += 1

end_time = time.time()
elapsed_time = end_time - start_time


def display_maze_with_path(maze, path):
    display = maze.copy()
    for (x, y) in path:
        display[x][y] = 8  # Highlight path cells

    display[start[0]][start[1]] = 5  # Mark start
    display[goal[0]][goal[1]] = 9    # Mark goal

    fig, ax = plt.subplots()
    cmap = plt.get_cmap('Set3')
    cax = ax.imshow(display, cmap=cmap)
    plt.colorbar(cax)

    # Add text labels for start, goal, and path
    for i in range(display.shape[0]):
        for j in range(display.shape[1]):
            if (i, j) == start:
                ax.text(j, i, 'S', ha='center', va='center', color='black', fontsize=12, weight='bold')
            elif (i, j) == goal:
                ax.text(j, i, 'G', ha='center', va='center', color='black', fontsize=12, weight='bold')
            elif (i, j) in path and (i, j) != start and (i, j) != goal:
                ax.text(j, i, '*', ha='center', va='center', color='gray')

    ax.set_title("Maze with Q-learning Path")
    plt.show()

if path:
    display_maze_with_path(maze, path) #learned path


plt.plot(episode_rewards)
plt.title("Q-learning Training Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)
plt.show()


print("\nTest Run Metrics:")
print(f"Steps to Goal: {steps}")
print(f"Path Cost: {path_cost}")
print(f"Time Taken: {elapsed_time:.6f} seconds")


success_count = 0
max_steps = 100
step_counts = []

print("\nEvaluation over 100 test runs...")

for _ in range(100):
    state = start
    visited = set()
    steps_eval = 0

    for _ in range(max_steps):
        r, c = state
        best_action = actions[np.argmax(q_table[r][c])]
        dr, dc = action_map[best_action]
        next_state = (r + dr, c + dc)

        if not is_valid(next_state) or next_state in visited:
            break

        if next_state == goal:
            success_count += 1
            step_counts.append(steps_eval)
            break

        visited.add(next_state)
        state = next_state
        steps_eval += 1


print(f"\nEvaluation Results:")
print(f"Success rate: {success_count}%")
if step_counts:
    print(f"Avg steps to goal: {sum(step_counts) / len(step_counts):.2f}")
else:
    print("Agent failed to reach the goal in any run.")


input("\nPress Enter to exit...")


#displays learning path at first and by closing the plotted image will make learned path to appear 
#close and press enter to terminate