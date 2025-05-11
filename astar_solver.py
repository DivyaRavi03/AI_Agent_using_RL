import heapq  
import numpy as np
import time
from maze import maze, start, goal  
import matplotlib.pyplot as plt     

# Heuristic function: Manhattan Distance - used to estimate the distance from current to goal
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# A* algorithm implementation
def astar(maze, start, goal):
    
    neighbors = [(0,1), (1,0), (0,-1), (-1,0)]

    close_set = set()              # Visited nodes
    came_from = {}                 # For reconstructing the path
    gscore = {start: 0}            # Cost from start to current node
    total_cost = {start: heuristic(start, goal)}  # Estimated total cost from start to goal

    open_heap = []  # Priority queue for nodes to explore
    heapq.heappush(open_heap, (total_cost[start], start))  # Initializing with start node

    while open_heap:
        current = heapq.heappop(open_heap)[1]  # Get the node with the lowest total cost

        if current == goal:  # Reconstructing path if goal is reached
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]  

        close_set.add(current)  # Helps to mark visited nodes

        for i, j in neighbors:
            neighbor = (current[0] + i, current[1] + j)  # Calculate neighboring cell
            tentative_g_score = gscore[current] + 1      # Cost from start to neighbor

            # Checking boundary and walls 
            if 0 <= neighbor[0] < maze.shape[0] and 0 <= neighbor[1] < maze.shape[1]:
                if maze[neighbor[0]][neighbor[1]] == 1:  
                    continue
            else:
                continue

            
            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0): # skipping in already visited loop
                continue

            # checking if this path to neighbor is better OR not in open_heap yet
            if tentative_g_score < gscore.get(neighbor, float('inf')) or neighbor not in [i[1] for i in open_heap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                total_cost[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_heap, (total_cost[neighbor], neighbor))  # Pushing to list

    return False  # if no path found





start_time = time.time() #start time
path = astar(maze, start, goal)
end_time = time.time() #end time
elapsed_time = end_time - start_time  # Total time taken

if path:
    print("\nPath found:")
    for step in path:
        print(step)

    print("\nEvaluation / Performance Metrics:")
    print(f"Number of steps: {len(path) - 1}")       # Exclude starting position
    print(f"Path cost: {len(path) - 1}")              # Same as number of steps for uniform cost
    print(f"Time taken: {elapsed_time:.6f} seconds")  # showing up to microseconds
else:
    print("No path found")



def display_maze_with_path(maze, path):
    display = maze.copy() 

    for (x, y) in path:
        display[x][y] = 8

    display[start[0]][start[1]] = 5  # Start -> 5
    display[goal[0]][goal[1]] = 9    # Goal  -> 9

    fig, ax = plt.subplots() #setting plot
    cmap = plt.get_cmap('Set3')     
    cax = ax.imshow(display, cmap=cmap)
    plt.colorbar(cax)

    for i in range(display.shape[0]):
        for j in range(display.shape[1]):
            if (i, j) == start:
                ax.text(j, i, 'S', ha='center', va='center', color='black', fontsize=12, weight='bold')
            elif (i, j) == goal:
                ax.text(j, i, 'G', ha='center', va='center', color='black', fontsize=12, weight='bold')
            elif (i, j) in path and (i, j) != start and (i, j) != goal:
                ax.text(j, i, '*', ha='center', va='center', color='gray')

    ax.set_title("Maze with A* Path")
    plt.show()

input("\nPress Enter to exit...")

if path:
    display_maze_with_path(maze, path) # Show the maze with the computed path

input("\nPress Enter to exit...") # Pause at end to keep window open


#close the plotted image to stop the execution