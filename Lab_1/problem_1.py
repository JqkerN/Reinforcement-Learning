import numpy as np
import src.maze as mz 



# Description of the maze as a numpy array
maze = np.array([
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 1, 1, 1],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 2, 0, 0]
])
# with the convention 
# 0 = empty cell
# 1 = obstacle
# 2 = exit of the Maze

# mz.draw_maze(maze)


# Create an environment 
can_stay = False
env = mz.Maze(maze, can_stay=can_stay)
# env.show()

# Finite horizon
horizon = 20
# Solve the MDP problem with dynamic programming 
V, policy = mz.dynamic_programming(env,horizon)

# Simulate the shortest path starting from position A
method = 'DynProg'
start  = ((0,0),(6,5))

path = env.simulate(start, policy, method)

mz.animate_solution(maze, path, can_stay=can_stay)