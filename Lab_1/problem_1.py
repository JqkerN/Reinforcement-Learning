"""
    KTH: EL2805 Reinforcement Learning 
               November 2020
                Problem 1

                Arthurs: 
Ilian Corneliussen  950418-2438  ilianc@kth.se
Daniel Hirsch       960202-5737  dhirsch@kth.se   
"""
import numpy as np
import src.maze as mz 



def main():
    # Description of the maze as a numpy array
    # with the convention 
    # 0 = empty cell
    # 1 = obstacle
    # 2 = exit of the Maze
    maze = np.array([
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 1, 1],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],  
        [0, 0, 0, 0, 1, 2, 0, 0]
    ])
    # mz.draw_maze(maze)

    # Create an environment 
    can_stay = False
    env = mz.Maze(maze, can_stay=can_stay)
    # env.show()

    # Finite horizon
    horizon = 20

    # Solve the MDP problem with dynamic programming 
    _, policy = mz.dynamic_programming(env,horizon)

    # Simulate the shortest path starting from position A with dynamic programming
    method = 'DynProg'
    start  = ((0,0),(6,5))  # ((Player_pose),(Minotaur_pose))
    path = env.simulate(start, policy, method) # Generate the path for the player and the minotaur

    # Animation of the game
    mz.animate_solution(maze, path, can_stay=can_stay)


if __name__ == '__main__':
    main()