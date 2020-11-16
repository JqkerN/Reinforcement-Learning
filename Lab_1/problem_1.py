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
import matplotlib.pyplot as plt



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
    ##################################
    ############## A #################
    ##################################
    A = False
    if A:
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


    ##################################
    ############## B #################
    ##################################
    B = True
    if B:
        # Generate plot for maximal probability as a function of T
        can_stay = [False, True]
        colors = ['black', 'red']
        
        max_T = 25
        nr_iterations = 20
        method = 'DynProg'
        start  = ((0,0),(6,5))  # ((Player_pose),(Minotaur_pose))

        for i in range(len(can_stay)):
            env = mz.Maze(maze, can_stay=can_stay[i])
            x_vec = list()
            y_vec = list()
            print(can_stay[i])
            for T in range(max_T):
                print('T={}'.format(T))
                y_value = 0
                horizon = T
                _, policy = mz.dynamic_programming(env,horizon)
                for _ in range(nr_iterations): 
                    result = False
                    path = env.simulate(start, policy, method) 
                    last_step = start
                    for step in path:
                        if step[0] == (6,5) and last_step[0] == (6,5):
                            result = True
                        last_step = step
                    if result:
                        y_value += 1
                y_vec.append(y_value/nr_iterations)
                x_vec.append(T)
            plt.figure(2)
            plt.plot(x_vec, y_vec, c=colors[i], label='Stay: {}'.format(can_stay[i]))
        plt.title('Probability of exiting the maze')
        plt.legend()
        plt.xlabel('T')
        plt.ylabel('Probability')
        plt.show()


if __name__ == '__main__':
    main()