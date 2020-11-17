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
import time


def B_1():
    ##################################
    ############## A #################
    ##################################
    print('Running problem 1, task B (1)')
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
    env.show()

    # Finite horizon
    horizon = 20

    # Solve the MDP problem with dynamic programming 
    _, policy = mz.dynamic_programming(env,horizon)

    # Simulate the shortest path starting from position A with dynamic programming
    method = 'ValIter'
    start  = ((0,0),(6,5))  # ((Player_pose),(Minotaur_pose))
    path = env.simulate(start, policy, method) # Generate the path for the player and the minotaur

    # Animation of the game
    mz.animate_solution(maze, path, method, can_stay=can_stay)

def B_2():
    ##################################
    ############## B #################
    ##################################
    print('Running problem 1, task B (2)')
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

    # Generate plot for maximal probability as a function of T
    can_stay = [False, True]
    colors = ['black', 'red']
    
    max_T = 25
    nr_iterations = 10000
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

def C():
    ##################################
    ############## C #################
    ##################################
    print('Running problem 1, task C')
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
    # Create an environment 
    can_stay = False
    env = mz.Maze(maze, can_stay=can_stay)

    # Solve the MDP problem with dynamic programming 
    gamma   = 1.0 - 1.0/30
    epsilon = 0.01
    print('Gamma: {:.2f}, Episilon: {:.2f}'.format(gamma, epsilon))
    _, policy = mz.value_iteration(env, gamma=gamma, epsilon=epsilon)

    # Simulate the shortest path starting from position A with dynamic programming
    method = 'ValIter'
    start  = ((0,0),(6,5))  # ((Player_pose),(Minotaur_pose))
    out_n_alive = 0
    iterations = 10000
    for _ in range(iterations):
        result = False
        path = env.simulate(start, policy, method) # Generate the path for the player and the minotaur
        for step in path:
            if step[0] == (6,5) and prev_step[0] == (6,5):
                result = True
                break
            prev_step = step
        if result:
            out_n_alive += 1

    print('Probability of getting out alive (10,000 runs): {}'.format(out_n_alive/iterations))
    # Animation of the game
    # for _ in range(100):
    #     path = env.simulate(start, policy, method, goal=(6,5)) 
    #     mz.animate_solution(maze, path, method, can_stay=can_stay)

if __name__ == '__main__':
    start = time.time()
    B_1()
    # B_2()
    # C()
    print('Script runtime: {}'.format(time.time() - start))