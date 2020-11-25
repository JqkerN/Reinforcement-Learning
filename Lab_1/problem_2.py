"""
    KTH: EL2805 Reinforcement Learning 
               November 2020
                Problem 1

                Arthurs: 
Ilian Corneliussen  950418-2438  ilianc@kth.se
Daniel Hirsch       960202-5737  dhirsch@kth.se   
"""
import numpy as np
import src.gotham as gotham 
import matplotlib.pyplot as plt
import time


def B():
    ##################################
    ############## A #################
    ##################################
    print('Running problem 2, task B')
    # Description of the maze as a numpy array
    # with the convention 
    # 0 = empty cell
    # 1 = bank
    city = np.array([
        [2, 0, 0, 0, 0, 2],
        [0, 0, 1, 0, 0, 0],
        [2, 0, 0, 0, 0, 2]
    ])

    # gotham.draw_city(city)

    # Create an environment 
    can_stay = False
    env = gotham.City(city, can_stay=can_stay)
    # env.show()
    print("Number of states: {}".format(env.n_states))
    # # Finite horizon
    # horizon = 20

    # # Generate plot for lambda and T
    # gamma_vec   = np.arange(start=0.1, stop=1, step = 0.01 )
    # epsilon     = 0.01
    # V_vec = list()

    # for gamma in gamma_vec:
    #     # print('Gamma: {:.2f}, Episilon: {:.2f}'.format(gamma, epsilon))
    #     V, policy = gotham.value_iteration(env, gamma=gamma, epsilon=epsilon)
    #     V_vec.append(V[8])
    # plt.plot(gamma_vec, V_vec, '-o', c='k')
    # plt.title('Value function as a function of the discount factor')
    # plt.xlabel('lambda')
    # plt.ylabel('Value function')
    # plt.show()


    # Solve the MDP problem with Value Iteration
    method = 'ValIter'
    gamma = 0.8
    epsilon = 0.01
    V, policy = gotham.value_iteration(env, gamma=gamma, epsilon=epsilon)
    
    # Simulate the shortest path starting from position A with Value Iteration
    start  = ((0,0),(1,2))  # ((Player_pose),(Minotaur_pose))
    path = env.simulate(start, policy, method, iterations=50) # Generate the path for the player and the minotaur
    
    banks = [(0,0), (2,0), (0, 5), (5,5)]
    score = 0
    prev_step = (2,2)
    for step in path:
        if step[0] in banks and prev_step[0] in banks:
            score += 10
        prev_step = step
    print('Score: {}'.format(score))
    # Animation of the game
    gotham.animate_solution(city, path, method, can_stay=can_stay, pause_time=0.5, save=True)



if __name__ == '__main__':
    start = time.time()
    B()
    print('Script runtime: {}'.format(time.time() - start))