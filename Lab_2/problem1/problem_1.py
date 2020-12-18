import numpy as np
import gym
import torch
from tqdm import trange
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import math
import time
from DQN_agent import Agent, RandomAgent
from collections import deque, namedtuple
from DQN_problem import ExperienceReplayBuffer, Experience, running_average

def A():
    """
    Genereate the agent network and saves it. 
    Main code for solving the first task in the lab instruction.
    """
    EPSILON_MAX = 0.99
    EPSILON_MIN = 0.05

    # Import and initialize the discrete Lunar Laner Environment
    env = gym.make('LunarLander-v2')
    env.reset()

    ### Create Experience replay buffer ###
    L = 20000
    buffer = ExperienceReplayBuffer(maximum_length=L)

    # Parameters
    N_episodes = 600                             # Number of episodes
    discount_factor = 0.985                      # Value of the discount factor
    n_ep_running_average = 50                    # Running average of 50 episodes
    n_actions = env.action_space.n               # Number of available actions
    dim_state = len(env.observation_space.high)  # State dimensionality
    N = 24                                       # BATCH SIZE
    # We will use these variables to compute the average episodic reward and
    # the average number of steps per episode
    episode_reward_list = []       # this list contains the total reward per episode
    episode_number_of_steps = []   # this list contains the number of steps per episode

    # Agent initialization
    agent = Agent(n_actions)

    ### Training process

    # trange is an alternative to range in python, from the tqdm library
    # It shows a nice progression bar that you can update with useful information
    print('N_episodes: {}\ndiscount_factor: {}\nL: {}\nN: {}\n'.format(N_episodes, discount_factor, L, N))
    EPISODES = trange(N_episodes, desc='Episode: ', leave=True)
    for k in EPISODES:
        # Reset enviroment data and initialize variables
        done = False
        state = env.reset()
        total_episode_reward = 0.
        t = 0
        epsilon = max(EPSILON_MIN, EPSILON_MAX * (EPSILON_MIN / EPSILON_MAX) ** (k / (0.9*N_episodes - 1)))

        while not done:
            if N_episodes-k <= 3 or k%20 == 0:
                env.render()  
            if t%int(L/N) == 0:
                agent.update_target_network()
            # Take a random action
            action = agent.forward(state, epsilon)

            # Get next state and reward.  The done variable
            # will be True if you reached the goal position,
            # False otherwise
            next_state, reward, done, _ = env.step(action)

            # Append experience to the buffer
            exp = Experience(state, action, reward, next_state, done)
            buffer.append(exp)

            # Update episode reward
            total_episode_reward += reward

            ### TRAINING ###
            # Perform training only if we have more than 3 elements in the buffer
            if len(buffer) >= N:
                # Sample a batch of N elements
                agent.backward(*buffer.sample_batch(n=N), discount_factor)
            
            # Update state for next iteration
            state = next_state
            t += 1

        # Append episode reward and total number of steps
        episode_reward_list.append(total_episode_reward)
        episode_number_of_steps.append(t)

        # Close environment
        env.close()

        # Updates the tqdm update bar with fresh information
        # (episode number, total reward of the last episode, total number of Steps
        # of the last episode, average reward, average number of steps)
        EPISODES.set_description(
            "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
            k, total_episode_reward, t,
            running_average(episode_reward_list, n_ep_running_average)[-1],
            running_average(episode_number_of_steps, n_ep_running_average)[-1]))

    # Saves the agent network
    torch.save(agent.network, 'neural-network-avg_'+str(running_average(episode_reward_list, n_ep_running_average)[-1])+'.pth')

    # Plot Rewards and steps
    _, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
    ax[0].plot([i for i in range(1, N_episodes+1)], episode_reward_list, label='Episode reward')
    ax[0].plot([i for i in range(1, N_episodes+1)], running_average(
        episode_reward_list, n_ep_running_average), label='Avg. episode reward')
    ax[0].set_xlabel('Episodes')
    ax[0].set_ylabel('Total reward')
    ax[0].set_title('Total Reward vs Episodes')
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    ax[1].plot([i for i in range(1, N_episodes+1)], episode_number_of_steps, label='Steps per episode')
    ax[1].plot([i for i in range(1, N_episodes+1)], running_average(
        episode_number_of_steps, n_ep_running_average), label='Avg. number of steps per episode')
    ax[1].set_xlabel('Episodes')
    ax[1].set_ylabel('Total number of steps')
    ax[1].set_title('Total number of steps vs Episodes')
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    plt.show()

def F():
    """
    Generates 3D plots of the angle and height of the lander. 
    """
    # Load model
    try:
        model = torch.load('neural-network-avg_203.pth')
        print('Network model: {}'.format(model))
    except:
        print('File neural-network-1.pth not found!')
        exit(-1)

    w = np.linspace(-math.pi, math.pi, 64)
    y = np.linspace(0, 1.5, 64)
    W, Y = np.meshgrid(w, y)

    Z = np.empty(W.shape)
    Z_arg = np.empty(W.shape)

    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            state = np.array([0, Y[i,j], 0, 0, W[i,j], 0, 0, 0])
            state_tensor = torch.tensor([state],
                                        requires_grad=True,
                                        dtype=torch.float32)
            Q_values = model.forward(state_tensor)
            
            Z[i,j] = Q_values.max(1)[0].item()
            Z_arg[i,j] = Q_values.max(1)[1].item()
            if Q_values.max(1)[1].item() == 2:
                print(Q_values.max(1)[1].item())

    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(W, Y, Z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.xlabel('w')
    plt.ylabel('y')
    plt.title('Q value')

    fig = plt.figure(2)
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(W, Y, Z_arg, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.xlabel('w')
    plt.ylabel('y')
    plt.title('Q action')
    plt.show()

if __name__ == '__main__':
    A() # Runs the main script.
    F() # Genererates the 3D plots. 