import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display
import math
from datetime import datetime
import os

# Implemented methods
methods = ['DynProg', 'ValIter']

# Some colours
LIGHT_RED    = '#FFC4CC'
LIGHT_GREEN  = '#95FD99'
BLACK        = '#000000'
WHITE        = '#FFFFFF'
LIGHT_PURPLE = '#E8D0FF'
LIGHT_ORANGE = '#FAE0C3'

class Maze:

    # Actions
    STAY       = 0
    MOVE_LEFT  = 1
    MOVE_RIGHT = 2
    MOVE_UP    = 3
    MOVE_DOWN  = 4

    # Give names to actions
    actions_names = {
        STAY: "stay",
        MOVE_LEFT: "move left",
        MOVE_RIGHT: "move right",
        MOVE_UP: "move up",
        MOVE_DOWN: "move down"
    }

    # Reward values
    STEP_REWARD = -1
    GOAL_REWARD = 0
    IMPOSSIBLE_REWARD = -math.inf
    EATEN_REWARD = -100


    def __init__(self, maze, can_stay=False, weights=None, random_rewards=False):
        """ Constructor of the environment Maze.
        """
        self.maze                     = maze
        self.actions                  = self.__actions(True)
        self.actions_minotaur         = self.__actions(can_stay)
        self.states, self.map         = self.__states()
        self.n_actions                = len(self.actions)
        self.n_actions_minotaur       = len(self.actions)
        self.n_states                 = len(self.states)
        self.transition_probabilities = self.__transitions()
        self.rewards                  = self.__rewards(weights=weights,
                                                random_rewards=random_rewards)

    def __actions(self, can_stay):
        actions = dict()
        if can_stay:
            actions[self.STAY]       = (0, 0)
        actions[self.MOVE_LEFT]  = (0,-1)
        actions[self.MOVE_RIGHT] = (0, 1)
        actions[self.MOVE_UP]    = (-1,0)
        actions[self.MOVE_DOWN]  = (1,0)
        return actions


    def __states(self):
        states = dict()
        map = dict()

        s = 0
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                for k in range(self.maze.shape[0]):
                    for l in range(self.maze.shape[1]):
                        if self.maze[i,j] != 1:
                            states[s] = ((i,j),(k,l))
                            map[((i,j),(k,l))] = s
                            s += 1
        return states, map

    def __move(self, state, action, action_minotaur):
        """ Makes a step in the maze, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.

            :return tuple next_cell: Position (x,y) on the maze that agent transitions to.
        """
        current_pos_player = self.states[state][0]
        current_pos_minotaur = self.states[state][1]

        # Compute the future position given current (state, action)
        row_player = current_pos_player[0] + self.actions[action][0]
        col_player = current_pos_player[1] + self.actions[action][1]

        row_minotaur = current_pos_minotaur[0] + self.actions_minotaur[action_minotaur][0]
        col_minotaur = current_pos_minotaur[1] + self.actions_minotaur[action_minotaur][1]

        wallhack =  (row_minotaur == -1) or (row_minotaur == self.maze.shape[0]) or \
                    (col_minotaur == -1) or (col_minotaur == self.maze.shape[1]) or \
                    (self.maze[row_minotaur, col_minotaur] == 1)

        if wallhack:
            row_minotaur += self.actions_minotaur[action_minotaur][0]
            col_minotaur += self.actions_minotaur[action_minotaur][1]

        # Is the future position an impossible one (minotaur)?
        out_of_bounds = (row_minotaur < 0) or (row_minotaur >= self.maze.shape[0]) or \
                        (col_minotaur < 0) or (col_minotaur >= self.maze.shape[1]) or \
                        (self.maze[row_minotaur, col_minotaur] == 1)


        # Is the future position an impossible one (player)?
        hitting_maze_walls =  (row_player == -1) or (row_player == self.maze.shape[0]) or \
                              (col_player == -1) or (col_player == self.maze.shape[1]) or \
                              (self.maze[row_player, col_player] == 1)

        if out_of_bounds: 
            return None
        elif hitting_maze_walls:
            return self.map[(current_pos_player, (row_minotaur, col_minotaur))]
        else:
            return self.map[((row_player, col_player), (row_minotaur, col_minotaur))]
    

    def __transitions(self):
        """ Computes the transition probabilities for every state action pair.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A
        """
        # Initialize the transition probailities tensor (S,S,A)
        dimensions = (self.n_states,self.n_states,self.n_actions)
        transition_probabilities = np.zeros(dimensions)

        # Compute the transition probabilities. Note that the transitions
        # are deterministic.
        for s in range(self.n_states):
            for a in range(self.n_actions):
                n = 0
                next_s_vec = list()
                for a_minotaur in self.actions_minotaur:
                    next_s = self.__move(s, a, a_minotaur)
                    if next_s != None:
                        n += 1
                        next_s_vec.append(next_s)
                
                for next_s in next_s_vec:
                    transition_probabilities[next_s, s, a] = 1/n
        return transition_probabilities

    def __rewards(self, weights=None, random_rewards=None):

        rewards = np.zeros((self.n_states, self.n_actions))

        # If the rewards are not described by a weight matrix
        if weights is None:
            for s in range(self.n_states):
                for a in range(self.n_actions):

                    n = 0
                    next_s_vec = list()
                    for a_minotaur in self.actions_minotaur:
                        next_s = self.__move(s, a, a_minotaur)
                        if next_s != None:
                            next_s_vec.append(next_s)
                            n += 1
                    
                    for next_s in next_s_vec:
                        # Reward for being eaten
                        if self.states[next_s][0] == self.states[next_s][1]:
                            rewards[s,a] += self.EATEN_REWARD
                        # Reward for hitting a wall
                        elif self.states[s][0] == self.states[next_s][0] and a != self.STAY:
                            rewards[s,a] += self.IMPOSSIBLE_REWARD
                        # Reward for reaching the exit
                        elif self.states[s][0] == self.states[next_s][0] and self.maze[self.states[next_s][0]] == 2:
                            rewards[s,a] += self.GOAL_REWARD
                        # Reward for taking a step to an empty cell that is not the exit
                        else:
                            rewards[s,a] += self.STEP_REWARD

                    rewards[s,a] /= n
                        

        # If the weights are descrobed by a weight matrix
        else:
            for s in range(self.n_states):
                 for a in range(self.n_actions):
                    n = 0
                    next_s_vec = list()
                    for a_minotaur in self.actions_minotaur:
                        next_s = self.__move(s, a, a_minotaur)
                        if next_s != None:
                            next_s_vec.append(next_s)
                            n += 1
                    
                    for next_s in next_s_vec:
                        i,j = self.states[next_s][0]
                        # Simply put the reward as the weights o the next state.
                        rewards[s,a] += weights[i][j]
                    
                    rewards[s,a] /= n

        return rewards

    def simulate(self, start, policy, method):
        if method not in methods:
            error = 'ERROR: the argument method must be in {}'.format(methods)
            raise NameError(error)

        path = list()
        if method == 'DynProg':
            # Deduce the horizon from the policy shape
            horizon = policy.shape[1]
            # Initialize current state and time
            t = 0
            s = self.map[start]
            # Add the starting position in the maze to the path
            path.append(start)
            while t < horizon-1:
                # Move to next state given the policy and the current state
                next_s_vec = list()
                for a_minotaur in self.actions_minotaur:
                    next_s = self.__move(s, policy[s,t], a_minotaur)
                    if next_s != None:
                        next_s_vec.append(next_s)
                next_s = np.random.choice(next_s_vec, 1)[0]
                
                # Add the position in the maze corresponding to the next state
                # to the path

                path.append(self.states[next_s])
                # Update time and state for next iteration
                t +=1
                s = next_s
                
        if method == 'ValIter':
            # Initialize current state, next state and time
            t = 1
            s = self.map[start]
            # Add the starting position in the maze to the path
            path.append(start)
            # Move to next state given the policy and the current state
            next_s = self.__move(s,policy[s])
            # Add the position in the maze corresponding to the next state
            # to the path
            path.append(self.states[next_s][0])
            # Loop while state is not the goal state
            while s != next_s:
                # Update state
                s = next_s
                # Move to next state given the policy and the current state
                next_s = self.__move(s,policy[s])
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s][0])
                # Update time and state for next iteration
                t += 1
        return path


    def show(self):
        print('The states are :')
        print(self.states)
        print('The actions are:')
        print(self.actions)
        print('The mapping of the states:')
        print(self.map)
        print('The rewards:')
        print(self.rewards)

def dynamic_programming(env, horizon):
    """ Solves the shortest path problem using dynamic programming
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input int horizon        : The time T up to which we solve the problem.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """

    # The dynamic prgramming requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p         = env.transition_probabilities
    r         = env.rewards
    n_states  = env.n_states
    n_actions = env.n_actions
    T         = horizon

    # The variables involved in the dynamic programming backwards recursions
    V      = np.zeros((n_states, T+1))
    policy = np.zeros((n_states, T+1))
    Q      = np.zeros((n_states, n_actions))


    # Initialization
    Q            = np.copy(r)
    V[:, T]      = np.max(Q,1)
    policy[:, T] = np.argmax(Q,1)

    # The dynamic programming bakwards recursion
    for t in range(T-1,-1,-1):
        # Update the value function acccording to the bellman equation
        for s in range(n_states):
            for a in range(n_actions):
                # Update of the temporary Q values
                Q[s,a] = r[s,a] + np.dot(p[:,s,a],V[:,t+1])
        # Update by taking the maximum Q value w.r.t the action a
        V[:,t] = np.max(Q,1)
        # The optimal action is the one that maximizes the Q function
        policy[:,t] = np.argmax(Q,1)
    return V, policy

def value_iteration(env, gamma, epsilon):
    """ Solves the shortest path problem using value iteration
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input float gamma        : The discount factor.
        :input float epsilon      : accuracy of the value iteration procedure.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """
    # The value itearation algorithm requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p         = env.transition_probabilities
    r         = env.rewards
    n_states  = env.n_states
    n_actions = env.n_actions

    # Required variables and temporary ones for the VI to run
    V   = np.zeros(n_states)
    Q   = np.zeros((n_states, n_actions))
    BV  = np.zeros(n_states)
    # Iteration counter
    n   = 0
    # Tolerance error
    tol = (1 - gamma)* epsilon/gamma

    # Initialization of the VI
    for s in range(n_states):
        for a in range(n_actions):
            Q[s, a] = r[s, a] + gamma*np.dot(p[:,s,a],V)
    BV = np.max(Q, 1)

    # Iterate until convergence
    while np.linalg.norm(V - BV) >= tol and n < 200:
        # Increment by one the numbers of iteration
        n += 1
        # Update the value function
        V = np.copy(BV)
        # Compute the new BV
        for s in range(n_states):
            for a in range(n_actions):
                Q[s, a] = r[s, a] + gamma*np.dot(p[:,s,a],V)
        BV = np.max(Q, 1)
        # Show error
        #print(np.linalg.norm(V - BV))

    # Compute policy
    policy = np.argmax(Q,1)
    # Return the obtained policy
    return V, policy

def draw_maze(maze):
    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED}

    # Give a color to each cell
    rows,cols    = maze.shape
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Remove the axis ticks and add title title
    ax = plt.gca()
    ax.set_title('The Maze')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    rows,cols    = maze.shape
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                            cellColours=colored_maze,
                            cellLoc='center',
                            loc=(0,0),
                            edges='closed')
    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows)
        cell.set_width(1.0/cols)
    plt.show()

def animate_solution(maze, path, can_stay=False):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED}

    # Size of the maze
    rows,cols = maze.shape

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Remove the axis ticks and add title title
    ax = plt.gca()
    ax.set_title('Policy simulation, can_stay=' + str(can_stay))
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_maze,
                     cellLoc='center',
                     loc=(0,0),
                     edges='closed')

    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows)
        cell.set_width(1.0/cols)

    stop = False        # Condition for stoping the iteration is fulfilled or not.
    history = dict()    # Keeping track of the old moves

    # Update the color at each frame
    for i in range(len(path)):
        if i > 0:
            try: # If the key has already been made
                history[path[i-1][0]] += '\nPlayer: '+str(i)
            except:
                history[path[i-1][0]] = 'Player: '+str(i)
            try: # If the key has already been made
                history[path[i-1][1]] += '\nMinotaur: '+str(i)
            except:
                history[path[i-1][1]] = 'Minotaur: '+str(i)

            # Remove the old player position and add index.
            grid.get_celld()[path[i-1][0]].set_facecolor(col_map[maze[path[i-1][0]]])
            grid.get_celld()[path[i-1][0]].get_text().set_text(history[path[i-1][0]])
            # Remove the old Minotaur postion and add index
            grid.get_celld()[path[i-1][1]].set_facecolor(col_map[maze[path[i-1][1]]])
            grid.get_celld()[path[i-1][1]].get_text().set_text(history[path[i-1][1]])
        
        # Update player position
        grid.get_celld()[path[i][0]].set_facecolor(LIGHT_ORANGE)
        grid.get_celld()[path[i][0]].get_text().set_text('Player')
        # Update Minotaur postion
        grid.get_celld()[path[i][1]].set_facecolor(LIGHT_RED)
        grid.get_celld()[path[i][1]].get_text().set_text('Minotaur')

        if i > 0:
            # Update cell if player has been eaten
            if path[i][0] == path[i][1]:
                grid.get_celld()[path[i][0]].set_facecolor(BLACK)
                grid.get_celld()[path[i][0]].get_text().set_text('OOO-NO I have been eaten!')
                print("OOO-NO I have been eaten!")
                stop = True
                
            # Update cell if player reaches goal
            elif path[i][0] == path[i-1][0] and maze[path[i][0]] == 2:
                grid.get_celld()[path[i][0]].set_facecolor(LIGHT_GREEN)
                grid.get_celld()[path[i][0]].get_text().set_text(history[path[i][0]] + '\nPlayer is out: ' + str(i))
                print("Congratulations! You reached the goal at t={}".format(i))
                stop = True
                          

        # display.display(fig)
        # display.clear_output(wait=True)
        plt.draw()
        plt.pause(0.1)
        if stop:
            break      
    plt.waitforbuttonpress(0)
    timestamp = datetime.now()
    file_path = os.path.join(os.path.realpath(os.path.dirname(__file__)), '..') 
    filename = file_path + "\images\problem_1\MazeRun_" + timestamp.strftime("%b-%d-%Y_%H-%M-%S")
    plt.savefig(fname=filename)
    plt.close(fig)

