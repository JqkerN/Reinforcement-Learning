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

class City:

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
    CAUGHT_REWARD = -50
    BANK_REWARD = 10
    IMPOSSIBLE_REWARD = -math.inf
    STEP_REWARD = -1
    


    def __init__(self, city, can_stay=False, weights=None, random_rewards=False):
        """ Constructor of the environment city.
        """
        self.city                     = city
        self.actions                  = self.__actions(True)
        self.actions_batman         = self.__actions(can_stay)
        self.states, self.map         = self.__states()
        self.n_actions                = len(self.actions)
        self.n_actions_batman       = len(self.actions)
        self.n_states                 = len(self.states)
        self.transition_probabilities = self.__transitions()
        self.rewards                  = self.__rewards(weights=weights,
                                                random_rewards=random_rewards)

    def __actions(self, can_stay):
        actions = dict()
        if can_stay:
            actions[self.STAY]   = (0, 0)
        actions[self.MOVE_LEFT]  = (0,-1)
        actions[self.MOVE_RIGHT] = (0, 1)
        actions[self.MOVE_UP]    = (-1,0)
        actions[self.MOVE_DOWN]  = (1, 0)
        return actions


    def __states(self):
        states = dict()
        map = dict()

        s = 0
        for i in range(self.city.shape[0]):
            for j in range(self.city.shape[1]):
                for k in range(self.city.shape[0]):
                    for l in range(self.city.shape[1]):
                        # if self.city[i,j] != 1:
                        states[s] = ((i,j),(k,l))
                        map[((i,j),(k,l))] = s
                        s += 1
        return states, map

    def __move(self, state, action, action_batman):
        """ Makes a step in the city, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.

            :return tuple next_cell: Position (x,y) on the city that agent transitions to.
        """
        current_pos_player, current_pos_batman = self.states[state]

        caught = False
        if current_pos_player == current_pos_batman:
            caught = True
            not_a_valid_move = False
            hitting_city_walls = False
            row_player, col_player = (0,0)
            row_batman, col_batman = (1,2)
        else:   
            # Compute the future position given current (state, action)
            row_player = current_pos_player[0] + self.actions[action][0]
            col_player = current_pos_player[1] + self.actions[action][1]

            row_batman = current_pos_batman[0] + self.actions_batman[action_batman][0]
            col_batman = current_pos_batman[1] + self.actions_batman[action_batman][1]

            # Is the future position an impossible one (minotaur)?
            not_a_valid_move =  (row_batman < 0) or (row_batman >= self.city.shape[0]) or \
                                (col_batman < 0) or (col_batman >= self.city.shape[1]) 

            if current_pos_player[0] == current_pos_batman[0] and current_pos_player[1] < current_pos_batman[1]:
                if action_batman == self.MOVE_RIGHT:
                    not_a_valid_move = True

            elif current_pos_player[0] == current_pos_batman[0] and current_pos_player[1] > current_pos_batman[1]:
                if action_batman == self.MOVE_LEFT:
                    not_a_valid_move = True

            elif current_pos_player[0] < current_pos_batman[0] and current_pos_player[1] == current_pos_batman[1]:
                if action_batman == self.MOVE_DOWN:
                    not_a_valid_move = True

            elif current_pos_player[0] > current_pos_batman[0] and current_pos_player[1] == current_pos_batman[1]:
                if action_batman == self.MOVE_UP:
                    not_a_valid_move = True

            elif current_pos_player[0] < current_pos_batman[0] and current_pos_player[1] < current_pos_batman[1]:
                if action_batman == self.MOVE_RIGHT or action_batman == self.MOVE_DOWN:
                    not_a_valid_move = True

            elif current_pos_player[0] > current_pos_batman[0] and current_pos_player[1] > current_pos_batman[1]:
                if action_batman == self.MOVE_LEFT or action_batman == self.MOVE_UP:
                    not_a_valid_move = True

            elif current_pos_player[0] > current_pos_batman[0] and current_pos_player[1] < current_pos_batman[1]:
                if action_batman == self.MOVE_RIGHT or action_batman == self.MOVE_UP:
                    not_a_valid_move = True

            elif current_pos_player[0] < current_pos_batman[0] and current_pos_player[1] > current_pos_batman[1]:
                if action_batman == self.MOVE_LEFT or action_batman == self.MOVE_DOWN:
                    not_a_valid_move = True


            # Is the future position an impossible one (player)?
            hitting_city_walls =  (row_player < 0) or (row_player >= self.city.shape[0]) or \
                                (col_player < 0) or (col_player >= self.city.shape[1]) or \
                                (self.city[row_player, col_player] == 1)

        if not_a_valid_move: 
            return None, caught
        elif hitting_city_walls:
            return self.map[(current_pos_player, (row_batman, col_batman))], caught
        else:
            return self.map[((row_player, col_player), (row_batman, col_batman))], caught
    

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
                for a_batman in self.actions_batman:
                    next_s, caught = self.__move(s, a, a_batman)
                    
                    if caught:
                        n = 1
                        next_s_vec = [next_s]
                        break

                    elif next_s != None:
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
                    for a_batman in self.actions_batman:
                        next_s, caught = self.__move(s, a, a_batman)
                        
                        if caught and next_s != None:
                            n = 1
                            next_s_vec = [next_s]
                            break

                        elif next_s != None:
                            n += 1
                            next_s_vec.append(next_s)
                    
                    for next_s in next_s_vec:
                        
                        if caught:
                            pass
                        # Reward for getting caught
                        elif self.states[next_s][0] == self.states[next_s][1]:
                            rewards[s,a] += self.CAUGHT_REWARD
                        # Reward for hitting a wall
                        elif self.states[s][0] == self.states[next_s][0] and a != self.STAY:
                            rewards[s,a] += self.IMPOSSIBLE_REWARD
                        # Reward for robbing a bank
                        elif self.city[self.states[s][0]] == 2 and self.city[self.states[next_s][0]] == 2:
                            rewards[s,a] += self.BANK_REWARD
                        # # # Reward for taking a step to an empty cell that is not the exit
                        # else:
                        #     rewards[s,a] += self.STEP_REWARD

                    rewards[s,a] /= n
                        

        # If the weights are descrobed by a weight matrix
        else:
            for s in range(self.n_states):
                 for a in range(self.n_actions):
                    n = 0
                    next_s_vec = list()
                    for a_batman in self.actions_batman:
                        next_s, caught = self.__move(s, a, a_batman)
                        
                        if caught:
                            n = 1
                            next_s_vec = [next_s]
                            break

                        elif next_s != None:
                            n += 1
                            next_s_vec.append(next_s)
                    
                    for next_s in next_s_vec:
                        i,j = self.states[next_s][0]
                        # Simply put the reward as the weights o the next state.
                        rewards[s,a] += weights[i][j]
                    
                    rewards[s,a] /= n

        return rewards

    def simulate(self, start, policy, method, iterations):
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
            # Add the starting position in the city to the path
            path.append(start)
            while t < horizon-1:
                # Move to next state given the policy and the current state
                next_s_vec = list()
                for a_batman in self.actions_batman:
                        next_s, caught = self.__move(s, policy[s,t], a_batman)
                        
                        if caught:
                            next_s_vec = [next_s]
                            break

                        elif next_s != None:
                            next_s_vec.append(next_s)

                next_s = np.random.choice(next_s_vec, 1)[0]
                
                # Add the position in the city corresponding to the next state
                # to the path

                path.append(self.states[next_s])
                # Update time and state for next iteration
                t +=1
                s = next_s
                
        if method == 'ValIter':
            # Initialize current state, next state and time
            t = 1
            s = self.map[start]
            # Add the starting position in the city to the path
            path.append(start)

            # Move to next state given the policy and the current state
            next_s_vec = list()
            for a_batman in self.actions_batman:
                next_s, caught = self.__move(s, policy[s], a_batman)
                
                if caught:
                    next_s_vec = [next_s]
                    break

                elif next_s != None:
                    next_s_vec.append(next_s)
            next_s = np.random.choice(next_s_vec, 1)[0]

            # Add the position in the city corresponding to the next state
            # to the path
            path.append(self.states[next_s])

            # Loop while state is not a terminal state
            while t <= iterations:
                # Update state
                s = next_s
                # Move to next state given the policy and the current state
                next_s_vec = list()
                for a_batman in self.actions_batman:
                        next_s, caught = self.__move(s, policy[s], a_batman)
                        
                        if caught:
                            next_s_vec = [next_s]
                            break

                        elif next_s != None:
                            next_s_vec.append(next_s)
                next_s = np.random.choice(next_s_vec, 1)[0]
                # Add the position in the city corresponding to the next state
                # to the path
                path.append(self.states[next_s])
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
        :input city env           : The city environment in which we seek to
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
        :input city env           : The city environment in which we seek to
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

def draw_city(city):
    # Map a color to each cell in the city
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED}

    # Give a color to each cell
    rows,cols    = city.shape
    colored_city = [[col_map[city[j,i]] for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the city
    fig = plt.figure(1, figsize=(cols,rows))

    # Remove the axis ticks and add title title
    ax = plt.gca()
    ax.set_title('Gotham City')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    rows,cols    = city.shape
    colored_city = [[col_map[city[j,i]] for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the city
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                            cellColours=colored_city,
                            cellLoc='center',
                            loc=(0,0),
                            edges='closed')
    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows)
        cell.set_width(1.0/cols)
    plt.show()
    plt.close(fig)

def animate_solution(city, path, method, can_stay=False, pause_time=0.5, save=False):

    # Map a color to each cell in the city
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED}

    # Size of the city
    rows,cols = city.shape

    # Create figure of the size of the city
    fig = plt.figure(1, figsize=(cols,rows))

    # Remove the axis ticks and add title title
    ax = plt.gca()
    ax.set_title('Policy simulation, can_stay=' + str(can_stay))
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    colored_city = [[col_map[city[j,i]] for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the city
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_city,
                     cellLoc='center',
                     loc=(0,0),
                     edges='closed')

    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows)
        cell.set_width(1.0/cols)

    outcome = False        # Condition for stoping the iteration is fulfilled or not.
    history = dict()    # Keeping track of the old moves

    # Update the color at each frame
    for i in range(len(path)):
        if i > 0:
            # Remove the old player position and add index.
            grid.get_celld()[path[i-1][0]].set_facecolor(col_map[city[path[i-1][0]]])
            grid.get_celld()[path[i-1][0]].get_text().set_text('')
            # Remove the old Minotaur postion and add index
            grid.get_celld()[path[i-1][1]].set_facecolor(col_map[city[path[i-1][1]]])
            grid.get_celld()[path[i-1][1]].get_text().set_text('')
        
 
        grid.get_celld()[path[i][1]].set_facecolor(LIGHT_RED)
        grid.get_celld()[path[i][1]].get_text().set_text('Batman')

        grid.get_celld()[path[i][0]].set_facecolor(LIGHT_ORANGE)
        grid.get_celld()[path[i][0]].get_text().set_text('Joker')


        if i > 0:
            # Update cell if player has been eaten
            if path[i][0] == path[i][1]:
                grid.get_celld()[path[i][0]].set_facecolor(BLACK)
                grid.get_celld()[path[i][0]].get_text().set_text('Caught')
                print("OOO-NO I have been caught!")
                outcome -= 50
                break
                
            # Update cell if player reaches goal
            elif path[i][0] == path[i-1][0] and city[path[i][0]] == 2:
                grid.get_celld()[path[i][0]].set_facecolor(LIGHT_ORANGE)
                grid.get_celld()[path[i][0]].get_text().set_text('Joker Robbing')
                # print("Robbing the bank at t={}".format(i))
                outcome += 10
                          

        # display.display(fig)
        # display.clear_output(wait=True)
        
        plt.draw()
        plt.pause(pause_time)
        # if outcome:
        #     break      
    
    # plt.waitforbuttonpress(0)
    if save:
        timestamp = datetime.now()
        file_path = os.path.join(os.path.realpath(os.path.dirname(__file__)), '..') 
        filename = file_path + "/images/problem_1/cityRun_" + method + "_" + timestamp.strftime("%b-%d-%Y_%H-%M-%S")
        plt.savefig(fname=filename)
        plt.close(fig)
    return outcome

