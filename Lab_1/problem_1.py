"""
    KTH: EL 2805 Reinforcement Learning 
               November 2020
                Problem 1

                Arthurs: 
Ilian Corneliussen  950418-2438  ilianc@kth.se
Daniel Hirsch       960202-5737  dhirsch@kth.se   
"""
import numpy as np
import sys
# np.set_printoptions(threshold=sys.maxsize) # Allows the whole matriz to be printed.

class Player():
    def __init__(self, pose, nr_states, can_stay=True):
        self.pose = pose
        empty_matrix = np.zeros((nr_states, nr_states))

        #             Action     Valid Action [bool]        Row Impact [%d]    Column Impact [%d]
        self.actions={'left' :   {'valid?': True,           'row':  0,         'column': -1},
                      'right':   {'valid?': True,           'row':  0,         'column':  1},
                      'up'   :   {'valid?': True,           'row': -1,         'column':  0},
                      'down' :   {'valid?': True,           'row':  1,         'column':  0},
                      'stay' :   {'valid?': can_stay,       'row':  0,         'column':  0}}

        #                         Action        Valid Action [bool]     probability matrix [2x2: %f]
        self.transition_matrix = {'left' :      {'valid?': True,       'probability': empty_matrix},
                                  'right':      {'valid?': True,       'probability': empty_matrix},
                                  'up'   :      {'valid?': True,       'probability': empty_matrix},
                                  'down' :      {'valid?': True,       'probability': empty_matrix},
                                  'stay' :      {'valid?': can_stay,   'probability': empty_matrix}}
    

    def generate_transition_probability(self, enviroment, wall_hack=False, verbose=False):
        """ Description: Creates the transition probability matrix
            for the human and the beast.
        """
        for r in range(enviroment.row):
            for c in range(enviroment.column):
                nr_possible_moves = 0   # nr of possible moves the player can make at current state
                moves = []              # The coordinates for that each move.
                if verbose:
                    print("\n______________________")
                    print("r: {}, c: {}".format(r,c))
                for action in self.actions:    # Iterate through all of the possible actions
                    next_move = [r,c]   # Creates the next move

                    # Check that the action is a valid action
                    if self.actions[action]['valid?'] == False:
                        continue

                    # Wall: Not a valid state
                    if enviroment.binary_map[tuple(next_move)] != 0: 
                            continue

                    while True:
                        # Adds the movement of the current action
                        next_move[0] += self.actions[action]['row']
                        next_move[1] += self.actions[action]['column']

                        # Checkes that we are not out-of-bounds
                        if next_move[0] < 0:
                            break
                        elif next_move[1] < 0:
                            break
                        elif next_move[0] >= enviroment.row:
                            break
                        elif next_move[1] >= enviroment.column:
                            break
                        
                        # Checks that next state is a valid state
                        elif enviroment.binary_map[tuple(next_move)] == 0:
                            nr_possible_moves += 1
                            moves.append([action, tuple(next_move)])
                            break

                        # Breaks if not wall hack is enabled        
                        if not wall_hack:
                            break

                if verbose:
                    print("moves: {} \nNr of possible moves: {}".format(moves, nr_possible_moves))

                for move in moves: # Calculates current state, next state and action.
                    state = r*enviroment.column + c
                    next_state = move[1][0]*enviroment.column + move[1][1]
                    action = move[0]
                    # print("Action: {},\t Move: {},\t state: {}".format(action, state, next_state))

                    if self.transition_matrix[action]['valid?']:
                        if wall_hack: # wall_hack implies that we have uniformly probability (Minotaur)
                            self.transition_matrix[action]['probability'][state, next_state] += 1/nr_possible_moves
                        else: # Always probability 1 for transition 
                            self.transition_matrix[action]['probability'][state, next_state] += 1

        # print(self.transition_matrix['right']['probability'])


class Enviroment():
    def __init__(self, binary_map, start=(0,0), goal=(7,6), T=20):
        #Index declaration:  0:= "valid space", 1:= "wall"
        self.binary_map =  binary_map
        self.nr_states = self.binary_map.size
        self.row, self.column = self.binary_map.shape
        self.start = start
        self.goal = goal
        self.T = T

    def reward_function(self):
        if self.human.pose == self.minotaur.pose:
            return -1
        elif self.human.pose == self.goal.pose:
            return 1
        elif self.space[self.human.pose] == 1:
            return -1000
        return 0




def main():

    # Generates the game enviroment, i.e. the maze etc. 
    binary_map =np.matrix( [[0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 1, 0, 0],
                            [0, 0, 1, 0, 0, 1, 1, 1],
                            [0, 0, 1, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 1, 1, 1, 1, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0]])

    game = Enviroment(binary_map=binary_map)

    # Generates the human player
    human = Player(pose=game.start, nr_states=binary_map.size)
    human.generate_transition_probability(enviroment=game)

    # Generates the minotaur player
    minotaur = Player(pose=game.goal, nr_states=binary_map.size, can_stay=False)
    minotaur.generate_transition_probability(enviroment=game, wall_hack=True, verbose=True)


    # Saves the files to the result folder. 
    with open("Lab_1/results/player_trans/left.txt",'w') as f:
        np.savetxt(f, human.transition_matrix["left"]["probability"], fmt='%d')
    with open("Lab_1/results/player_trans/right.txt",'w') as f:
        np.savetxt(f, human.transition_matrix["right"]["probability"], fmt='%d')
    with open("Lab_1/results/player_trans/up.txt",'w') as f:
        np.savetxt(f, human.transition_matrix["up"]["probability"], fmt='%d')
    with open("Lab_1/results/player_trans/down.txt",'w') as f:
        np.savetxt(f, human.transition_matrix["down"]["probability"], fmt='%d')
    with open("Lab_1/results/player_trans/stay.txt",'w') as f:
        np.savetxt(f, human.transition_matrix["stay"]["probability"], fmt='%d')
    
    with open("Lab_1/results/minotaur_trans/left.txt",'w') as f:
        np.savetxt(f, minotaur.transition_matrix["left"]["probability"]*100, fmt='%d')
    with open("Lab_1/results/minotaur_trans/right.txt",'w') as f:
        np.savetxt(f, minotaur.transition_matrix["right"]["probability"]*100, fmt='%d')
    with open("Lab_1/results/minotaur_trans/up.txt",'w') as f:
        np.savetxt(f, minotaur.transition_matrix["up"]["probability"]*100, fmt='%d')
    with open("Lab_1/results/minotaur_trans/down.txt",'w') as f:
        np.savetxt(f, minotaur.transition_matrix["down"]["probability"]*100, fmt='%d')
    with open("Lab_1/results/minotaur_trans/stay.txt",'w') as f:
        np.savetxt(f, minotaur.transition_matrix["stay"]["probability"]*100, fmt='%d')




if __name__ == '__main__':
    main()
