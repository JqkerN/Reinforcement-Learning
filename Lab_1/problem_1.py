"""
EL 2805 Reinforcement Learning 
Problem 1

Arthurs: 
Ilian Corneliussen 950418-2438
Daniel Hirsch 960202-5737
"""
import numpy as np


class Player():
    def __init__(self, pose, nr_states, can_stay=True):
        self.pose = pose
        if can_stay:
            self.actions={'left'    : -1,
                        'right'   : 1,
                        'up'      : -1,
                        'down'    : 1,
                        'stay'    : 0}
            self.transition_matrix = {  'left' : np.zeros((nr_states, nr_states)),
                                        'right': np.zeros((nr_states, nr_states)),
                                        'up'   : np.zeros((nr_states, nr_states)),
                                        'down' : np.zeros((nr_states, nr_states)),
                                        'stay' : np.zeros((nr_states, nr_states))}

        else:
            self.actions={'left'    : -1,
                        'right'   : 1,
                        'up'      : -1,
                        'down'    : 1}
            self.transition_matrix = {  'left' : np.zeros((nr_states, nr_states)),
                                        'right': np.zeros((nr_states, nr_states)),
                                        'up'   : np.zeros((nr_states, nr_states)),
                                        'down' : np.zeros((nr_states, nr_states))}



class Enviroment():
    def __init__(self, start=(0,0), goal=(7,6), T=20):
        # 0=valid space
        # 1=wall
        self.map =  np.matrix( [[0, 0, 1, 0, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0, 1, 0, 0],
                                [0, 0, 1, 0, 0, 1, 1, 1],
                                [0, 0, 1, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 1, 1, 1, 1, 1, 1, 0],
                                [0, 0, 0, 0, 1, 0, 0, 0]])
        self.nr_states = self.map.size
        self.row, self.col = self.map.shape
        self.start = start
        self.goal = goal
        self.human = Player(start, self.nr_states)
        self.minotaur = Player(goal, self.nr_states, can_stay=False)
        self.T = T

    def transition_probability(self):
        # NOTE: HUMAN
        # Human transition probability: It can move left, right, down, up and stay
        # at the current position.
        verbose_human = False # Set to True for prints.
        for r in range(self.row):
            for c in range(self.col):
                nr_possible_moves = 0   # nr of possible moves the minotaur can make at current state
                moves = []              # The coordinates for that each move.
                if verbose_human:
                    print("\n______________________")
                    print("r: {}, c: {}".format(r,c))
                for action in self.human.actions:    # Iterate through all of the possible actions
                    next_move = [r,c]

                    # Wall: not a valid state
                    if self.map[tuple(next_move)] != 0: 
                            continue

                    # Right: checks for possible move            
                    if action == 'right':
                        next_move[1] += 1
                        if next_move[1] >= self.col:
                            break
                        elif self.map[tuple(next_move)] == 0:
                            nr_possible_moves += 1
                            moves.append([action, tuple(next_move)])
                            break

                    # Left: checks for possible move
                    if action == 'left':
                        next_move[1] -= 1
                        if next_move[1] < 0:
                            break
                        elif self.map[tuple(next_move)] == 0:
                            nr_possible_moves += 1
                            moves.append([action, tuple(next_move)])
                            break

                    # Up: checks for possible move
                    if action == 'down':
                        next_move[0] += 1
                        if next_move[0] >= self.row:
                            break
                        elif self.map[tuple(next_move)] == 0:
                            nr_possible_moves += 1
                            moves.append([action, tuple(next_move)])
                            break

                    # Down: checks for possible move
                    if action == 'up':
                        next_move[0] -= 1
                        if next_move[0] < 0:
                            break
                        elif self.map[tuple(next_move)] == 0:
                            nr_possible_moves += 1
                            moves.append([action, tuple(next_move)])
                            break

                if verbose_human:
                    print("moves: {} \nNr of possible moves: {}".format(moves, nr_possible_moves))

                for move in moves: 
                    current_state = r*self.col + c
                    move_to_state = move[1][0]*self.col + move[1][1]
                    # print("Action: {},\t Move: {},\t state: {}".format(move[0], move[1], move_to_state))
                    self.minotaur.transition_matrix[move[0]][current_state, move_to_state] += 1
        # print(self.minotaur.transition_matrix['left'])

        


        # NOTE: MINOTAUR
        # Minotaur transition probability: It can move left, right, down, up and (ev. stay
        # at the current position). The minotaur can walk through walls. 
        verbose = False # Set to True for prints.
        for r in range(self.row):
            for c in range(self.col):
                nr_possible_moves = 0   # nr of possible moves the minotaur can make at current state
                moves = []              # The coordinates for that each move.
                if verbose:
                    print("\n______________________")
                    print("r: {}, c: {}".format(r,c))
                for action in self.minotaur.actions:    # Iterate through all of the possible actions
                    next_move = [r,c]
                    if self.map[tuple(next_move)] != 0: # If at a wall => not a valid state
                            continue

                    while True:
                        # Right: checks for possible move
                        if action == 'right':
                            next_move[1] += 1
                            if next_move[1] >= self.col:
                                break
                            elif self.map[tuple(next_move)] == 0:
                                nr_possible_moves += 1
                                moves.append([action, tuple(next_move)])
                                break

                        # Left: checks for possible move
                        if action == 'left':
                            next_move[1] -= 1
                            if next_move[1] < 0:
                                break
                            elif self.map[tuple(next_move)] == 0:
                                nr_possible_moves += 1
                                moves.append([action, tuple(next_move)])
                                break

                        # Up: checks for possible move
                        if action == 'down':
                            next_move[0] += 1
                            if next_move[0] >= self.row:
                                break
                            elif self.map[tuple(next_move)] == 0:
                                nr_possible_moves += 1
                                moves.append([action, tuple(next_move)])
                                break

                        # Down: checks for possible move
                        if action == 'up':
                            next_move[0] -= 1
                            if next_move[0] < 0:
                                break
                            elif self.map[tuple(next_move)] == 0:
                                nr_possible_moves += 1
                                moves.append([action, tuple(next_move)])
                                break
                if verbose:
                    print("moves: {} \nNr of possible moves: {}".format(moves, nr_possible_moves))
                for move in moves: 
                    current_state = r*self.col + c
                    move_to_state = move[1][0]*self.col + move[1][1]
                    # print("Action: {},\t Move: {},\t state: {}".format(move[0], move[1], move_to_state))
                    self.minotaur.transition_matrix[move[0]][current_state, move_to_state] += 1/nr_possible_moves
        # print(self.minotaur.transition_matrix['left'])
            






    def reward_function(self):
        if self.human.pose == self.minotaur.pose:
            return -1
        elif self.human.pose == self.goal.pose:
            return 1
        elif self.space[self.human.pose] == 1:
            return -1000
        return 0




def main():
    game = Enviroment()
    game.transition_probability()




if __name__ == '__main__':
    main()
