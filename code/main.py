
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import time
from inputimeout import inputimeout, TimeoutOccurred

# global variables
# simulation parameters
N_QAGENT = 10         #iterations of the basic value iteration agent
N_IRLAGENT = 10          #number of iterations of IRLplus to perform
EXPLORE = 0.3           #the explore proportion: (1-EXPLORE) for exloit
MANUAL_FEEDBACK = 0.1   #reward feedback from human: + and -
NEUTRAL_FEEDBACK = 0.05 #if no feedback, this reward applied (+)
LOGGING = False         #set full logging to terminal or not...
# maze setup - leave alone for now
BOARD_ROWS = 6
BOARD_COLS = 5
cargo_1 = (5, 4)
#cargo_2 = (4, 4)
warehouse_1 = (0,0)
warehouse_2 = (0,1)
block_1 = (4, 1)
block_2 = (2, 4)
block_3 = (0, 3)
START = (5, 0)          #5 row, 1 column
#hp = (0,4) ### initial human_point
success_number=0



##########################################################
# The maze environment
##########################################################
class State:
    def __init__(self, state=START):

        ###

        self.board = np.zeros([BOARD_ROWS, BOARD_COLS])
        self.board[4, 1] = -1 #block1
        self.board[2, 4] = -1 #block2
        self.board[0, 3] = -1  #block3

        self.board[5, 4] = 5  # cargo1
        #self.board[4, 4] = -50  # cargo2

        self.board[0, 0] = 100  # warehouse1
        self.board[0, 1] = -100  # warehouse2
        ###

        self.state = state
        self.isEnd = False
        self.num_cargo = 0
        self.num_warehouse = 0
        self.cargoIndex_1 = 0
        self.warehouseIndex_1 = 0


    def giveReward(self):
        ###
        if self.state == warehouse_1:
            return 100
        elif self.state == warehouse_2:
            return -100
        #elif self.state == cargo_2:
            #return -50
        #else:
            #return 0
        ###

    def isEndFunc(self,states):
        ###
        #print('END_states:',states)
        cargo_index=0
        warehouse_index=0
        #print('cargo_1:',cargo_1)
        if (cargo_1 in states) :
            cargo_index = states.index(cargo_1)
            self.num_cargo=self.num_cargo+1
            #print('self.num_cargo',self.num_cargo)
        if (warehouse_1 in states):
            warehouse_index = states.index(warehouse_1)
            self.num_warehouse = self.num_warehouse + 1
            #print('self.num_warehouse',self.num_warehouse)
        if self.num_cargo == 1:
            self.cargoIndex_1 = cargo_index
            #print('self.cargoIndex_1:',self.cargoIndex_1)
        if self.num_warehouse == 1:
            self.warehouseIndex_1 = warehouse_index
            #print('self.warehouseIndex_1',self.warehouseIndex_1)
        ###

        if ((self.state == warehouse_1) and (cargo_1 in states) and (warehouse_1 in states) and (self.cargoIndex_1 < self.warehouseIndex_1) and
                (self.cargoIndex_1!=0) and (self.warehouseIndex_1!=0) ) or (self.state == warehouse_2) :

            print('self.cargoIndex_1:',self.cargoIndex_1)
            print('self.warehouseIndex_1:',self.warehouseIndex_1)
            #print()
            self.isEnd = True

            if self.state != warehouse_2:
                print('--------       **success**       -----------')
            else:
                print('--------       **failure**       -----------')
        ###

    def nxtPosition(self, action):
        """
        action: up, down, left, right
        -------------
        0 | 1 | 2| 3|
        1 |
        2 |
        return next position
        """
        if action == "up":
            nxtState = (self.state[0] - 1, self.state[1])
        elif action == "down":
            nxtState = (self.state[0] + 1, self.state[1])
        elif action == "left":
            nxtState = (self.state[0], self.state[1] - 1)
        else:
            nxtState = (self.state[0], self.state[1] + 1)
        # if next state legal
        ###
        if (nxtState[0] >= 0) and (nxtState[0] <= 5):
            if (nxtState[1] >= 0) and (nxtState[1] <= 4):
                if nxtState != (4, 1) and nxtState != (2, 4) and nxtState != (0, 3):
                    return nxtState
        ###
        return self.state


    def showBoard(self,hp):
        ###
        self.board[self.state] = 1
        print('self.state',self.state)
        self.board[hp] = -1 ###
        #print('type',type(list(hp)[0]))
        for i in range(0, BOARD_ROWS):
            print('-----------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                if self.board[i, j] == 1 and i== list(self.state)[0] and j== list(self.state)[1]:
                    token = '*' #robot
                elif self.board[i, j] == 100 and ([i,j]==[0,0]):
                    token = 'W' #warehouse1
                elif self.board[i, j] == 100 and ([i,j]==[5,4]):
                    token = 'C' #cargo1
                elif (self.board[i, j] == -1 and ([i,j]==[4,1])) or (self.board[i, j] == -1 and ([i,j]==[2,4])) or (self.board[i, j] == -1 and ([i,j]==[0,3])):
                    token = 'K' #block
                elif self.board[i, j] == -100 and ([i,j]==[0,1]):
                    token = 'X' # warehouse2
                elif self.board[i, j] == -1 and i== list(hp)[0] and j== list(hp)[1]:
                    token = 'H' #
                elif self.board[i, j] == 0:
                    token = '0'
                out += token + ' | '
            print(out)
        print('-----------------')
        ###




##########################################################
# Agent using basic value iteration
##########################################################
class Agent:

    def __init__(self):
        self.states = []
        self.numStates = []
        self.actions = ["up", "down", "left", "right"]
        self.State = State()
        self.lr = 0.2
        self.exp_rate = EXPLORE
        self.mean_moves = 0.0

        # initial state reward
        ###
        self.state_values = {}
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                ###
                self.state_values[(i, j)] = 0  # set initial value to 0
                if (i,j) == (4, 1) or (i,j) == (2, 4) or (i,j) == (0, 3):
                    self.state_values[(i, j)] = -1 #block
                elif (i,j) == (5, 4):
                    self.state_values[(i, j)] = 100 #cargo_1
                elif (i,j) == (0, 0):
                    self.state_values[(i, j)] = 0 #initial warehouse_1,
                elif (i,j) == (0, 1):
                    self.state_values[(i, j)] = -100 # warehouse_2
        ###


    def chooseAction(self):
        # choose action with most expected value
        mx_nxt_reward = 0
        action = ""

        if np.random.uniform(0, 1) <= self.exp_rate:
            action = np.random.choice(self.actions)
        else:
            # greedy action
            for a in self.actions:
                nxt_reward = self.state_values[self.State.nxtPosition(a)]
                if nxt_reward >= mx_nxt_reward:
                    action = a
                    mx_nxt_reward = nxt_reward
                else:
                    action = 'None'
        return action

    def takeAction(self, action):
        position = self.State.nxtPosition(action)
        return State(state=position)

    def reset(self):
        self.states = []
        self.State = State()

    def play(self, rounds=10):
        i = 0
        print ("")
        print ("Q-LEARNING START")
        print ("")
        stepCounter = 0
        while i < rounds:
            # to the end of game back propagate reward
            if self.State.isEnd:
                # back propagate
                reward = self.State.giveReward()
                # explicitly assign end state to reward values
                self.state_values[self.State.state] = reward  # this is optional
                print ("--------------------------------------- Game End Reward", reward)
                print ("--------------------------------------- Num Steps Used: ", stepCounter)
                for s in reversed(self.states):
                    reward = self.state_values[s] + self.lr * (reward - self.state_values[s])
                    self.state_values[s] = round(reward, 3)
                self.reset()
                self.numStates.append(stepCounter)
                stepCounter = 0
                i += 1
            else:
                stepCounter += 1

                ###
                cycle = (stepCounter-1) / 6 #
                cycle_rest =  (stepCounter-1) % 6
                if cycle == 0:
                    if stepCounter == 1:
                        hp = (0, 2)
                    elif stepCounter == 2:
                        hp = (1, 2)
                    elif stepCounter == 3:
                        hp = (2, 2)
                    elif stepCounter == 4:
                        hp = (3, 2)
                    elif stepCounter == 5:
                        hp = (4, 2)
                    elif stepCounter == 6:
                        hp = (5, 2)
                else:
                    if cycle_rest == 0:
                        hp = (0, 2)
                    elif cycle_rest == 1:
                        hp = (1, 2)
                    elif cycle_rest == 2:
                        hp = (2, 2)
                    elif cycle_rest == 3:
                        hp = (3, 2)
                    elif cycle_rest == 4:
                        hp = (4, 2)
                    elif cycle_rest == 5:
                        hp = (5, 2)
                self.state_values[hp] = -1

                current_state = self.State.state
                reward = self.state_values[current_state] + self.lr * ((-1) - self.state_values[current_state])
                self.state_values[current_state] = round(reward, 3)
                #print('human_point', hp)
                ###


                action = self.chooseAction()


                # append trace
                self.states.append(self.State.nxtPosition(action))
                if (LOGGING):
                    print("  current position {} action {}".format(self.State.state, action))
                # by taking the action, it reaches the next state
                self.State = self.takeAction(action)


                #print('self.State',self.State)
                #print('self.State.state:', self.State.state)
                current_state = self.State.state
                if self.State.state == cargo_1:
                    ###
                    #print('cargo1_state', self.State.state)
                    self.state_values[cargo_1] = 0  # after arrive cargo1, the reward_value of cargo1 become from 50 to 0
                    self.state_values[warehouse_1] = 100  # after arrive cargo1, the reward_value of warehouse become from 0 to 100
                    reward = self.state_values[current_state]  + self.lr * (100 - self.state_values[current_state])
                    self.state_values[current_state] = round(reward, 3)
                    ###

                # mark is end
                self.State.isEndFunc(self.states) ###
                if (LOGGING):
                    print ("    |--> next state", self.State.state)

                self.state_values[hp] = 0  ###

    def showValues(self):
        print ("")
        for i in range(0, BOARD_ROWS):
            print ("-------------------------------------")
            out = '| '
            for j in range(0, BOARD_COLS):
                out += str(self.state_values[(i, j)]).ljust(6) + ' | '
            print(out)
        print ("-------------------------------------")
        print (self.numStates)
        print ("")
        self.PlotDemo()


    def PlotDemo(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        x_major_locator = MultipleLocator(1)
        ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的刻度间隔设置为1，并存在变量里
        #y_major_locator = MultipleLocator(100)  # 把y轴的刻度间隔设置为10，并存在变量里
        #ax.yaxis.set_major_locator(y_major_locator)# 把y轴的主刻度设置为10的倍数
        ax.set_ylim(0, max(self.numStates))  # set the range of y axis
        ax.plot([1,2,3,4,5,6,7,8,9,10], self.numStates)
        plt.show()

'''
##########################################################
# Interactive RL agent 
##########################################################
class IRLAgent:

    def __init__(self):
        self.states = []
        self.numStates = []
        self.actions = ["up", "down", "left", "right"]
        self.State = State()
        self.lr = 0.2
        self.exp_rate = EXPLORE

        # initial state reward
        ###
        self.state_values = {}
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                ###
                self.state_values[(i, j)] = 0  # set initial value to 0
                if (i, j) == (4, 1) or (i, j) == (2, 4) or (i, j) == (0, 3):
                    self.state_values[(i, j)] = -1  # block
                elif (i, j) == (5, 4):
                    self.state_values[(i, j)] = 100  # cargo_1
                elif (i, j) == (0, 0):
                    self.state_values[(i, j)] = 0  # initial warehouse_1,
                elif (i, j) == (0, 1):
                    self.state_values[(i, j)] = -100  # warehouse_2
        ###

    def chooseAction(self):
        # choose action with most expected value
        mx_nxt_reward = 0
        action = ""

        if np.random.uniform(0, 1) <= self.exp_rate:
            action = np.random.choice(self.actions)
        else:
            # greedy action
            for a in self.actions:
                nxt_reward = self.state_values[self.State.nxtPosition(a)]
                if nxt_reward >= mx_nxt_reward:
                    action = a
                    mx_nxt_reward = nxt_reward
                else:
                    action = 'None'
        return action

    def takeAction(self, action):
        position = self.State.nxtPosition(action)
        return State(state=position)

    def reset(self):
        self.states = []
        self.State = State()

    def play(self, rounds=3):
        i = 0
        print ("")
        print ("IRL START")
        print ("")
        stepCounter = 0
        while i < rounds:
            # to the end of game back propagate reward
            if self.State.isEnd:
                # back propagate
                reward = self.State.giveReward()
                # explicitly assign end state to reward values
                self.state_values[self.State.state] = reward  # this is optional
                print ("--------------------------------------- Game End Reward:", reward)
                print ("--------------------------------------- Num Steps Used: ", stepCounter)
                for s in reversed(self.states):
                    reward = self.state_values[s] + self.lr * (reward - self.state_values[s])
                    self.state_values[s] = round(reward, 3)
                self.reset()
                self.numStates.append(stepCounter)
                stepCounter = 0
                i += 1
            else:
                print("_________________________________________________")
                stepCounter += 1
                ###
                # print('self.State.state:',self.State.state)
                ###
                cycle = (stepCounter - 1) / 6  #
                cycle_rest = (stepCounter - 1) % 6
                if cycle == 0:
                    if stepCounter == 1:
                        hp = (0, 2)
                    elif stepCounter == 2:
                        hp = (1, 2)
                    elif stepCounter == 3:
                        hp = (2, 2)
                    elif stepCounter == 4:
                        hp = (3, 2)
                    elif stepCounter == 5:
                        hp = (4, 2)
                    elif stepCounter == 6:
                        hp = (5, 2)
                else:
                    if cycle_rest == 0:
                        hp = (0, 2)
                    elif cycle_rest == 1:
                        hp = (1, 2)
                    elif cycle_rest == 2:
                        hp = (2, 2)
                    elif cycle_rest == 3:
                        hp = (3, 2)
                    elif cycle_rest == 4:
                        hp = (4, 2)
                    elif cycle_rest == 5:
                        hp = (5, 2)
                self.state_values[hp] = -1

                current_state = self.State.state
                reward = self.state_values[current_state] + self.lr * ((-1) - self.state_values[current_state])
                self.state_values[current_state] = round(reward, 3)
                # print('human_point', hp)
                ###

                self.State.showBoard(hp)
                action = self.chooseAction()
                # append trace
                self.states.append(self.State.nxtPosition(action))
                current_state = self.State.state    #current state before action is executed
                if (LOGGING):
                    print("  current position {} action {}".format(self.State.state, action))

                print('self.State.state:', self.State.state)
                if self.State.state == cargo_1:
                    ###
                    # print('cargo1_state', self.State.state)
                    self.state_values[
                        cargo_1] = 0  # after arrive cargo1, the reward_value of cargo1 become from 50 to 0
                    self.state_values[
                        warehouse_1] = 100  # after arrive cargo1, the reward_value of warehouse become from 0 to 100
                    reward = self.state_values[current_state] + self.lr * (100 - self.state_values[current_state])
                    self.state_values[current_state] = round(reward, 3)
                    ###

                # by taking the action, it reaches the next state
                self.State = self.takeAction(action)
                # mark is end
                self.State.isEndFunc(self.states)
                if (LOGGING):
                    print ("    |--> next state", self.State.state)

                # 区别之处与Q-learning，还是选择用𝜖−贪婪法探索，但是在探索过程中的每一步都加入人工的reward
                # Q-learning是最后后退更新每一个state的reward，在更新之前,所有state的reward为0（except for 1 and -1），
                # IRL也是最后后退更新每一个state的reward，在更新之前,所有state的reward不为0，而是已经加入人工的初始化rewards.
                # for IRL allow user to define reward:
                #  - get reward from user:
                feedback = input("      *was* this g(ood) or b(ad): ")
                u_reward = 0
                if feedback == "g":
                    u_reward = MANUAL_FEEDBACK
                elif feedback == "b":
                    u_reward = -MANUAL_FEEDBACK
                else:
                    #not recognised, assume ok...
                    u_reward = NEUTRAL_FEEDBACK
                #  - update the value of the current state only
                reward = self.state_values[current_state] + self.lr * (u_reward - self.state_values[current_state])
                self.state_values[current_state] = round(reward, 3)

                self.showValues()
                self.state_values[hp] = 0  ### aviod human in time

    def showValues(self):
        for i in range(0, BOARD_ROWS):
            print ("-------------------------------------")
            out = '| '
            for j in range(0, BOARD_COLS):
                out += str(self.state_values[(i, j)]).ljust(6) + ' | '
            print(out)
        print ("-------------------------------------")
        print (self.numStates)
        print ("")
'''

##########################################################
# Interactive RL agent - changed from week 2
##########################################################
class IRLAgentPlus:

    def __init__(self):
        self.states = []
        self.numStates = []
        self.actions = ["up", "down", "left", "right"]
        self.State = State()
        self.lr = 0.2
        self.exp_rate = EXPLORE

        # initial state reward
        ###
        self.state_values = {}
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                ###
                self.state_values[(i, j)] = 0  # set initial value to 0
                if (i, j) == (4, 1) or (i, j) == (2, 4) or (i, j) == (0, 3):
                    self.state_values[(i, j)] = -1  # block
                elif (i, j) == (5, 4):
                    self.state_values[(i, j)] = 100  # cargo_1
                elif (i, j) == (0, 0):
                    self.state_values[(i, j)] = 0  # initial warehouse_1,
                elif (i, j) == (0, 1):
                    self.state_values[(i, j)] = -100 #warehouse_2
        ###


    def chooseAction(self, rand=False, actAvoid=0):
        # choose action with most expected value,
        #  unless we want an explicitly different action...
        mx_nxt_reward = 0
        action = ""

        # when the proposed action bad, choice the action which is not the bad we think, and execute，then feedback the reward to the executed action
        if rand == True:    #this part intentionally left uncommented: what does it do?
            flag = True
            while (flag):
                action = np.random.choice(self.actions)
                if action == actAvoid:
                    flag = True
                else:
                    flag = False
            return action

        if np.random.uniform(0, 1) <= self.exp_rate:
            action = np.random.choice(self.actions)
        else:
            # greedy action
            for a in self.actions:
                nxt_reward = self.state_values[self.State.nxtPosition(a)]
                if nxt_reward >= mx_nxt_reward:
                    action = a
                    mx_nxt_reward = nxt_reward
                else:
                    action = 'None'
        return action

    def takeAction(self, action):
        position = self.State.nxtPosition(action)
        return State(state=position)

    def reset(self):
        self.states = []
        self.State = State()

    def play(self, rounds=1):
        i = 0
        print ("")
        print ("IRLplus START")
        print ("")
        stepCounter = 0
        while i < rounds:
            # to the end of game back propagate reward
            if self.State.isEnd:
                # back propagate
                reward = self.State.giveReward()
                # explicitly assign end state to reward values
                self.state_values[self.State.state] = reward  # this is optional
                print ("--------------------------------------- Game End Reward:", reward)
                print ("--------------------------------------- Num Steps Used: ", stepCounter)
                for s in reversed(self.states):
                    reward = self.state_values[s] + self.lr * (reward - self.state_values[s])
                    self.state_values[s] = round(reward, 3)
                self.reset()
                self.numStates.append(stepCounter)
                stepCounter = 0
                i += 1
            else:
                print("_________________________________________________")
                stepCounter += 1

                ###
                #print('self.State.state:',self.State.state)
                ###
                cycle = (stepCounter - 1) / 6  #
                cycle_rest = (stepCounter - 1) % 6
                if cycle == 0:
                    if stepCounter == 1:
                        hp = (0, 2)
                    elif stepCounter == 2:
                        hp = (1, 2)
                    elif stepCounter == 3:
                        hp = (2, 2)
                    elif stepCounter == 4:
                        hp = (3, 2)
                    elif stepCounter == 5:
                        hp = (4, 2)
                    elif stepCounter == 6:
                        hp = (5, 2)
                else:
                    if cycle_rest == 0:
                        hp = (0, 2)
                    elif cycle_rest == 1:
                        hp = (1, 2)
                    elif cycle_rest == 2:
                        hp = (2, 2)
                    elif cycle_rest == 3:
                        hp = (3, 2)
                    elif cycle_rest == 4:
                        hp = (4, 2)
                    elif cycle_rest == 5:
                        hp = (5, 2)
                self.state_values[hp] = -1

                current_state = self.State.state
                reward = self.state_values[current_state] + self.lr * ((-1) - self.state_values[current_state])
                self.state_values[current_state] = round(reward, 3)
                # print('human_point', hp)
                ###

                self.State.showBoard(hp) ###
                action = self.chooseAction()


                print("Next action chosen: ", action)
                # for IRLplus, allow user to judge the proposed action:
                #  - get feedback from user:
                feedback = input("      *will* this action be g(ood) or b(ad): ")    #if using python2, may need to replace "input" with "raw_input"
                u_reward = 0
                if feedback == "g":
                    action = action
                elif feedback == "b":
                    #choose a new action, not the same as existing...
                    newAction = 0
                    newAction = self.chooseAction(True, action)
                    action = newAction
                else:
                    #not recognised, assume ok, so carry on...
                    action = action
                # append trace
                self.states.append(self.State.nxtPosition(action))
                print('self.states:',self.states)
                current_state = self.State.state    #current state before action is executed

                print('self.State.state:',self.State.state)
                if self.State.state == cargo_1:
                    ###
                    # print('cargo1_state', self.State.state)
                    self.state_values[cargo_1] = 0  # after arrive cargo1, the reward_value of cargo1 become from 50 to 0
                    self.state_values[warehouse_1] = 100  # after arrive cargo1, the reward_value of warehouse become from 0 to 100
                    reward = self.state_values[current_state] + self.lr * (100 - self.state_values[current_state])
                    self.state_values[current_state] = round(reward, 3)
                    ###


                # by taking the action, it reaches the next state
                self.State = self.takeAction(action)

                # mark is end
                self.State.isEndFunc(self.states) ###
                self.State.showBoard(hp) ###

                #  - get reward from user:
                print("      action actually used: ", action)
                feedback = input("      /was/ this action g(ood) or b(ad): ")
                u_reward = 0
                if feedback == "g":
                    u_reward = MANUAL_FEEDBACK
                    # keep the same action as proposed
                elif feedback == "b":
                    u_reward = -MANUAL_FEEDBACK
                else:
                    #not recognised, assume ok, so carry on...
                    u_reward = NEUTRAL_FEEDBACK
                #  - update the value of the current state only
                reward = self.state_values[current_state] + self.lr * (u_reward - self.state_values[current_state])
                self.state_values[current_state] = round(reward, 3)

                self.showValues()
                self.state_values[hp] = 0  ### aviod human in time



    def showValues(self):
        for i in range(0, BOARD_ROWS):
            print ("-------------------------------------")
            out = '| '
            for j in range(0, BOARD_COLS):
                out += str(self.state_values[(i, j)]).ljust(6) + ' | '
            print(out)
        print ("-------------------------------------")
        print (self.numStates)
        print ("")
        #self.PlotDemo()
        print('Human_guided:', 2 * sum(self.numStates))

    def PlotDemo(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        x_major_locator = MultipleLocator(1)
        ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的刻度间隔设置为1，并存在变量里
        # y_major_locator = MultipleLocator(100)  # 把y轴的刻度间隔设置为10，并存在变量里
        # ax.yaxis.set_major_locator(y_major_locator)# 把y轴的主刻度设置为10的倍数
        ax.set_ylim(0, max(self.numStates))  # set the range of y axis
        ax.set_xlabel('episodes')
        plt.ylabel("number of actions")
        ax.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], self.numStates, label='number of actions')
        plt.show()






##########################################################
# Interactive RL agent_one - changed from week 2
##########################################################
class IRLAgentPlus_one:

    def __init__(self):
        self.states = []
        self.numStates = []
        self.actions = ["up", "down", "left", "right"]
        self.State = State()
        self.lr = 0.2
        self.exp_rate = EXPLORE
        self.auto_action = 0
        self.time = True

        # initial state reward
        ###
        self.state_values = {}
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                ###
                self.state_values[(i, j)] = 0  # set initial value to 0
                if (i, j) == (4, 1) or (i, j) == (2, 4) or (i, j) == (0, 3):
                    self.state_values[(i, j)] = -1  # block
                elif (i, j) == (5, 4):
                    self.state_values[(i, j)] = 100  # cargo_1
                elif (i, j) == (0, 0):
                    self.state_values[(i, j)] = 0  # initial warehouse_1,
                elif (i, j) == (0, 1):
                    self.state_values[(i, j)] = -100 #warehouse_2
        ###


    def chooseAction(self, rand=False, actAvoid=0):
        # choose action with most expected value,
        #  unless we want an explicitly different action...
        mx_nxt_reward = 0
        action = ""

        # when the proposed action bad, choice the action which is not the bad we think, and execute，then feedback the reward to the executed action
        if rand == True:    #this part intentionally left uncommented: what does it do?
            flag = True
            while (flag):
                action = np.random.choice(self.actions)
                if action == actAvoid:
                    flag = True
                else:
                    flag = False
            return action

        if np.random.uniform(0, 1) <= self.exp_rate:
            action = np.random.choice(self.actions)
        else:
            # greedy action
            for a in self.actions:
                nxt_reward = self.state_values[self.State.nxtPosition(a)]
                if nxt_reward >= mx_nxt_reward:
                    action = a
                    mx_nxt_reward = nxt_reward
                else:
                    action = 'None'
        return action

    def takeAction(self, action):
        position = self.State.nxtPosition(action)
        return State(state=position)

    def reset(self):
        self.states = []
        self.State = State()

    def play(self, rounds=1):
        i = 0
        print ("")
        print ("IRLplus START")
        print ("")
        stepCounter = 0
        while i < rounds:
            # to the end of game back propagate reward
            if self.State.isEnd:
                # back propagate
                reward = self.State.giveReward()
                # explicitly assign end state to reward values
                self.state_values[self.State.state] = reward  # this is optional
                print ("--------------------------------------- Game End Reward:", reward)
                print ("--------------------------------------- Num Steps Used: ", stepCounter)
                for s in reversed(self.states):
                    reward = self.state_values[s] + self.lr * (reward - self.state_values[s])
                    self.state_values[s] = round(reward, 3)
                self.reset()
                self.numStates.append(stepCounter)
                stepCounter = 0
                i += 1
            else:
                print("_________________________________________________")
                stepCounter += 1

                ###
                #print('self.State.state:',self.State.state)
                ###
                cycle = (stepCounter - 1) / 6  #
                cycle_rest = (stepCounter - 1) % 6
                if cycle == 0:
                    if stepCounter == 1:
                        hp = (0, 2)
                    elif stepCounter == 2:
                        hp = (1, 2)
                    elif stepCounter == 3:
                        hp = (2, 2)
                    elif stepCounter == 4:
                        hp = (3, 2)
                    elif stepCounter == 5:
                        hp = (4, 2)
                    elif stepCounter == 6:
                        hp = (5, 2)
                else:
                    if cycle_rest == 0:
                        hp = (0, 2)
                    elif cycle_rest == 1:
                        hp = (1, 2)
                    elif cycle_rest == 2:
                        hp = (2, 2)
                    elif cycle_rest == 3:
                        hp = (3, 2)
                    elif cycle_rest == 4:
                        hp = (4, 2)
                    elif cycle_rest == 5:
                        hp = (5, 2)
                self.state_values[hp] = -1

                current_state = self.State.state
                reward = self.state_values[current_state] + self.lr * ((-1) - self.state_values[current_state])
                self.state_values[current_state] = round(reward, 3)
                # print('human_point', hp)
                ###

                self.State.showBoard(hp) ###
                action = self.chooseAction()


                print("Next action chosen: ", action)
                # for IRLplus, allow user to judge the proposed action:
                #  - get feedback from user:

                #feedback = input("      *will* this action be g(ood) or b(ad): ")    #if using python2, may need to replace "input" with "raw_input"
                u_reward = 0

                ###
                try:
                    # 3s input for limition, otherwise automatically excute the action proposed
                    feedback = inputimeout(prompt='You have 3 seconds to input：', timeout=3)
                except TimeoutOccurred:
                    feedback = 'g'
                    self.auto_action = self.auto_action+1
                    print('self.auto_action:',self.auto_action)

                if feedback == "g":
                    action = action
                elif feedback == "b":
                    #choose a new action, not the same as existing...
                    newAction = 0
                    newAction = self.chooseAction(True, action)
                    action = newAction
                else:
                    #not recognised, assume ok, so carry on...
                    action = action
                # append trace
                self.states.append(self.State.nxtPosition(action))
                print('self.states:',self.states)
                current_state = self.State.state    #current state before action is executed

                print('self.State.state:',self.State.state)
                if self.State.state == cargo_1:
                    ###
                    # print('cargo1_state', self.State.state)
                    self.state_values[cargo_1] = 0  # after arrive cargo1, the reward_value of cargo1 become from 50 to 0
                    self.state_values[warehouse_1] = 100  # after arrive cargo1, the reward_value of warehouse become from 0 to 100
                    reward = self.state_values[current_state] + self.lr * (100 - self.state_values[current_state])
                    self.state_values[current_state] = round(reward, 3)
                    ###


                # by taking the action, it reaches the next state
                self.State = self.takeAction(action)

                # mark is end
                self.State.isEndFunc(self.states) ###
                self.State.showBoard(hp) ###

                #  - get reward from user:
                print("      action actually used: ", action)
                feedback = input("      /was/ this action g(ood) or b(ad): ")
                u_reward = 0
                if feedback == "g":
                    u_reward = MANUAL_FEEDBACK
                    # keep the same action as proposed
                elif feedback == "b":
                    u_reward = -MANUAL_FEEDBACK
                else:
                    #not recognised, assume ok, so carry on...
                    u_reward = NEUTRAL_FEEDBACK
                #  - update the value of the current state only
                reward = self.state_values[current_state] + self.lr * (u_reward - self.state_values[current_state])
                self.state_values[current_state] = round(reward, 3)

                self.showValues()
                self.state_values[hp] = 0  ### aviod human in time

        return

    def showValues(self):
        for i in range(0, BOARD_ROWS):
            print ("-------------------------------------")
            out = '| '
            for j in range(0, BOARD_COLS):
                out += str(self.state_values[(i, j)]).ljust(6) + ' | '
            print(out)
        print ("-------------------------------------")
        print (self.numStates)
        print ("")
        #self.PlotDemo()
        print('auto_action:',self.auto_action)
        #print('Human_guided:', 2 * sum(self.numStates)-self.auto_action)

    def all_showValues(self):
        for i in range(0, BOARD_ROWS):
            print ("-------------------------------------")
            out = '| '
            for j in range(0, BOARD_COLS):
                out += str(self.state_values[(i, j)]).ljust(6) + ' | '
            print(out)
        print ("-------------------------------------")
        print (self.numStates)
        print ("")
        #self.PlotDemo()
        print('auto_action:',self.auto_action)
        print('Human_guided:', 2 * sum(self.numStates)-self.auto_action)

    def PlotDemo(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        x_major_locator = MultipleLocator(1)
        ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的刻度间隔设置为1，并存在变量里
        # y_major_locator = MultipleLocator(100)  # 把y轴的刻度间隔设置为10，并存在变量里
        # ax.yaxis.set_major_locator(y_major_locator)# 把y轴的主刻度设置为10的倍数
        ax.set_ylim(0, max(self.numStates))  # set the range of y axis
        ax.set_xlabel('episodes')
        plt.ylabel("number of actions")
        ax.plot([1, 2,3,4,5,6,7,8,9,10], self.numStates,label='number of actions')
        plt.show()


##########################################################
# Interactive RL agent_two - changed from week 2
##########################################################
class IRLAgentPlus_two:

    def __init__(self):
        self.states = []
        self.numStates = []
        self.actions = ["up", "down", "left", "right"]
        self.actions_one = ["right","left"]
        self.actions_two = ["up", "left"]

        self.State = State()
        self.lr = 0.2
        self.exp_rate = EXPLORE
        self.prefered_action = 0
        self.time = True

        # initial state reward
        ###
        self.state_values = {}
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                ###
                self.state_values[(i, j)] = 0  # set initial value to 0
                if (i, j) == (4, 1) or (i, j) == (2, 4) or (i, j) == (0, 3):
                    self.state_values[(i, j)] = -1  # block
                elif (i, j) == (5, 4):
                    self.state_values[(i, j)] = 100  # cargo_1
                elif (i, j) == (0, 0):
                    self.state_values[(i, j)] = 0  # initial warehouse_1,
                elif (i, j) == (0, 1):
                    self.state_values[(i, j)] = -100 #warehouse_2
        ###


    def chooseAction(self, rand=False, actAvoid=0, cargo1_value=0):
        # choose action with most expected value,
        #  unless we want an explicitly different action...
        mx_nxt_reward = 0
        action = ""

        # when the proposed action bad, choice the action which is not the bad we think, and execute，then feedback the reward to the executed action
        if rand == True:    #this part intentionally left uncommented: what does it do?
            flag = True
            while (flag):
                action = np.random.choice(self.actions)
                if action == actAvoid:
                    flag = True
                else:
                    flag = False
            return action

        #if np.random.uniform(0, 1) <= self.exp_rate:
            #action = np.random.choice(self.actions)
        if np.random.uniform(0, 1) <= self.exp_rate: #self.exp_rate=0.5
            print("Preference_action")
            if cargo1_value == 100:
                action = np.random.choice(self.actions_one)
            elif cargo1_value != 100:
                action = np.random.choice(self.actions_two)
            self.prefered_action = self.prefered_action + 1
        else:
            # greedy action
            for a in self.actions:
                nxt_reward = self.state_values[self.State.nxtPosition(a)]
                if nxt_reward >= mx_nxt_reward:
                    action = a
                    mx_nxt_reward = nxt_reward
                else:
                    action = 'None'
        return action

    def takeAction(self, action):
        position = self.State.nxtPosition(action)
        return State(state=position)

    def reset(self):
        self.states = []
        self.State = State()

    def play(self, rounds=1):
        i = 0
        print ("")
        print ("IRLplus START")
        print ("")
        stepCounter = 0
        while i < rounds:
            # to the end of game back propagate reward
            if self.State.isEnd:
                # back propagate
                reward = self.State.giveReward()
                # explicitly assign end state to reward values
                self.state_values[self.State.state] = reward  # this is optional
                print ("--------------------------------------- Game End Reward:", reward)
                print ("--------------------------------------- Num Steps Used: ", stepCounter)
                for s in reversed(self.states):
                    reward = self.state_values[s] + self.lr * (reward - self.state_values[s])
                    self.state_values[s] = round(reward, 3)
                self.reset()
                self.numStates.append(stepCounter)
                stepCounter = 0
                i += 1
            else:
                print("_________________________________________________")
                stepCounter += 1

                ###
                #print('self.State.state:',self.State.state)
                ###
                cycle = (stepCounter - 1) / 6  #
                cycle_rest = (stepCounter - 1) % 6
                if cycle == 0:
                    if stepCounter == 1:
                        hp = (0, 2)
                    elif stepCounter == 2:
                        hp = (1, 2)
                    elif stepCounter == 3:
                        hp = (2, 2)
                    elif stepCounter == 4:
                        hp = (3, 2)
                    elif stepCounter == 5:
                        hp = (4, 2)
                    elif stepCounter == 6:
                        hp = (5, 2)
                else:
                    if cycle_rest == 0:
                        hp = (0, 2)
                    elif cycle_rest == 1:
                        hp = (1, 2)
                    elif cycle_rest == 2:
                        hp = (2, 2)
                    elif cycle_rest == 3:
                        hp = (3, 2)
                    elif cycle_rest == 4:
                        hp = (4, 2)
                    elif cycle_rest == 5:
                        hp = (5, 2)
                self.state_values[hp] = -1

                current_state = self.State.state
                reward = self.state_values[current_state] + self.lr * ((-1) - self.state_values[current_state])
                self.state_values[current_state] = round(reward, 3)
                # print('human_point', hp)
                ###

                self.State.showBoard(hp) ###
                action = self.chooseAction(False, 0, self.state_values[cargo_1])


                print("Next action chosen: ", action)
                # for IRLplus, allow user to judge the proposed action:
                #  - get feedback from user:

                #feedback = input("      *will* this action be g(ood) or b(ad): ")    #if using python2, may need to replace "input" with "raw_input"
                u_reward = 0

                #  - get feedback from user:
                feedback = input("      *will* this action be g(ood) or b(ad): ")  # if using python2, may need to replace "input" with "raw_input"
                u_reward = 0
                if feedback == "g":
                    action = action
                elif feedback == "b":
                    # choose a new action, not the same as existing...
                    newAction = 0
                    newAction = self.chooseAction(True, action)
                    action = newAction
                else:
                    # not recognised, assume ok, so carry on...
                    action = action
                # append trace
                self.states.append(self.State.nxtPosition(action))
                print('self.states:',self.states)
                current_state = self.State.state    #current state before action is executed

                print('self.State.state:',self.State.state)
                if self.State.state == cargo_1:
                    ###
                    # print('cargo1_state', self.State.state)
                    self.state_values[cargo_1] = 0  # after arrive cargo1, the reward_value of cargo1 become from 50 to 0
                    self.state_values[warehouse_1] = 100  # after arrive cargo1, the reward_value of warehouse become from 0 to 100
                    reward = self.state_values[current_state] + self.lr * (100 - self.state_values[current_state])
                    self.state_values[current_state] = round(reward, 3)
                    ###

                # by taking the action, it reaches the next state
                self.State = self.takeAction(action)

                # mark is end
                self.State.isEndFunc(self.states) ###
                self.State.showBoard(hp) ###

                #  - get reward from user:
                print("      action actually used: ", action)
                feedback = input("      /was/ this action g(ood) or b(ad): ")
                u_reward = 0
                if feedback == "g":
                    u_reward = MANUAL_FEEDBACK
                    # keep the same action as proposed
                elif feedback == "b":
                    u_reward = -MANUAL_FEEDBACK
                else:
                    #not recognised, assume ok, so carry on...
                    u_reward = NEUTRAL_FEEDBACK
                #  - update the value of the current state only
                reward = self.state_values[current_state] + self.lr * (u_reward - self.state_values[current_state])
                self.state_values[current_state] = round(reward, 3)

                self.showValues()
                self.state_values[hp] = 0  ### aviod human in time

        return

    def showValues(self):
        for i in range(0, BOARD_ROWS):
            print ("-------------------------------------")
            out = '| '
            for j in range(0, BOARD_COLS):
                out += str(self.state_values[(i, j)]).ljust(6) + ' | '
            print(out)
        print ("-------------------------------------")
        print (self.numStates)
        print ("")
        #self.PlotDemo()
        print('prefered_action:',self.prefered_action)
        #print('Human_guided:', 2 * sum(self.numStates)-self.auto_action)

    def all_showValues(self):
        for i in range(0, BOARD_ROWS):
            print ("-------------------------------------")
            out = '| '
            for j in range(0, BOARD_COLS):
                out += str(self.state_values[(i, j)]).ljust(6) + ' | '
            print(out)
        print ("-------------------------------------")
        print (self.numStates)
        print ("")
        #self.PlotDemo()
        print('prefered_action:',self.prefered_action)
        print('Human_guided:', 2 * sum(self.numStates))

    def PlotDemo(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        x_major_locator = MultipleLocator(1)
        ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的刻度间隔设置为1，并存在变量里
        # y_major_locator = MultipleLocator(100)  # 把y轴的刻度间隔设置为10，并存在变量里
        # ax.yaxis.set_major_locator(y_major_locator)# 把y轴的主刻度设置为10的倍数
        ax.set_ylim(0, max(self.numStates))  # set the range of y axis
        ax.set_xlabel('episodes')
        plt.ylabel("number of actions")
        ax.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], self.numStates, label='number of actions')
        plt.show()


##########################################################
# Main
##########################################################
if __name__ == "__main__":

    '''
    ag = Agent()
    ag.play(N_QAGENT)
    print("_________________________________________________")
    print("")
    print("Q-learning agent: ", N_QAGENT, " iterations")
    print(ag.showValues())
    '''



    '''
    irl = IRLAgent()
    irl.play(3)            # <--- Uncomment this to enable the IRLAgent
    print("_________________________________________________")
    print("")
    print ("")
    print ("IRL agent:")
    print(irl.showValues())
    print("")
    '''


    '''
    irlp = IRLAgentPlus()
    irlp.play(N_IRLAGENT)            # <--- Uncomment this to enable IRLAgentPlus
    print ("_________________________________________________")
    print ("")
    print ("")
    print ("IRLplus agent: ", N_IRLAGENT, " iterations")
    print(irlp.showValues())
    irlp.PlotDemo()
    '''


    '''
    irlp = IRLAgentPlus_one()
    irlp.play(N_IRLAGENT)  # <--- Uncomment this to enable IRLAgentPlus
    print("_________________________________________________")
    print("")
    print("")
    print("IRLplus_one agent: ", N_IRLAGENT, " iterations")
    print(irlp.all_showValues())
    irlp.PlotDemo()
    '''



    irlp = IRLAgentPlus_two()
    irlp.play(N_IRLAGENT)  # <--- Uncomment this to enable IRLAgentPlus
    print("_________________________________________________")
    print("")
    print("")
    print("IRLplus_two agent: ", N_IRLAGENT, " iterations")
    print(irlp.all_showValues())
    irlp.PlotDemo()









