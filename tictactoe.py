from __future__ import print_function
from collections import defaultdict
from itertools import count
import numpy as np
from numpy import random as rd
import matplotlib.pyplot as plt
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions
from torch.autograd import Variable
import os
#import sys

os.chdir('/Users/arielkelman/Documents/Ariel/EngSci3-PhysicsOption/Winter2018/CSC411 - Machine Learning/Project4/CSC411-A4/')

def rand_seeding(k):
    rd.seed(k)
    random.seed(k)
    torch.manual_seed(k)
rand_seeding(0)

class Environment(object):
    """
    The Tic-Tac-Toe Environment
    """
    # possible ways to win
    win_set = frozenset([(0,1,2), (3,4,5), (6,7,8), # horizontal
                         (0,3,6), (1,4,7), (2,5,8), # vertical
                         (0,4,8), (2,4,6)])         # diagonal
    # statuses
    STATUS_VALID_MOVE = 'valid'
    STATUS_INVALID_MOVE = 'inv'
    STATUS_WIN = 'win'
    STATUS_TIE = 'tie'
    STATUS_LOSE = 'lose'
    STATUS_DONE = 'done'

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset the game to an empty board."""
        self.grid = np.array([0] * 9) # grid
        self.turn = 1                 # whose turn it is
        self.done = False             # whether game is done
        return self.grid

    def render(self):
        """Print what is on the board."""
        map = {0:'.', 1:'x', 2:'o'} # grid label vs how to plot
        print(''.join(map[i] for i in self.grid[0:3]))
        print(''.join(map[i] for i in self.grid[3:6]))
        print(''.join(map[i] for i in self.grid[6:9]))
        print('====')

    def check_win(self):
        """Check if someone has won the game."""
        for pos in self.win_set:
            s = set([self.grid[p] for p in pos])
            if len(s) == 1 and (0 not in s):
                return True
        return False

    def step(self, action):
        """Mark a point on position action."""
        assert type(action) == int and action >= 0 and action < 9
        # done = already finished the game
        if self.done:
            return self.grid, self.STATUS_DONE, self.done
        # action already have something on it
        if self.grid[action] != 0:
            return self.grid, self.STATUS_INVALID_MOVE, self.done
        # play move
        self.grid[action] = self.turn
        if self.turn == 1:
            self.turn = 2
        else:
            self.turn = 1
        # check win
        if self.check_win():
            self.done = True
            return self.grid, self.STATUS_WIN, self.done
        # check tie
        if all([p != 0 for p in self.grid]):
            self.done = True
            return self.grid, self.STATUS_TIE, self.done
        return self.grid, self.STATUS_VALID_MOVE, self.done

    def random_step(self):
        """Choose a random, unoccupied move on the board to play."""
        pos = [i for i in range(9) if self.grid[i] == 0]
        move = random.choice(pos)
        return self.step(move)

    def play_against_random(self, action):
        """Play a move, and then have a random agent play the next move."""
        state, status, done = self.step(action)
        if not done and self.turn == 2:
            state, s2, done = self.random_step()
            if done:
                if s2 == self.STATUS_WIN:
                    status = self.STATUS_LOSE
                elif s2 == self.STATUS_TIE:
                    status = self.STATUS_TIE
                else:
                    raise ValueError("???")
        return state, status, done

class Policy(nn.Module):
    """
    The Tic-Tac-Toe Policy
    """
    def __init__(self, input_size=27, hidden_size=64, output_size=9):
        #super(Policy, self).__init__() #Python 2
        super().__init__() #Python 3
        # TODO - done?
        self.Linear1 = nn.Linear(input_size, hidden_size)
        self.Linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # TODO - done?
        h = F.relu( self.Linear1(x) )
        out = F.softmax( self.Linear2(h) )
        return out

def select_action(policy, state):
    """Samples an action from the policy at the state."""
    state = torch.from_numpy(state).long().unsqueeze(0)
    state = torch.zeros(3,9).scatter_(0,state,1).view(1,27)
    pr = policy(Variable(state))
    m = torch.distributions.Categorical(pr)
    action = m.sample()
    log_prob = torch.sum(m.log_prob(action))
    return action.data[0], log_prob

def compute_returns(rewards, gamma=1.0):
    """
    Compute returns for each time step, given the rewards
      @param rewards: list of floats, where rewards[t] is the reward
                      obtained at time step t
      @param gamma: the discount factor
      @returns list of floats representing the episode's returns
          G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + ... 

    >>> compute_returns([0,0,0,1], 1.0)
    [1.0, 1.0, 1.0, 1.0]
    >>> compute_returns([0,0,0,1], 0.9)
    [0.7290000000000001, 0.81, 0.9, 1.0]
    >>> compute_returns([0,-0.5,5,0.5,-10], 0.9)
    [-2.5965000000000003, -2.8850000000000002, -2.6500000000000004, -8.5, -10.0]
    """
    # TODO - Done
    k = len(rewards)
    rewards = np.array(rewards)
    gammas = np.array( [gamma**(i) for i in range(k) ] )
    G = [ sum( rewards[i:]*gammas[:k-i] ) for i in range(k) ]
    return G

def finish_episode(saved_rewards, saved_logprobs, gamma=1.0):
    """Samples an action from the policy at the state."""
    policy_loss = []
    returns = compute_returns(saved_rewards, gamma)
    returns = torch.Tensor(returns)
    # subtract mean and std for faster training
    returns = (returns - returns.mean()) / (returns.std() +
                                            np.finfo(np.float32).eps)
    for log_prob, reward in zip(saved_logprobs, returns):
        policy_loss.append(-log_prob * reward)
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward(retain_graph=True)
    # note: retain_graph=True allows for multiple calls to .backward()
    # in a single step

def get_reward(status):
    """Returns a numeric given an environment status."""
    return {
            Environment.STATUS_VALID_MOVE  : 0, # TODO - done
            Environment.STATUS_INVALID_MOVE: -1, 
            Environment.STATUS_WIN         : 1, 
            Environment.STATUS_TIE         : 0,
            Environment.STATUS_LOSE        : 0
    }[status]

def train(policy, env, gamma=1.0, log_interval=1000):
    """Train policy gradient."""
    optimizer = optim.Adam(policy.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10000, gamma=0.9)
    
    #running counts - track metrics during training
    running_reward = 0
    moves, inv_moves = 0, 0 #number of moves and invalid moves
    games, wins = 0, 0 #number of games and wins 
    m = {}
    m['avg_return'] = []
    m['frac_inv_moves'] = []
    m['i_episode'] = []
    m['win_rate'] = []

    for i_episode in range(50000): #count(1):
        saved_rewards = []
        saved_logprobs = []
        state = env.reset()
        done = False
        while not done:
            action, logprob = select_action(policy, state)
            state, status, done = env.play_against_random(action)
            reward = get_reward(status)
            saved_logprobs.append(logprob)
            saved_rewards.append(reward)
            if status == Environment.STATUS_INVALID_MOVE:
                inv_moves += 1
            moves += 1
        if status == Environment.STATUS_WIN:
            wins += 1
        games += 1
        
        R = compute_returns(saved_rewards)[0]
        running_reward += R
        
        finish_episode(saved_rewards, saved_logprobs, gamma)
        
        if i_episode % log_interval == 0 and i_episode != 0:
            avg_return = running_reward / log_interval
            print('Episode {}\t Average return: {:.2f}\t %Inv Moves: {:.3f}'.format(i_episode, avg_return, inv_moves/moves) )
            m['avg_return'] += [avg_return]
            m['frac_inv_moves'] += [inv_moves/moves]
            m['win_rate'] += [wins/games]
            m['i_episode'] += [i_episode]
            running_reward = 0
            moves, inv_moves = 0, 0
            games, wins = 0, 0

        save = True #save file
        if i_episode % (log_interval) == 0 and save:
            torch.save(policy.state_dict(),
                       "resources/weight_checkpoints/policy-%d.pkl" % i_episode)

        if i_episode % 1 == 0: # batch_size
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
    return m

def first_move_distr(policy, env):
    """Display the distribution of first moves."""
    state = env.reset()
    state = torch.from_numpy(state).long().unsqueeze(0)
    state = torch.zeros(3,9).scatter_(0,state,1).view(1,27)
    pr = policy(Variable(state))
    return pr.data


def load_weights(policy, episode):
    """Load saved weights""" 
    weights = torch.load("resources/weight_checkpoints/policy-%d.pkl" % episode)
    policy.load_state_dict(weights)

def play_games(policy, env, num):
    results = []
    player2 = env.play_against_random
    
    for i in range(num):
        state = env.reset()
        done = False
        while not done:
            action, logprob = select_action(policy, state)
            state, status, done = player2(action)
            #reward = get_reward(status)
        
        if status == 'win':
            results += [1]
        elif status == 'lose':
            results += [2]
        elif status == 'tie':
            results += [0]
        else:
            print('Invalid end of game')
            return
    return results




if __name__ == '__main__':
    
    ## Part 1
    if False: 
        rand_seeding(0)
        env.render()
        env = Environment()
        env.step(4)
        env.render()
        env.step(0)
        env.render()
        env.step(2)
        env.render()
        env.step(3)
        env.render()
        env.step(6)
        env.render()

    
    ## Part 5
    rand_seeding(0)
    policy = Policy()
    env = Environment()
    
    res = play_games(policy, env, 100) 
    a = [x for x in res if x == 1 ]
    print( len(a)/len(res) ) #win_rate before training
    
    st = first_move_distr(policy, env) #starting (random) first move dist
    m = train(policy, env, gamma=0.99, log_interval=1000)
    en = first_move_distr(policy, env) #first move dist after training
    
    res = play_games(policy, env, 100)
    a = [x for x in res if x == 1 ]
    print( len(a)/len(res) ) #win_rate after training

    frac_inv_move = m['frac_inv_moves']
    avg_return = m['avg_return']
    win_rate = m['win_rate']
    episode = m['i_episode']

    filename = 'part5a'
    plt.scatter(episode, avg_return, label='Average Returns')
    plt.title('Learning Curve')
    plt.xlabel('episodes')
    plt.ylabel('average returns')
    plt.savefig('resources/'+filename)
    #plt.show()
    plt.close()
    
    filename = 'part5c'
    plt.scatter(episode, frac_inv_move, label='Invalid Moves')
    plt.title('Invalid Moves')
    plt.xlabel('episodes')
    plt.ylabel('fraction of invalid moves')
    plt.savefig('resources/'+filename)
    #plt.show()
    plt.close()


    ## Part 6
    episodes = [0] + episode
    w = [] 
    l = []
    t = []
    n = 1000 #number of games to test on
    for i_episode in episodes:
        p = Policy()
        e = Environment()
        load_weights(p, i_episode)
        res = play_games(p, e, n)
        
        a = [x for x in res if x == 1 ]
        w += [ len(a)/n ]
        b = [x for x in res if x == 2 ]
        l += [ len(b)/n ]
        c = [x for x in res if x == 0 ]
        t += [ len(c)/n ]
    
    filename = 'part6'
    plt.scatter(episodes, w, label='Win Rate')
    plt.scatter(episodes, l, label='Lose Rate')
    plt.scatter(episodes, t, label='Tie Rate')
    plt.title('Win-Lose-Tie Rates')
    plt.xlabel('episodes')
    plt.ylabel('rates')
    plt.legend()
    plt.savefig('resources/'+filename)
    #plt.show()
    plt.close()



    ## MISC
    '''
    import sys
    if len(sys.argv) == 1:
        # `python tictactoe.py` to train the agent
        train(policy, env)
    else:
        # `python tictactoe.py <ep>` to print the first move distribution
        # using weightt checkpoint at episode int(<ep>)
        ep = int(sys.argv[1])
        load_weights(policy, ep)
        print(first_move_distr(policy, env))
    '''

