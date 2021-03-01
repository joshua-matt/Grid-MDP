import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

class GridMDP:
    def __init__(self, w, h, c, r, p):
        self.s = (h-1,w-1)
        self.A = '←↑→↓.'
        self.w = w
        self.h = h
        self.R = c*np.ones((h,w))

        self.r = r
        self.p = p
        self.S_term = r + p
        for s in r:
            self.R[s] = 1
        for s in p:
            self.R[s] = -1

    def action(self,a):
        self.s = self.T(a,self.s)

    def T(self,a,s): # Transition function
        if a == '←':
            return (s[0],
                    0 if s[1] == 0 else s[1]-1)
        elif a == '→':
            return (s[0],
                    self.w-1 if s[1]==self.w-1 else s[1]+1)
        elif a == '↑':
            return (0 if s[0] == 0 else s[0]-1,
                    s[1])
        elif a == '↓':
            return (self.h-1 if s[0]==self.h-1 else s[0]+1,
                    s[1])
        else:
            return s

class Agent:
    def __init__(self, mdp, gamma):
        self.mdp = mdp
        self.gamma = gamma
        self.V = np.zeros((mdp.h,mdp.w))
        self.P = np.reshape(np.array(['.' for i in range(mdp.h*mdp.w)]), (mdp.h, mdp.w))

    def q(self,s,a):
        return self.mdp.R[s] + self.gamma * self.V[self.mdp.T(a,s)]

    def value_iteration(self, iters):
        V_ = np.zeros((self.mdp.h,self.mdp.w))

        for i in range(iters):
            self.V = V_

            for j in range(self.mdp.h):
                for k in range(self.mdp.w):
                    s = (j,k)
                    if s in self.mdp.S_term:
                        V_[s] = self.mdp.R[s]
                        continue
                    V_[s] = max([self.q(s,a) for a in self.mdp.A])

    def extract_policy(self):
        for j in range(self.mdp.h):
            for k in range(self.mdp.w):
                s = (j, k)
                for a in self.mdp.A:
                    if self.q(s, a) > self.q(s, self.P[s]):
                        self.P[s] = a

    def policy_evaluation(self, iters=100):
        V_ = np.zeros((self.mdp.h, self.mdp.w))

        largest_diff = -1000000000
        first = True

        """while largest_diff > ep or first:
            first = False"""
        for i in range(iters):
            self.V = V_

            for j in range(self.mdp.h):
                for k in range(self.mdp.w):
                    s = (j, k)
                    if s in self.mdp.S_term:
                        V_[s] = self.mdp.R[s]
                        continue

                    V_[s] = self.mdp.R[s] + self.gamma * self.V[self.mdp.T(self.P[s], s)]
                    """if abs(V_[s] - self.V[s]) > largest_diff and s not in self.mdp.S_term:
                        largest_diff = abs(V_[s] - self.V[s])
                        print(V_[s],s)
        print(largest_diff)"""

    def policy_iteration(self):
        unchanged = False
        while not unchanged:
            unchanged = True
            self.policy_evaluation()

            for j in range(self.mdp.h):
                for k in range(self.mdp.w):
                    s = (j, k)
                    for a in self.mdp.A:
                        if self.q(s, a) > self.q(s, self.P[s]):
                            self.P[s] = a
                            unchanged = False

    def FVMC_evaluation(self, episodes):
        N = 0.000000000001*np.ones((mdp.h,mdp.w)) # Avoid zero division
        S = np.zeros((mdp.h,mdp.w))

        def discounted_sum(rewards, k):
            if k == len(rewards):
                return 0
            return rewards[k] + self.gamma * discounted_sum(rewards, k + 1)

        for i in range(episodes):
            # Episode
            rewards = []
            states = []
            while True:
                if self.mdp.s not in states: # Increment first-visit counter
                    N[self.mdp.s] += 1

                rewards.append(self.mdp.R[self.mdp.s]) # Add to reward sequence
                states.append(self.mdp.s) # Add to state sequence

                if self.mdp.s in self.mdp.S_term:
                    break

                self.mdp.action(self.P[self.mdp.s]) # Agent takes action according to policy

            for state in list(set(states)): # Find first visit
                S[state] += discounted_sum(rewards, states.index(state))

            self.mdp.s = (np.random.randint(self.mdp.h),np.random.randint(self.mdp.w)) # Reinitialize in random state

        return np.divide(S,N)

    def TD_evaluation(self, updates, alpha):
        V_ = 0.5*np.ones((self.mdp.h,self.mdp.w))

        for i in range(updates):
            for s in self.mdp.S_term: # Maintain terminal values
                V_[s] = self.mdp.R[s]
            state = self.mdp.s # Store previous state
            Vt = V_[state] # Store previous value estimate
            self.mdp.action(self.P[state]) # Take action according to policy
            new_state = self.mdp.s
            V_[state] += alpha * (self.mdp.R[state] + self.gamma*V_[new_state] - Vt) # TD update

            if new_state in self.mdp.S_term:
                self.mdp.s = (np.random.randint(self.mdp.h),np.random.randint(self.mdp.w))
        return V_


w = 5
h = 5

default_r = -0.1

gamma = 0.8

plot_policy = True
use_value_iter = True
compare_mc = False
compare_td = True

maze = [(i,1) for i in range(h-1)] + [(i,3) for i in range(1,h)] + [(i,5) for i in range(1,h-1)] + [(1,4)] # 7x7 minimum

mdp = GridMDP(w,h,default_r, [(0,0)], [(2,2)])
a = Agent(mdp,gamma)

if use_value_iter:
    a.value_iteration(1000)
    a.extract_policy()
    print("True value function:\n", a.V)
    print()
    if compare_mc:
        print("Monte Carlo estimate:\n", a.FVMC_evaluation(100))
        print()
    if compare_td:
        print("TD estimate:\n", a.TD_evaluation(5000, 0.1))
        print()
else:
    a.policy_iteration()
rg_map = LinearSegmentedColormap.from_list("rg", [(1,0,0), (1,1,1), (0,1,0)])
scale = max(abs(np.max(a.V)), abs(np.min(a.V)))

plt.pcolormesh(np.flipud(a.V), cmap=rg_map, vmax=scale, vmin=-scale)
plt.xticks(np.arange(0,mdp.w+1,1))
plt.yticks(np.arange(0,mdp.h+1,1))
plt.grid()

for y in range(mdp.h):
    for x in range(mdp.w):
        if (mdp.h-y-1,x) in mdp.r:
            plt.text(x+0.1, y+0.7, '★', fontsize=16)
        elif (mdp.h-y-1,x) in mdp.p:
            plt.text(x+0.1, y+0.7, '☠', fontsize=16)
        plt.text(x+0.5, y+0.5, np.flipud(a.P)[y,x] if plot_policy else '%.2f' % (np.flipud(a.V)[y,x]),
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontsize=24)

plt.title("%dx%d Grid MDP: γ=%.2f" % (mdp.w,mdp.h,a.gamma))

plt.colorbar()
plt.show()