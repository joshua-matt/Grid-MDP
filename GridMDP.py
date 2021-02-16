import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

class GridMDP:
    def __init__(self, w, h, c, r, p):
        self.s = (0, 0)
        self.A = 'lrud.'
        self.w = w
        self.h = h
        self.R = c*np.ones((h,w))
        self.S_term = r + p
        for s in r:
            self.R[s] = 1
        for s in p:
            self.R[s] = -1

    def T(self,a,s): # Transition function
        if a == 'l':
            return (0 if s[0] == 0 else s[0]-1,s[1])
        elif a == 'r':
            return (self.w-1 if s[0]==self.w-1 else s[0]+1,s[1])
        elif a == 'u':
            return (s[0], 0 if s[1] == 0 else s[1]-1)
        elif a == 'd':
            return (s[0], self.h-1 if s[1]==self.h-1 else s[1]+1)
        else:
            return s

    def Re(self):
        return self.R[self.s]

class Agent:
    def __init__(self, mdp, gamma):
        self.mdp = mdp
        self.gamma = gamma
        self.V = np.zeros((mdp.h,mdp.w))
        self.P = np.array(['.' for i in range(mdp.h*mdp.w)])

    def value_iteration(self, iters):
        Vp = np.zeros((self.mdp.h,self.mdp.w))

        for i in range(iters):
            self.V = Vp

            for j in range(self.mdp.h):
                for k in range(self.mdp.w):
                    s = (j,k)
                    if s in self.mdp.S_term:
                        Vp[s] = self.mdp.R[s]
                        continue
                    #print([self.mdp.R[self.mdp.T(a)] + self.gamma * self.V[self.mdp.T(a)] for a in self.mdp.A])
                    Vp[s] = max([self.mdp.R[s] + self.gamma*self.V[self.mdp.T(a,s)] for a in self.mdp.A])

    def policy_evaluation(self, iters):
        return

    def policy_improvement(self):
        return

    def policy_iteration(self, iters):
        return


mdp = GridMDP(5,5,-0.1, [(0,0)], [(i,1) for i in range(1,5)])
a = Agent(mdp,0.1)
a.value_iteration(100)
#print(a.V)

rg_map = LinearSegmentedColormap.from_list("rg", [(1,0,0), (1,1,1), (0,1,0)])

plt.pcolormesh(np.flipud(a.V), cmap=rg_map)
plt.colorbar()
plt.show()