import numpy as np


class QuadraticCost:

    def __init__(self, Q, R, S):
        self.Q = Q
        self.R = R
        self.S = S

    def run_cost(self, x, u, xd):
        e = x-xd
        l = 0.5*(u.T @ self.R @ u + e.T @ self.Q @ e)
        return l

    def run_cost_grad(self, x, u, xd):
        e = x-xd
        l_x = self.Q @ e
        l_xx = self.Q
        l_u = self.R @ u
        l_uu = self.R
        l_ux = np.zeros([np.shape(self.R)[0], np.shape(self.Q)[0]])
        l_xu = l_ux.T
        return l_x, l_u, l_xx, l_xu, l_ux, l_uu

    def term_cost(self, x, xf):
        e = x - xf
        phi = 0.5*(e.T @ self.S @ e)
        return phi

    def term_cost_grad(self, x, xf):
        e = x - xf
        phi_x = self.S @ e
        phi_xx = self.S
        return phi_x, phi_xx
