import numpy as np
from theta_net import ThetaLearner

class Quadrotor_theta:
    def __init__(self, dynamics, model, dt):
        self.x0 = dynamics.x0
        self.xf = dynamics.xf
        self.n = 12
        self.m = 4
        self.dt = dt
        self.f_wrong = dynamics.system_dyn
        self.grad_wrong = dynamics.system_grad
        self.model = model

    def system_dyn(self, state, control):
        input = np.concatenate([state, control])
        return self.f_wrong(state, control)+self.model(input, False).T

    def system_propagate(self, state, control, k):
        f = self.system_dyn(state, control)
        state_next = state + self.dt * f
        return state_next

    def system_grad(self, state, control, k):
        input = np.concatenate([state, control])
        fx_theta, fu_theta = self.model.getfxfu(input)
        fx_wrong, fu_wrong = self.grad_wrong(state, control, k)
        return fx_wrong+fx_theta, fu_wrong+fu_theta
