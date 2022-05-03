import torch
import torch.nn as nn

"""
input is [sin(phi), sin(theta), sin(psi), 
          cos(phi), cos(theta), cos(psi), 
          phidot, thetadot, psidot, 
          vx, vy, vz, 
          u1, u2, u3, u4]
"""
class ThetaLearner(nn.Module):
    def __init__(self, dt, optimizer=None, criterion=None):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(16, 64),
            # nn.Sigmoid(),
            # nn.Linear(64, 64),
            # nn.ReLU(),
            # nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 12)
        )
        self.dt = dt
        self.optimizer = optimizer
        self.criterion = criterion

    def forward(self, x, preprocessed):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x.T).float()
            if not preprocessed:
                x = self.preprocess(x)
            f = self.model(x)
            return f.detach().numpy()
        else:
            if not preprocessed:
                x = self.preprocess(x)
            return self.model(x)

    def getfxfu(self, x):
        J = torch.zeros((12, 16))
        x = torch.from_numpy(x.T).float()
        angles = x[:,0:3]
        x = self.preprocess(x)
        x.requires_grad = True
        preds = self.forward(x, True)
        for i in range(12):
            grd = torch.zeros((1,12))
            grd[0,i] = 1
            preds.backward(gradient=grd, retain_graph=True)
            J[i,:] = x.grad
            x.grad.zero_()

        # Get fx w.r.t. angles with chain rule and summing??? I think this makes sense
        dsin = J[:,0:3]                 # fx w.r.t. sin(angles)
        dcos = J[:,3:6]                 # fx w.r.t. cos(angles)
        sin = torch.sin(angles)
        cos = torch.cos(angles)
        dsincos = dsin*cos
        dcos_neg_sin = dcos*-sin
        dangles = dsincos+dcos_neg_sin

        # fx w.r.t. position is 0 logically
        fx = torch.cat((dangles, J[:,6:12], torch.zeros((12,3))),1)
        fx = fx*self.dt
        fu = J[:,12:] * self.dt
        return fx.detach().numpy(), fu.detach().numpy()

    def train(self, x, target):
        target = target.float()
        out = self.forward(self.preprocess(x.float()), True).float()
        loss = 0
        if x.shape[0] != 1:
            av = torch.mean(target,0).float()
            st = torch.std(target,0).float()
            st[torch.where(st==0)] = 1
            loss = self.criterion((out-av)/st, (target-av)/st)
        else:
            loss = self.criterion(out, target)
        if torch.isnan(loss):
            print('loss is nan!')
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # batch_acc = accuracy(out, target)
        return loss.detach().numpy()

    def preprocess(self, x):
        angles = x[:,0:3]
        sin = torch.sin(angles)
        cos = torch.cos(angles)
        return torch.cat((sin, cos, x[:,3:9], x[:,12:]),1)


