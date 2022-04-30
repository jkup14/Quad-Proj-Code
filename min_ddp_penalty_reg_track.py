import numpy as np
from scipy import integrate as integ



class MinDDPReg:
    """
    Dynamic Differential Programming for given dynamics and cost function
    """
    def __init__(self, x_desired, xbar, ubar, dynamics, cost, max_iters, conv_threshold, N, lambda_min=0.1, lambda_max=1e10, dlambda=1.6, reg_ind=0, ctrl_bounds=None, verbose = False):
        """
        :type x0: np.ndarray
        :type xf: np.ndarray
        :type xbar: np.ndarray
        :type ubar: np.ndarray
        :type num_iter: int
        :type N: int
        :type dynamics: function
        :type cost: function
        """
        self.x0 = dynamics.x0
        self.xf = dynamics.xf
        self.x_desired = x_desired
        self.ubar = ubar
        self.xbar = xbar
        self.max_iters = max_iters
        self.conv_threshold = conv_threshold
        self.N = N
        self.dynamics = dynamics
        self.cost = cost
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.dlambda = dlambda
        self.reg_ind = reg_ind
        if ctrl_bounds is not None:
            self.ctrl_bounds = ctrl_bounds  # Format (u_lower, u_upper) tuple with mx1 vectors
        else:
            self.ctrl_bounds = None
        self.verbose = verbose

    def compute_optimal_solution(self):
        # Initialize input and state trajectory
        x0 = self.x0
        xf = self.xf
        ubar = self.ubar
        xbar = self.xbar
        x_desired = self.x_desired
        max_iters = self.max_iters
        conv_threshold = self.conv_threshold
        N = self.N
        dynamics = self.dynamics
        cost = self.cost
        safety_function = self.safety_function

        n = dynamics.n
        m = dynamics.m
        J = np.zeros(max_iters)           # cost for each iteration

        L = self.costofTraj(xbar, ubar, x_desired)
        dV = L
        ii = 0
        reg = 0
        deltaV = np.zeros([1, 2])
        while ii < max_iters and dV > conv_threshold:
            J[ii] = L
            k_u = np.zeros([m, N])
            K_u = np.zeros([m, n, N])
            Vx, Vxx = cost.term_cost_grad(xbar[:, [-1]], xf)
            # Backward propagation:
            for k in range(N - 2, -1, -1):
                Qx, Qu, Qxx, Qxu, Qux, Quu = self.ddp_matrices(xbar[:, [k]], ubar[:, [k]], x_desired[:,[k]], Vx, Vxx, k)
                # for Quu regularizarion:
                # Quu = Quu + lambda *np.eye(m)
                # compute ff and fb control gains
                Quu = Quu + np.eye(m) * self.reg_lambda()
                invQuu = np.linalg.inv(Quu)
                k_u[:, [k]] = -invQuu @ Qu
                K_u[:, :, k] = -invQuu @ Qux
                # k_u[:, [k]] = -np.linalg.solve(Quu, Qu)
                # K_u[:, :, k] = -np.linalg.solve(Quu, Qux)
            # compute value function grad and hess
            #     Vx = Qx - Qxu @ np.linalg.inv(Quu) @ Qu
            #     Vxx = Qxx - Qxu @ np.linalg.inv(Quu) @ Qux
                Vx = Qx + Qxu @ k_u[:, [k]]
                Vxx = Qxx + Qxu @ K_u[:, :, k]
            #     Vx = Qx + K_u[:, :, k].T @ Quu @ k_u[:, [k]] + K_u[:, :, k].T @ Qu + Qxu @ k_u[:, [k]]
            #     Vxx = Qxx + K_u[:, :, k].T @ Quu @ K_u[:, :, k] + K_u[:, :, k].T @ Qux + Qxu @ K_u[:, :, k]
                # make sure Vxx is positive definite (avoid numerical issues)
                Vxx = 0.5 * (Vxx + Vxx.T)
                # for expected decrease in cost:
                deltaV += [np.asscalar(k_u[:, k].T @ Qu), 0.5 * k_u[:, k].T @ Quu @ k_u[:, k]]
                # V[k] = Q_o - 0.5 * Qu.T @ np.linalg.inv(Quu) @ Qu
                # deltaV[:, k] = [k_u[:, [k]].T @ Qu, 0.5 * k_u[:, [k]].T @ Quu @ k_u[:, [k]]]

            forwardpass = 0
            x = np.zeros([n, N])
            u = np.zeros([m, N-1])
            x[:, [0]] = x0
            # alpha0 = np.linspace(1, 0, 11)
            alpha0 = np.power(10, np.linspace(0, -3, 11))
            # alpha0 = np.array([1])
            safe = True
            for jj in range(alpha0.shape[0]):
                alpha = alpha0[jj]
                # Forward propagation:
                L_new = 0.0
                safe = True
                deltax = x[:, [0]] - xbar[:, [0]]
                for k in range(N-1):
                    deltau = alpha*k_u[:, [k]] + K_u[:, :, k] @ deltax
                    u[:, [k]] = ubar[:, [k]] + deltau
                    if self.ctrl_bounds is not None:
                        u[:, k] = np.clip(u[:,k], self.ctrl_bounds[0], self.ctrl_bounds[1])
                    x[:, [k+1]] = dynamics.system_propagate(x[:, [k]], u[:, [k]], k)
                    deltax = x[:, [k+1]] - xbar[:, [k+1]]

                    L_new += self.cost.run_cost(x[:, [k]], u[:, [k]], x_desired[:,[k]])

                L_new += self.cost.term_cost(x[:, [-1]], self.xf)

                true_reduction = L - L_new

                expected_reduction = - alpha*(deltaV[0,0] + alpha*deltaV[0,1])
                z = true_reduction / expected_reduction
                if expected_reduction < 0:
                    z = np.sign(true_reduction)
                    print('negative expected reduction in cost!!')
                if z >= 0:
                    forwardpass = 1
                    break
            if forwardpass == 1:
                xbar = x
                ubar = u
                if self.reg_ind != 0:
                    print("Regularization Worked!")
                    self.reg_ind -= 2
                dV = L - L_new

                L = L_new
            else:
                # if safe: #useful if the only bas active is tolerant
                #     print('Line search failed but no collision detected, stopping loop')
                #     break
                print("Line search failed, trying lambda=",self.reg_lambda())

                self.reg_ind += 1

            if self.verbose:
                print("Iteration", ii, "Cost:", L)
            ii += 1

        if self.verbose:
            print('done')
        converged = dV <= conv_threshold
        return x, u, K_u, np.resize(J,ii), converged


    def costofTraj(self, x, u, x_desired):
        traj_cost = 0.0
        for i in range(self.N-1):
            traj_cost += self.cost.run_cost(x[:,[i]], u[:,[i]], x_desired[:,[i]])
        xN = x[:, [-1]]
        traj_term_cost = self.cost.term_cost(xN, self.xf)
        traj_cost += traj_term_cost
        return traj_cost


    def ddp_matrices(self, x, u, x_desired, Vx, Vxx, k):
        fx, fu = self.dynamics.system_grad(x, u, k)
        Lx, Lu, Lxx, Lxu, Lux, Luu = self.cost.run_cost_grad(x, u, x_desired)
        Qx = Lx + fx.T @ Vx
        Qu = Lu + fu.T @ Vx
        Qxx= Lxx + fx.T @ Vxx @ fx
        Qxu = Lxu + fx.T @ Vxx @ fu
        Qux = Qxu.T
        Quu = Luu + fu.T @ Vxx @ fu
        return Qx, Qu, Qxx, Qxu, Qux, Quu

    def reg_lambda(self):
        if self.reg_ind <= 0:
            self.reg_ind = 0
            return 0
        else:
            lam = self.lambda_min*self.dlambda**(self.reg_ind-1)
            if lam > self.lambda_max:
                print("Max Lambda Reached!")
                return None
            return lam

