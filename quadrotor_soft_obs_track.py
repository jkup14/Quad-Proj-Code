from systems_dynamics.quadrotor import Quadrotor
from systems_dynamics.multi_agent import Multi_Agent
from costs.quadratic_cost import QuadraticCost
from ddp_algorithms.min_ddp_penalty_reg_track import MinDDPReg
from systems_constraints import obstacles_3d
from systems_constraints import distance_3d
from systems_constraints import obstacles_soft_sig_3d
from bas_functions import bas_dynamics
from bas_functions import embed_dynamics
from plottings.plot_3dvehicle import plot3dvehicle
from plottings.plot_3dvehicle import animate3dVehicle_Multi_track
import numpy as np
import matplotlib.pyplot as plt
import time
import random as r


"""Specify horizon, sampling time, and generate time vector"""
N = 300
dt = 0.02
T = dt*N
times = np.linspace(0, dt*N-dt, N)

"""choose system's dynamics"""
dynamics = Quadrotor(dt)
# dynamics.x0 = np.array([[0], [0], [0], [0], [0], [0], [0], [0], [0], [4], [4], [0],
#                         [0], [0], [0], [0], [0], [0], [0], [0], [0], [-4.1], [-4.5], [0]])
# dynamics.xf = np.array([[0], [0], [0], [0], [0], [0], [0], [0], [0], [-4], [-4], [0],
#                         [0], [0], [0], [0], [0], [0], [0], [0], [0], [4.5], [3.8], [0]])


dynamics.x0 = np.array([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [-0.1], [0.1]])

dynamics.xf =np.array([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]])

""" Define safety constraints (function h), generate BaS and embed into dynamics"""
constraints_class = list()
# constraints_class = obstacles_2d.Obstacles3D(dynamics.n, 'manual_obstacles1')
# safety_function = [constraints_class.safety_function]
# dbas_dynamics = [bas_dynamics.BaSDynamics(dynamics, safety_function[0])]
# embedded_dynamics = embed_dynamics.EmbedDynamics(dynamics, dbas_dynamics, [1])
#
#
p = 10
m = 5
c1 = 5
c2 = 5
constraints_class1 = obstacles_3d.Obstacles3D(dynamics.n, 'manual_obstacles1', 1, 'centralized')
constraints_class = [constraints_class1]
safety_function = list()
for i in range(len(constraints_class)):
    safety_function.append(constraints_class[i].safety_function)
# Define BaS dynamics given system's dynamics and safety function
n_bas1 = 1
n_bas_vector = [n_bas1]
n_bas = sum(n_bas_vector)
bas = list()
for i in range(len(constraints_class)):
    bas.append(bas_dynamics.BaSDynamics(dynamics, safety_function[i], N, False, 'tolerant_barrier', n_bas_vector[i], tol_params=[p, m, c1, c2]))

embedded_dynamics = embed_dynamics.EmbedDynamics(dynamics, bas, n_bas_vector)
# overwrite dynamics
dynamics = embedded_dynamics
n_bas = dynamics.n_bas


""" Define Cost """
# State and Control Cost matrices
Q_pos = 200
Q = 0.0*np.eye(dynamics.n)
Q[9,9] = Q_pos
Q[10,10] = Q_pos
Q[11,11] = Q_pos
Q_dbas = 15
Q[dynamics.n - 1, dynamics.n - 1] = Q_dbas

R = 0.5*1e-2 * np.eye(dynamics.m)

# state terminal cost matrix
S = 0*np.eye(dynamics.n)


# Define Cost
cost = QuadraticCost(Q, R, S)


""" Initialize DDP Solver """
# DDP Iterations and convergence threshold
max_iters = 500
conv_threshold = 1e-3
# nominal input:
ubar = np.zeros((dynamics.m,N-1))
# ubar[0, :] = np.ones((1,N-1))
xbar = np.zeros((dynamics.n,N))
xbar[:, [0]] = dynamics.x0
# generate nominal state:
for ii in range(N-1):
    xbar[:, [ii+1]]=dynamics.system_propagate(xbar[:, [ii]], ubar[:, [ii]], ii)

#generate tracking trajectory:
t_pi =2*np.pi*0.96*times/times[-1]
x_des = 6*np.sin(t_pi)
y_des = 5*np.sin(2*t_pi)
traj_desired = np.zeros((dynamics.n,N))
traj_desired[9,:] = x_des
traj_desired[10,:] = y_des
""" DDP Solver """
solver = MinDDPReg(traj_desired, xbar, ubar, dynamics, cost, max_iters, conv_threshold, N, safety_function, verbose = True, tolerant_ind = [0])
# Compute Trajectory


start = time.time()
X, U, K_u, J, c = solver.compute_optimal_solution()
end = time.time()
# X[-1,:] = X[-1,:]*0
# Q_pos = 0.1
# Q = 0.01*np.eye(dynamics.n)
# Q[9,9] = Q_pos
# Q[10,10] = Q_pos
# Q[11,11] = Q_pos
# Q_dbas = 0.1
# Q[dynamics.n - 1, dynamics.n - 1] = Q_dbas
# cost = QuadraticCost(Q, R, S)
# start = time.time()
# solver = MinDDPReg(X, xbar, ubar, dynamics, cost, max_iters, conv_threshold, N, safety_function, verbose = True, tolerant_ind = [0])
# X, U, K_u, J, c = solver.compute_optimal_solution()
# end = time.time()


# np.save('../results/' "Centralized_" + dyn + "_" + str(n_agents) + 'agent_delta_' + str(delta) + '_qdbas_' + str(Q_dbas) + "_sd_" + str(sd), X)
print('elapsed time=', end - start)
""" Plot and print data """
# plot3dvehicle(dynamics.x0,dynamics.xf, T, X, U, 12,n_agents)
# plt.show()
# plot_2dvehicle.animate2dVehicle_Multi(n_agents, dynamics1.n, delta, dynamics.x0, dynamics.xf, times, X)
obstacle_info = constraints_class1.obstacles()
fig2 = animate3dVehicle_Multi_track(1, 12, 1, dynamics.x0, dynamics.xf, traj_desired, times, X, U, obstacle_info)
fig, ax = plt.subplots(1)
ax.plot(times, X[-1, :])
plt.rcParams["font.family"] = "Times New Roman"
plt.xlabel('Time (s)', fontsize=12, math_fontfamily='cm', fontname="Times New Roman")
plt.ylabel(r'Tolerant Discrete Barrier State $ \overline{\beta} $', fontsize=12, math_fontfamily='cm', fontname="Times New Roman")
# plt.title("DBaS vs Time", fontsize='large', fontname="Times New Roman")
plt.rcParams.update({
    "text.usetex": True})
plt.show()
print('xf_des=', dynamics.xf)
print('xf_true=', X[:, [-1]])
print('cost=', J)
print('iters needed=', J.shape[0])