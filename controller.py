import sys
import airsim
import os
import numpy as np
from dynamics import Quadrotor
from quadratic_cost import QuadraticCost
from min_ddp_penalty_reg_track import MinDDPReg
from plot_3dvehicle import animate3dVehicle_Multi_track
import matplotlib.pyplot as plt


"""Format Paths"""
airsim_install = '/Users/joshuakuperman/Desktop/CS7643/project/AirSim' # e.g. /Users/dnsge/dev/AirSim
sys.path.append(airsim_install + '/PythonClient')
sys.path.append(airsim_install + '/PythonClient/multirotor')

"""Problem Setup"""
"""Specify horizon, sampling time, and generate time vector"""
N = 750
dt = 0.02
T = dt*N
times = np.linspace(0, dt*N-dt, N)
dynamics = Quadrotor(dt)
"""Start and Goal States"""
dynamics.x0 = np.array([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [-10]])
dynamics.xf =np.array([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [-10]])
""" Define Cost """
# State and Control Cost matrices
Q_pos = 10
Q = 0.0*np.eye(dynamics.n)
Q[9,9] = Q_pos
Q[10,10] = Q_pos
Q[11,11] = Q_pos
R = 10 * np.eye(dynamics.m)
S = 0*np.eye(dynamics.n)
cost = QuadraticCost(Q, R, S)
""" Initialize DDP Solver """
max_iters = 50                     # DDP Iterations
conv_threshold = 1e-3               # convergence threshold
ubar = np.zeros((dynamics.m,N-1))   # nominal input
xbar = np.zeros((dynamics.n,N))
xbar[:, [0]] = dynamics.x0
for ii in range(N-1):               # generate nominal state
    xbar[:, [ii+1]]=dynamics.system_propagate(xbar[:, [ii]], ubar[:, [ii]], ii)

"""Generate Tracking Trajectory (figure 8)"""
t_pi =2*np.pi*times/times[-1]
x_des = 3*np.sin(t_pi)
y_des = 2.5*np.sin(2*t_pi)
traj_desired = np.zeros((dynamics.n,N))
traj_desired[9,:] = x_des
traj_desired[10,:] = y_des
traj_desired[11,:] = -10*np.ones((N))

"""Control Bounds"""
ctrl_bounds = np.array([[0,0,0,0],[1,1,1,1]])

""" DDP Solver """
solver = MinDDPReg(traj_desired, xbar, ubar, dynamics, cost, max_iters, conv_threshold, N, verbose = True, ctrl_bounds=ctrl_bounds)
X, U, K_u, J, c = solver.compute_optimal_solution()

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Async methods returns Future. Call join() to wait for task to complete.
client.takeoffAsync().join()
client.moveToPositionAsync(0, 0, -10, 5).join()

"""Follow Figure 8"""
for i in range(N-1):
    fl, fr, rr, rl = U[:,i]
    client.moveByMotorPWMsAsync(fr, rl, fl, rr, dt).join()


fig1,ax1 = plt.subplots(1)
ax1.plot(times[0:-1], U.T)
fig2,ax2 = plt.subplots(1)
ax2.plot(times, X[9:].T)
ax2.plot(times, traj_desired[9:].T, color='r')
ax2.legend(['x','y','z','x','y','z'])
# fig2 = animate3dVehicle_Multi_track(1, 12, 1, dynamics.x0, dynamics.xf, traj_desired, times, X, U, np.array([]))
plt.show()






# take images
# responses = client.simGetImages([
#     airsim.ImageRequest("0", airsim.ImageType.DepthVis),
#     airsim.ImageRequest("1", airsim.ImageType.DepthPlanar, True)])
# print('Retrieved images: %d', len(responses))
#
# # do something with the images
# for response in responses:
#     if response.pixels_as_float:
#         print("Type %d, size %d" % (response.image_type, len(response.image_data_float)))
#         airsim.write_pfm(os.path.normpath('py1.pfm'), airsim.get_pfm_array(response))
#     else:
#         print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
#         airsim.write_file(os.path.normpath('py1.png'), response.image_data_uint8)
#         client.armDisarm(True)
