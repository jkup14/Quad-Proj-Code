import sys
import airsim
import os
import numpy as np
from dynamics import Quadrotor
from quadratic_cost import QuadraticCost
from min_ddp_penalty_reg_track import MinDDPReg
from plot_3dvehicle import animate3dVehicle_Multi_track
import matplotlib.pyplot as plt
from helper_functions import quaternion2Angle


"""Format Paths"""
airsim_install = '/Users/joshuakuperman/Desktop/CS7643/project/AirSim' # e.g. /Users/dnsge/dev/AirSim
sys.path.append(airsim_install + '/PythonClient')
sys.path.append(airsim_install + '/PythonClient/multirotor')

"""Problem Setup"""
"""Specify horizon, sampling time, and generate time vector"""
N_MPC = 1000
N = 100
dt = 0.02
T = dt*N
times = np.linspace(0, dt*N-dt, N)
times_MPC = np.linspace(0, dt*N_MPC-dt, N_MPC)
dynamics = Quadrotor(dt)
"""Start and Goal States"""
dynamics.x0 = np.array([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [-10]])
dynamics.xf =np.array([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [-10]])
""" Define Cost """
# State and Control Cost matrices
Q_pos = 20
Q_vel = 2
Q = 0.0*np.eye(dynamics.n)
Q[6,6] = Q_vel
Q[7,7] = Q_vel
Q[8,8] = Q_vel
Q[9,9] = Q_pos
Q[10,10] = Q_pos
Q[11,11] = Q_pos
R = 1* np.eye(dynamics.m)
S = 0*np.eye(dynamics.n)
cost = QuadraticCost(Q, R, S)
""" Initialize DDP Solver """
max_iters = 10                     # DDP Iterations
conv_threshold = 1e-3               # convergence threshold
ubar = np.zeros((dynamics.m,N-1))   # nominal input
xbar = np.zeros((dynamics.n,N))


"""Generate Tracking Trajectory that lasts for 20 seconds(figure 8)"""
T_loop = 20
N_loop = int(T_loop/dt)
t_pi =2*np.pi*np.linspace(0, T_loop-dt, int(T_loop/dt))/T_loop
x_des = 3*np.sin(t_pi)
y_des = 2.5*np.sin(2*t_pi)
traj_desired = np.zeros((dynamics.n,N_loop))
traj_desired[9,:] = x_des
traj_desired[10,:] = y_des
traj_desired[11,:] = -10*np.ones((N_loop))
traj_desired[6:9,:] = np.diff(traj_desired[9:], prepend=traj_desired[9:,[0]])

"""Control Bounds"""
ctrl_bounds = np.array([[0,0,0,0],[1,1,1,1]])

""" DDP Solver """
solver = MinDDPReg(traj_desired, xbar, ubar, dynamics, cost, max_iters, conv_threshold, N, verbose = False, ctrl_bounds=ctrl_bounds)
# X, U, K_u, J, c = solver.compute_optimal_solution()

""" Connect to the AirSim simulator """
# client = airsim.MultirotorClient()
# client.confirmConnection()
# client.enableApiControl(True)
# client.armDisarm(True)
#
# # Async methods returns Future. Call join() to wait for task to complete.
# client.takeoffAsync().join()
# client.moveToPositionAsync(0, 0, -10, 5).join()

"""Replan every timestep and follow the figure eight forever"""
x_prev = dynamics.x0
x0_curr = dynamics.x0
x_traj = np.zeros((dynamics.n,N_MPC))
u_traj = np.zeros((dynamics.m,N_MPC))
for i in range(N_MPC):
    """Find where on the desired path we are closest to"""
    # pose = client.simGetVehiclePose()
    # position = np.array([pose.position.x_val, pose.position.y_val, pose.position.z_val])
    # orientation = np.array(quaternion2Angle(pose.orientation))
    # position = np.array([1, 1, 10])
    # orientation = np.array([0.01, 0.01, 0.01])
    # x0_curr = np.concatenate([orientation, (orientation-x_prev[0:3].squeeze())/dt, (position-x_prev[9:].squeeze())/dt, position])

    position = x0_curr[9:]
    start_ind = np.argmin(np.sum((traj_desired[9:]-position)**2,0))
    # print(start_ind)
    x_desired = np.roll(traj_desired,-start_ind,1)
    solver.x_desired = x_desired[:,:N]
    xbar[:, [0]] = x0_curr
    for ii in range(N - 1):  # generate nominal state
        xbar[:, [ii + 1]] = dynamics.system_propagate(xbar[:, [ii]], ubar[:, [ii]], ii)
    solver.xbar = xbar
    solver.ubar = ubar
    solver.x0 = x0_curr
    solver.xf = solver.x_desired[:,[-1]]
    X, U, K_u, J, c = solver.compute_optimal_solution()
    x0_curr = dynamics.system_propagate(x0_curr, U[:,0],0)
    x_traj[:, [i]] = x0_curr
    u_traj[:, [i]] = U[:, [0]]
    ubar = np.concatenate([U[:,1:], U[:,[-1]]],1)
    print(i*dt, ' seconds, Cost is', J[-1], ' after', np.shape(J)[0], ' iterations.')

x_traj = np.concatenate([dynamics.x0, x_traj],1)












#
# """Follow Figure 8"""
# for i in range(N-1):
#     fl, fr, rr, rl = U[:,i]
#     client.moveByMotorPWMsAsync(fr, rl, fl, rr, dt).join()


fig1,ax1 = plt.subplots(1)
ax1.plot(times_MPC, u_traj.T)
fig2,ax2 = plt.subplots(1)
ax2.plot(times_MPC, x_traj[9:,:-1].T)
ax2.plot(times_MPC, traj_desired[9:, :N_MPC].T, color='r')
ax2.legend(['x','y','z'])
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


