import numpy as np


class Quadrotor:
    def __init__(self, dt):
        self.n = 12
        self.m = 4
        self.dt = dt
        self.g = 9.81               # gravitational force (m/s^2)
        self.mass = 1               # quad mass (kg)
        self.Ix = 1                 # quad x-axis inertia
        self.Iy = 1                 # quad y-axis inertia
        self.Iz = 1                 # quad z-axis inertia
        self.tau_wx = 0             # torque due to wind on the x-axis
        self.tau_wy = 0             # torque due to wind on the y-axis
        self.tau_wz = 0             # torque due to wind on the z-axis
        self.f_wx = 0               # force due to wind on the x-axis
        self.f_wy = 0               # force due to wind on the y-axis
        self.f_wz = 0               # force due to wind on the z-axis
        self.b = 10                 # thrust factor
        self.l = 0.5               # distance between rotor and center of drone
        self.d = 20                # drag factor
        self.x0 = np.array([[0], [0], [0], [0], [0], [0], [0], [0], [0], [5], [5], [5]])
        self.xf = np.array([[0], [0], [0], [0], [0], [0], [0], [0], [0], [-4], [-2], [1]])

    def system_dyn(self, state, control):
        # print(control)
        phi, theta, psi = state.item(0), state.item(1), state.item(2)   # roll, pitch and yaw angles in earth frame
        phi_rate, theta_rate, psi_rate = state.item(3), state.item(4), state.item(5)  # roll, pitch, yaw velocities in body frame
        vx, vy, vz = state.item(6), state.item(7), state.item(8)        # x, y, and z linear velocities in body frame
        x, y, z = state.item(9), state.item(10), state.item(11)         # x, y, and z in earth frame
        om1, om2, om3, om4 = control.item(0), control.item(1), control.item(2), control.item(3)
        ft = self.b*(om1+om2+om3+om4)
        tau_x = self.b*self.l*(om3-om1)
        tau_y = self.b*self.l*(om4-om2)
        tau_z = self.d*(om2+om4-om1-om3)

        phi_dot = phi_rate + psi_rate*np.cos(phi)*np.tan(theta) + theta_rate*np.sin(phi)*np.tan(theta)
        theta_dot = theta_rate*np.cos(phi) - psi_rate*np.sin(phi)
        psi_dot = (psi_rate*np.cos(phi))/np.cos(theta) + (theta_rate*np.sin(phi))/np.cos(theta)
        phi_rate_dot = (self.tau_wx + tau_x)/self.Ix + (theta_rate*psi_rate*(self.Iy - self.Iz))/self.Ix
        theta_rate_dot = (self.tau_wy + tau_y)/self.Iy + (phi_rate*psi_rate*(self.Ix - self.Iz))/self.Iy
        psi_rate_dot = (self.tau_wz + tau_z)/self.Iz + (phi_rate*theta_rate*(self.Ix - self.Iy))/self.Iz
        vx_dot = psi_rate*vy - theta_rate*vz + self.f_wx/self.mass - self.g*np.sin(theta)
        vy_dot = phi_rate*vz - psi_rate*vx + self.f_wy/self.mass + self.g*np.cos(theta)*np.sin(phi)
        vz_dot = theta_rate*vx - phi_rate*vy - (ft - self.f_wz)/self.mass + self.g*np.cos(phi)*np.cos(theta)
        x_dot = vz*(np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.sin(theta)) - vy*(np.cos(phi)*np.sin(psi) - np.cos(psi)*np.sin(phi)*np.sin(theta)) + vx*np.cos(theta)*np.cos(psi)
        y_dot = vy*(np.cos(phi)*np.cos(psi) + np.sin(phi)*np.sin(theta)*np.sin(psi)) - vz*(np.cos(psi)*np.sin(phi) - np.cos(phi)*np.sin(theta)*np.sin(psi)) + vx*np.cos(theta)*np.sin(psi)
        z_dot = vz*np.cos(phi)*np.cos(theta) - vx*np.sin(theta) + vy*np.cos(theta)*np.sin(phi)

        f = np.array([[phi_dot], [theta_dot], [psi_dot],[phi_rate_dot],[theta_rate_dot],[psi_rate_dot], [vx_dot], [vy_dot], [vz_dot], [x_dot], [y_dot], [z_dot]])
        return f

    def system_propagate(self, state, control, k):
        f = self.system_dyn(state, control)
        state_next = state + self.dt * f
        return state_next

    def system_grad(self, state, control, k):
        # print(control)
        phi, theta, psi = state.item(0), state.item(1), state.item(2)   # roll, pitch and yaw angles in earth frame
        phi_rate, theta_rate, psi_rate = state.item(3), state.item(4), state.item(5)  # roll, pitch, yaw velocities in body frame
        vx, vy, vz = state.item(6), state.item(7), state.item(8)        # x, y, and z linear velocities in body frame
        x, y, z = state.item(9), state.item(10), state.item(11)         # x, y, and z in earth frame
        om1, om2, om3, om4 = control

        fx = np.eye(self.n) + self.dt * np.array([[theta_rate*np.cos(phi)*np.tan(theta) - psi_rate*np.sin(phi)*np.tan(theta), psi_rate*np.cos(phi)*(np.tan(theta)**2 + 1) + theta_rate*np.sin(phi)*(np.tan(theta)**2 + 1), 0, 1, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta), 0, 0, 0, 0, 0, 0],
                                                  [-psi_rate*np.cos(phi)-theta_rate*np.sin(phi), 0, 0, 0, np.cos(phi), -np.sin(phi), 0, 0, 0, 0, 0, 0],
                                                  [(theta_rate*np.cos(phi))/np.cos(theta)-(psi_rate*np.sin(phi))/np.cos(theta), (psi_rate*np.cos(phi)*np.sin(theta))/np.cos(theta)**2+(theta_rate*np.sin(phi)*np.sin(theta))/np.cos(theta)**2, 0, 0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta), 0, 0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, (psi_rate*(self.Iy-self.Iz))/self.Ix, (theta_rate*(self.Iy-self.Iz))/self.Ix, 0, 0, 0, 0, 0, 0],
                                                  [0, 0, 0, -(psi_rate*(self.Ix-self.Iz))/self.Iy, 0, -(phi_rate*(self.Ix-self.Iz))/self.Iy, 0, 0, 0, 0, 0, 0],
                                                  [0, 0, 0, (theta_rate*(self.Ix-self.Iy))/self.Iz, (phi_rate*(self.Ix-self.Iy))/self.Iz, 0, 0, 0, 0, 0, 0, 0],
                                                  [0, -self.g*np.cos(theta), 0, 0, -vz, vy, 0, psi_rate, -theta_rate, 0, 0, 0],
                                                  [self.g*np.cos(phi)*np.cos(theta), -self.g*np.sin(phi)*np.sin(theta), 0, vz, 0, -vx, -psi_rate, 0, phi_rate, 0, 0, 0],
                                                  [-self.g*np.cos(theta)*np.sin(phi), -self.g*np.cos(phi)*np.sin(theta), 0, -vy, vx, 0, theta_rate, -phi_rate, 0, 0, 0, 0],
                                                  [vy*(np.sin(phi)*np.sin(psi)+np.cos(phi)*np.cos(psi)*np.sin(theta))+vz*(np.cos(phi)*np.sin(psi)-np.cos(psi)*np.sin(phi)*np.sin(theta)), vz*np.cos(phi)*np.cos(theta)*np.cos(psi)-vx*np.cos(psi)*np.sin(theta)+vy*np.cos(theta)*np.cos(psi)*np.sin(phi), vz*(np.cos(psi)*np.sin(phi)-np.cos(phi)*np.sin(theta)*np.sin(psi))-vy*(np.cos(phi)*np.cos(psi) + np.sin(phi)*np.sin(theta)*np.sin(psi)) - vx*np.cos(theta)*np.sin(psi), 0,  0, 0, np.cos(theta)*np.cos(psi), np.cos(psi)*np.sin(phi)*np.sin(theta) - np.cos(phi)*np.sin(psi), np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.sin(theta), 0, 0, 0],
                                                  [-vy*(np.cos(psi)*np.sin(phi) - np.cos(phi)*np.sin(theta)*np.sin(psi)) - vz*(np.cos(phi)*np.cos(psi) + np.sin(phi)*np.sin(theta)*np.sin(psi)), vz*np.cos(phi)*np.cos(theta)*np.sin(psi) - vx*np.sin(theta)*np.sin(psi) + vy*np.cos(theta)*np.sin(phi)*np.sin(psi), vz*(np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.sin(theta)) - vy*(np.cos(phi)*np.sin(psi) - np.cos(psi)*np.sin(phi)*np.sin(theta)) + vx*np.cos(theta)*np.cos(psi), 0,  0, 0, np.cos(theta)*np.sin(psi), np.cos(phi)*np.cos(psi) + np.sin(phi)*np.sin(theta)*np.sin(psi), np.cos(phi)*np.sin(theta)*np.sin(psi) - np.cos(psi)*np.sin(phi), 0, 0, 0],
                                                  [vy*np.cos(phi)*np.cos(theta) - vz*np.cos(theta)*np.sin(phi),   - vx*np.cos(theta) - vz*np.cos(phi)*np.sin(theta) - vy*np.sin(phi)*np.sin(theta), 0, 0,  0, 0, -np.sin(theta), np.cos(theta)*np.sin(phi), np.cos(phi)*np.cos(theta), 0, 0, 0]])

        fu = self.dt * np.array([[0.,    0.,    0.,    0.],
                                 [0.,    0.,    0.,    0.],
                                 [0.,    0.,    0.,    0.],
                                 [-self.b*self.l/self.Ix, 0., self.b*self.l/self.Ix,  0.],
                                 [0., -self.b*self.l/self.Iy,  0.,  self.b*self.l/self.Iy],
                                 [-self.d/self.Iz, self.d/self.Iz,  -self.d/self.Iz,  self.d/self.Iz],
                                 [0.,    0.,    0.,    0.],
                                 [0.,    0.,    0.,    0.],
                                 [-self.b/self.mass, -self.b/self.mass, -self.b/self.mass, -self.b/self.mass],
                                 [0.,    0.,    0.,    0.],
                                 [0.,    0.,    0.,    0.],
                                 [0.,    0.,    0.,    0.]])

        return fx, fu
