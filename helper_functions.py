import numpy as np

def quaternion2Angle(quat):
    qw, qx, qy, qz = quat.w_val, quat.x_val, quat.y_val, quat.z_val
    sinr_cosp = 2*(qw*qx+qy*qz)
    cosr_cosp = 1-2*(qx**2+qy**2)
    phi = np.atan2(sinr_cosp, cosr_cosp)

    sinp = 2*(qw*qy-qz**2)
    theta = np.asin(sinp)
    if sinp >= 1:
        theta = np.sign(sinp)*np.pi/2

    siny_cosp = 2*(qw*qz+qx*qy)
    cosy_cosp = 1-2*(qy**2+qz**2)
    psi = np.atan2(siny_cosp, cosy_cosp)
    return phi, theta, psi