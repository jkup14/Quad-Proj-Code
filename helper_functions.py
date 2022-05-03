import numpy as np
from torch.utils.data import Dataset, DataLoader

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

class DynamicsDataset(Dataset):
  """
  This is a custom dataset class. It can get more complex than this, but simplified so you can understand what's happening here without
  getting bogged down by the preprocessing
  """
  def __init__(self, X, Y):
    self.X = X
    self.Y = Y
    if len(self.X) != len(self.Y):
      raise Exception("The length of X does not match the length of Y")

  def __len__(self):
    return len(self.X)

  def __getitem__(self, index):
    # note that this isn't randomly selecting. It's a simple get a single item that represents an x and y
    _x = self.X[index]
    _y = self.Y[index]

    return _x, _y
