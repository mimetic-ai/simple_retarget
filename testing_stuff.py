import math
import torch
import pytorch_kinematics as pk
import pinocchio as pin
import numpy as np


chain = pk.build_serial_chain_from_urdf(open('robot_description/panda.urdf').read(), 'panda_hand')
th = torch.tensor([0.0, -math.pi / 4.0, 0.0, math.pi / 2.0, 0.0, math.pi / 4.0, 0.0])
# (1,6,7) tensor, with 7 corresponding to the DOF of the robot
J = chain.jacobian(th)
J_w = J[0][0:3]
J_w = torch.reshape(J_w, (1, 3, 7))
print(J_w)


chain_2 = pk.build_serial_chain_from_urdf(open('robot_description/panda.urdf').read(), 'panda_link5')
th = torch.tensor([0.0, -math.pi / 4.0, 0.0, math.pi / 2.0, 0.0, math.pi / 4.0, 0.0])
# (1,6,7) tensor, with 7 corresponding to the DOF of the robot
J_2 = chain_2.jacobian(th)
J_e = J_2[0][0:3]
J_e = torch.reshape(J_e, (1, 3, 7))
print(J_e)


model = pin.buildModelFromUrdf('robot_description/panda.urdf')
pos_lim_lo = np.array(model.lowerPositionLimit)
pos_lim_hi = np.array(model.upperPositionLimit)
print(pos_lim_hi)
print(pos_lim_lo)
