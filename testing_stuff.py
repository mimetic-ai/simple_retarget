import math
import torch
import pytorch_kinematics as pk
import pinocchio as pin
import numpy as np
from image_proc import *
from robot_analysis import YumiArm
import matplotlib.pyplot as plt
import kinpy as kp

import json

# chain = pk.build_serial_chain_from_urdf(open('robot_description/panda.urdf').read(), 'panda_hand')
# th = torch.tensor([0.0, -math.pi / 4.0, 0.0, math.pi / 2.0, 0.0, math.pi / 4.0, 0.0])
# # (1,6,7) tensor, with 7 corresponding to the DOF of the robot
# J = chain.jacobian(th)
# J_w = J[0][0:3]
# J_w = torch.reshape(J_w, (1, 3, 7))
# print(J_w)


# chain_2 = pk.build_serial_chain_from_urdf(open('robot_description/panda.urdf').read(), 'panda_link5')
# th = torch.tensor([0.0, -math.pi / 4.0, 0.0, math.pi / 2.0, 0.0, math.pi / 4.0, 0.0])
# # (1,6,7) tensor, with 7 corresponding to the DOF of the robot
# J_2 = chain_2.jacobian(th)
# J_e = J_2[0][0:3]
# J_e = torch.reshape(J_e, (1, 3, 7))
# print(J_e)


# model = pin.buildModelFromUrdf('robot_description/panda.urdf')
# print(model.names)
# for names in model.names:
#     print(names)
# pos_lim_lo = np.array(model.lowerPositionLimit)
# pos_lim_hi = np.array(model.upperPositionLimit)
# print(pos_lim_hi)
# print(pos_lim_lo)
# degrees_list = np.rad2deg(pos_lim_lo)
# degrees_list_2 = np.rad2deg(pos_lim_hi)
# print(degrees_list)
# print(degrees_list_2)

# path = 'AMCParser/boxing_arm_data.json'
# f = open(path)
# data = json.load(f)
# arm_pos = {'l_shoulder': None, 'l_elbow': None, 'l_wrist': None}
# arm_pos['l_shoulder'] = torch.tensor(data['lclavicle'][0])
# arm_pos['l_elbow'] = torch.tensor(data['lhumerus'][0])
# arm_pos['l_wrist'] = torch.tensor(data['lhand'][0])

# print(arm_pos)

model = pin.buildModelFromUrdf('robot_description/yumi.urdf')
pos_lim_lo = model.lowerPositionLimit
pos_lim_hi = model.upperPositionLimit
pos_lim_hi_left = pos_lim_hi[0:7]
pos_lim_hi_right = pos_lim_hi[7:14]

pos_lim_lo_left = pos_lim_lo[0:7]
pos_lim_lo_right = pos_lim_lo[7:14]

print((pos_lim_hi_left))
print((pos_lim_lo_left))
print(model)
# urdf = 'robot_description/yumi.urdf'
# yumi_robot = YumiArm('robot_description/yumi.urdf')

path = 'tf_joint_motion.json'
f = open(path)
data = json.load(f)
print(data)