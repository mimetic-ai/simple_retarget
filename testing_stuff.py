import json
import math
import pytorch_kinematics as pk
# import pinocchio as pin
import torch

# urdf = 'robot_description/tiago_right_arm_1.urdf'
# chain = pk.build_serial_chain_from_urdf(open(urdf).read(), 'arm_right_7_link')
# print(chain.get_joint_parameter_names())

# # specify joint values (can do so in many forms)
# th = [0.0, -math.pi / 4.0, 0.0, math.pi / 2.0, 0.0, math.pi / 4.0, 0.0, 0.0]
# # do forward kinematics and get transform objects; end_only=False gives a dictionary of transforms for all links
# ret = chain.forward_kinematics(th, end_only=False)


# # print(chain)
# # print(ret)
# # # model = pin.buildModelFromUrdf('robot_description/tiago_dual.urdf.xacro')
# # # pos_lim_lo = model.lowerPositionLimit
# # # pos_lim_hi = model.upperPositionLimit
# # # pos_lim_hi_left = pos_lim_hi[0:7]
# # # pos_lim_hi_right = pos_lim_hi[7:14]

# # # pos_lim_lo_left = pos_lim_lo[0:7]
# # # pos_lim_lo_right = pos_lim_lo[7:14]

# model = pin.buildModelFromUrdf(urdf)
# pos_lim_lo = model.lowerPositionLimit
# pos_lim_hi = model.upperPositionLimit
# print(pos_lim_lo)
# print(pos_lim_hi)

# # path = "AMCParser/boxing_arm_data_2.json"
# # f = open(path)
# # data = json.load(f)
# # print(data['rwrist'][236])
# # print(data['rhumerus'][236])
# # print(data['rclavicle'][236])

left_ub = torch.tensor([2.74889357, 1.09083078, 1.57079633, 2.35619449, 2.0943951, 1.57079633, 2.0943951])
left_lb = torch.tensor([ 0.,-1.57079633, -3.53429174, -0.39269908,-2.0943951,-1.57079633,-2.0943951])

sample_input = torch.tensor([1000.0, -1000.0, -0.0, 1.0, -1.0, -0.1, 0.1])

def bounded_output(x, lower, upper):
    scale = torch.tensor([u - l for u, l in zip(upper, lower)])
    lower = torch.tensor(lower)
    return torch.relu(x) * scale + lower


print(bounded_output(x=sample_input, lower=left_lb, upper=left_ub))

