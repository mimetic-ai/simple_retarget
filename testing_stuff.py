import json
import math
import pytorch_kinematics as pk


urdf = 'robot_description/tiago_dual.urdf'
chain = pk.build_serial_chain_from_urdf(open(urdf).read(), 'arm_left_7_link')
print(chain.get_joint_parameter_names())

# specify joint values (can do so in many forms)
th = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# do forward kinematics and get transform objects; end_only=False gives a dictionary of transforms for all links
ret = chain.forward_kinematics(th, end_only=False)

print(ret)


# print(chain)
# print(ret)
# # model = pin.buildModelFromUrdf('robot_description/tiago_dual.urdf.xacro')
# # pos_lim_lo = model.lowerPositionLimit
# # pos_lim_hi = model.upperPositionLimit
# # pos_lim_hi_left = pos_lim_hi[0:7]
# # pos_lim_hi_right = pos_lim_hi[7:14]

# # pos_lim_lo_left = pos_lim_lo[0:7]
# # pos_lim_lo_right = pos_lim_lo[7:14]

# model = pin.buildModelFromUrdf(urdf)
# pos_lim_lo = model.lowerPositionLimit
# pos_lim_hi = model.upperPositionLimit
# print(pos_lim_lo)
# print(pos_lim_hi)

# path = "AMCParser/boxing_arm_data_2.json"
# f = open(path)
# data = json.load(f)
# print(data['rwrist'][236])
# print(data['rhumerus'][236])
# print(data['rclavicle'][236])

