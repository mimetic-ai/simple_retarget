import pytorch_kinematics as pk
import math
import numpy as np
import torch

chain = pk.build_serial_chain_from_urdf(open('robot_description/panda.urdf').read(), 'panda_hand')
# require gradient through the input joint values
# th = torch.tensor([0.0, -math.pi / 4.0, 0.0, math.pi / 2.0, 0.0, math.pi / 4.0, 0.0], requires_grad=True)
# tg = chain.forward_kinematics(th)
# m = tg.get_matrix()
# pos = m[:, :3, 3]
# pos.norm().backward()

pred = torch.rand(5, 1)
actual = torch.rand(5, 1)