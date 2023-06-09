import pytorch_kinematics as pk
import math
import numpy as np
import torch



class RobotArm:
    def __init__(self, urdf):
        self.kinematic_chain = pk.build_serial_chain_from_urdf(open(urdf).read(), 'panda_hand')
        self.joints = self.kinematic_chain.get_joint_parameter_names()
        self.num_joints = len(self.joints)
        self.links = []
        self.chain_traverse(self.kinematic_chain._root, self.links)
        self.num_links = len(self.links)
        self.shoulder = 'panda_link0'
        self.elbow = 'panda_link5'
        self.wrist = 'panda_hand'
        self.joint_angles = torch.tensor([0,0, 0,0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.arm_pose = self.kinematic_chain.forward_kinematics(self.joint_angles, end_only = False)
        self.keypoint_pos = self.getKeyPointPoses()

    def chain_traverse(self, root, link_list):
        link_list.append(root.name)
        for children in root.children:
            self.chain_traverse(children, link_list)
        return
    
    # def setElbow(self):
    #     ind = int(self.num_joints/2)
    #     for x in range(self.num_joints):
    #         if x == ind:
    #             self.elbow = self.joints[x]
    
    # def setWrist(self):
    #     self.wrist = self.joints[self.num_joints - 1]
    # def setShoulder(self):
    #     self.shoulder = self.joints[0]
    def setJointAngles(self, joint_angles):
        self.joint_angles = joint_angles
        #self.joint_angles.requires_grad_ = True
        self.arm_pose = self.kinematic_chain.forward_kinematics(joint_angles, end_only = False)
    
    def getKeyPointPoses(self):
        keypoint_dict = {self.shoulder: torch.tensor([0, 0, 0], requires_grad=True, dtype=torch.float), self.elbow: torch.tensor([0, 0, 0]), self.wrist: torch.tensor([0, 0, 0])}
        shoulder_pose = self.arm_pose[self.shoulder]
        shoulder_matrix = shoulder_pose.get_matrix()
        shoulder_pos = shoulder_matrix[:, :3, 3]
        for keys in self.arm_pose:
            if keys in self.elbow:
                elbow_pose = self.arm_pose[self.elbow]
                elbow_matrix = elbow_pose.get_matrix()
                elbow_pos = elbow_matrix[:, :3, 3]
                keypoint_dict[self.elbow] = elbow_pos[0] - shoulder_pos[0]
            elif keys in self.wrist:
                wrist_pose = self.arm_pose[self.wrist]
                wrist_matrix = wrist_pose.get_matrix()
                wrist_pos = wrist_matrix[:, :3, 3]
                keypoint_dict[self.wrist] = wrist_pos[0] - shoulder_pos[0]
        return keypoint_dict
            
               

    

panda_robot = RobotArm('robot_description/panda.urdf')
# print(panda_robot.kinematic_chain)
# print("----------------")
# print(panda_robot.joints)
# print("----------------")
# print(panda_robot.num_joints)
# print("----------------")
# print(panda_robot.links)
# print("----------------")
# print(panda_robot.num_links)
# print("----------------")
# print(panda_robot.shoulder)
# print("----------------")
# print(panda_robot.elbow)
# print("----------------")
# print(panda_robot.wrist)
# print("----------------")
# print(panda_robot.joint_angles)
# print("----------------")
# print(panda_robot.joint_angles)
# print(panda_robot.arm_pose)
# print("----------------")
# print(panda_robot.keypoint_pos)
# print("----------------")