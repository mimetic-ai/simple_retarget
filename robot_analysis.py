import pytorch_kinematics as pk
import math
import numpy as np
import torch



class TiagoDual:
    def __init__(self, urdf):
        self.kinematic_chain_l = pk.build_serial_chain_from_urdf(open(urdf).read(), 'arm_left_7_link')
        self.kinematic_chain_r = pk.build_serial_chain_from_urdf(open(urdf).read(), 'arm_right_7_link')
        self.shoulder_l = 'arm_left_1_link'
        self.elbow_l = 'arm_left_3_link'
        self.wrist_l = 'arm_left_7_link'
        self.shoulder_r = 'arm_right_1_link'
        self.elbow_r = 'arm_right_3_link'
        self.wrist_r = 'arm_right_7_link'
        self.joint_angles_l = torch.tensor([0,0, 0,0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.arm_pose_l = self.kinematic_chain_l.forward_kinematics(self.joint_angles_l, end_only = False)
        self.joint_angles_r = torch.tensor([0,0, 0,0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.arm_pose_r = self.kinematic_chain_r.forward_kinematics(self.joint_angles_r, end_only = False)
        self.keypoint_pos_l = self.getKeyPointPosesLeft()
        self.keypoint_pos_r = self.getKeyPointPosesRight()
    
    
    
    def getElbowPosition(self, arm_name):
        link1 = "arm_{}_3_link".format(arm_name)
        link2 = "arm_{}_4_link".format(arm_name)
        if arm_name == "left":
            elbow_pos_high = self.arm_pose_l[link1]
            elbow_pos_low = self.arm_pose_l[link2]
        elif arm_name == "right":
            elbow_pos_high = self.arm_pose_r[link1]
            elbow_pos_low = self.arm_pose_r[link2]
        link1_pos = elbow_pos_high.get_matrix()[:, :3, 3][0]
        link2_pos = elbow_pos_low.get_matrix()[:, :3, 3][0]
        elbow_pos = (link1_pos + link2_pos)/2
        return elbow_pos
    
    def setJointAnglesLeft(self, joint_angles):
        self.joint_angles_l = torch.cat((torch.tensor([0.0]), joint_angles), 0)
        print(self.joint_angles_l)
        #self.joint_angles.requires_grad_ = True
        self.arm_pose_l = self.kinematic_chain_l.forward_kinematics(self.joint_angles_l, end_only = False)
        # print(self.arm_pose)
    def setJointAnglesRight(self, joint_angles):
        self.joint_angles_r = torch.cat((torch.tensor([0.0]), joint_angles), 0)
        #self.joint_angles.requires_grad_ = True
        self.arm_pose_r = self.kinematic_chain_r.forward_kinematics(self.joint_angles_r, end_only = False)
        # print(self.arm_pose)
    
    def getKeyPointPosesLeft(self):
        keypoint_dict = {self.shoulder_l: torch.tensor([0, 0, 0], requires_grad=True, dtype=torch.float), self.elbow_l: torch.tensor([0, 0, 0]), self.wrist_l: torch.tensor([0, 0, 0])}
        shoulder_pose = self.arm_pose_l[self.shoulder_l]
        shoulder_matrix = shoulder_pose.get_matrix()
        shoulder_pos = shoulder_matrix[:, :3, 3]
        print("shoulder position original ", shoulder_pos[0])
        for keys in self.arm_pose_l:
            if keys in self.elbow_l:
                elbow_pose = self.arm_pose_l[self.elbow_l]
                elbow_matrix = elbow_pose.get_matrix()
                elbow_pos = elbow_matrix[:, :3, 3]
                keypoint_dict[self.elbow_l] = elbow_pos[0] - shoulder_pos[0]
            elif keys in self.wrist_l:
                wrist_pose = self.arm_pose_l[self.wrist_l]
                wrist_matrix = wrist_pose.get_matrix()
                wrist_pos = wrist_matrix[:, :3, 3]
                print("wrist pos original ", wrist_pos[0])
                keypoint_dict[self.wrist_l] = wrist_pos[0] - shoulder_pos[0]
        return keypoint_dict
    def getKeyPointPosesRight(self):
        keypoint_dict = {self.shoulder_r: torch.tensor([0, 0, 0], requires_grad=True, dtype=torch.float), self.elbow_r: torch.tensor([0, 0, 0]), self.wrist_r: torch.tensor([0, 0, 0])}
        shoulder_pose = self.arm_pose_r[self.shoulder_r]
        shoulder_matrix = shoulder_pose.get_matrix()
        shoulder_pos = shoulder_matrix[:, :3, 3]
        for keys in self.arm_pose_r:
            if keys in self.elbow_r:
                elbow_pose = self.arm_pose_r[self.elbow_r]
                elbow_matrix = elbow_pose.get_matrix()
                elbow_pos = elbow_matrix[:, :3, 3]
                keypoint_dict[self.elbow_r] = elbow_pos[0] - shoulder_pos[0]
            elif keys in self.wrist_r:
                wrist_pose = self.arm_pose_r[self.wrist_r]
                wrist_matrix = wrist_pose.get_matrix()
                wrist_pos = wrist_matrix[:, :3, 3]
                keypoint_dict[self.wrist_r] = wrist_pos[0] - shoulder_pos[0]
        return keypoint_dict

class RobotArm:
    def __init__(self, urdf):
        self.kinematic_chain = pk.build_serial_chain_from_urdf(open(urdf).read(), 'yumi_link_7_l')
        self.joints = self.kinematic_chain.get_joint_parameter_names()
        self.num_joints = len(self.joints)
        self.links = []
        self.chain_traverse(self.kinematic_chain._root, self.links)
        self.num_links = len(self.links)
        self.shoulder = 'panda_link0'
        self.elbow = 'panda_link3'
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
        # print(self.arm_pose)
    
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
            
               

    

# panda_robot = YumiArm('robot_description/yumi.urdf')
# print(panda_robot.kinematic_chain_l)
# print(panda_robot.kinematic_chain_r)
# print("----------------")
# print(panda_robot.arm_pose_l)
# print(panda_robot.arm_pose_r)
# print("----------------")
# print(panda_robot.keypoint_pos_l)
# print(panda_robot.keypoint_pos_r)
# print("----------------")