import kinpy as kp
import math
import numpy as np

# chain = kp.build_chain_from_urdf(open('robot_description/panda.urdf').read())
# print(chain)
# print(chain.get_joint_parameter_names())
# th = {'panda_joint1': math.pi / 4.0, 'panda_joint2': math.pi / 4.0, 'panda_joint3': math.pi / 4.0, 'panda_joint4': math.pi / 4.0, 'panda_joint5': math.pi / 2.0, 'panda_joint6': math.pi / 2.0, 'panda_joint7': math.pi / 2.0}
# ret = chain.forward_kinematics(th)
# print(ret)
# for items in ret:
#     print(ret[items].pos)
# print(chain)
# print(chain._root)
# print(chain._root.children[1].name)

# link_list = []
# def chain_traverse(root, link_list):
#     link_list.append(root.name)
#     for children in root.children:
# #         chain_traverse(children, link_list)
# #     return
# # chain_traverse(chain._root, link_list)
# print(link_list)

class RobotArm:
    def __init__(self, urdf):
        self.kinematic_chain = kp.build_chain_from_urdf(open(urdf).read())
        self.joints = self.kinematic_chain.get_joint_parameter_names()
        self.num_joints = len(self.joints)
        self.links = []
        self.chain_traverse(self.kinematic_chain._root, self.links)
        self.num_links = len(self.links)
        self.shoulder = 'panda_link0_frame'
        self.elbow = 'panda_link5_frame'
        self.wrist = 'panda_hand_frame'
        self.joint_angles = {'panda_joint1': math.pi / 4.0, 'panda_joint2': math.pi / 4.0, 'panda_joint3': math.pi / 4.0, 'panda_joint4': math.pi / 4.0, 'panda_joint5': math.pi / 2.0, 'panda_joint6': math.pi / 2.0, 'panda_joint7': math.pi / 2.0}
        self.arm_pose = self.kinematic_chain.forward_kinematics(self.joint_angles)
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
        for x in range(len(joint_angles)):
            self.joint_angles[self.joints[x]] = math.radians(float(joint_angles[x]))

    def updateArmPose(self):
        self.arm_pose = self.kinematic_chain.forward_kinematics(self.joint_angles)
    
    def getKeyPointPoses(self):
        keypoint_dict = {self.shoulder: np.asarray([0, 0, 0]), self.elbow: np.asarray([0, 0, 0]), self.wrist: np.asarray([0, 0, 0])}
        for keys in self.arm_pose:
            if keys in self.elbow:
                keypoint_dict[self.elbow] = np.asarray(self.arm_pose[keys].pos) - np.asarray(self.arm_pose['panda_link0'].pos)
            elif keys in self.wrist:
                keypoint_dict[self.wrist] = np.asarray(self.arm_pose[keys].pos) - np.asarray(self.arm_pose['panda_link0'].pos)
        return keypoint_dict
            
               

    

panda_robot = RobotArm('robot_description/panda.urdf')
print(panda_robot.kinematic_chain)
print("----------------")
print(panda_robot.joints)
print("----------------")
print(panda_robot.num_joints)
print("----------------")
print(panda_robot.links)
print("----------------")
print(panda_robot.num_links)
print("----------------")
print(panda_robot.shoulder)
print("----------------")
print(panda_robot.elbow)
print("----------------")
print(panda_robot.wrist)
print("----------------")
print(panda_robot.joint_angles)
print("----------------")
print(panda_robot.joint_angles)
print(panda_robot.arm_pose)
print("----------------")
print(panda_robot.keypoint_pos)
print("----------------")