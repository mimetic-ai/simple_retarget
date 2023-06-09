import cyipopt
import numpy as np
import pytorch_kinematics as pk
from image_proc import *
from simple_retarget import *
from robot_analysis import RobotArm
import matplotlib.pyplot as plt
import kinpy as kp
import pinocchio as pin
import math


class IVPAngles():
    def __init__(self):
        self.robot = RobotArm('robot_description/panda.urdf')
        self.arm_pose = wrtShoulder(getArmPosesFrame('media/sample_retarget_pose.jpg'))

    def objective(self, x):
        """Returns the scalar value of the objective given x."""

        ##x is the joint angles
        self.robot.setJointAngles(x)
        keypoint_dict = self.robot.getKeyPointPoses()
        loss = newLoss(arm_pos=self.arm_pose, robot_pos=keypoint_dict)
        return loss

    def gradient(self, x):
        """Returns the gradient of the objective with respect to x."""
        ##get current predicted positions and ground truth positions
        robot_pos = self.robot.getKeyPointPoses()
        true_elbow = arm_pos['l_elbow'].detach().numpy()
        true_wrist = arm_pos['l_wrist'].detach().numpy()
        pred_elbow = robot_pos['panda_link5'].detach().numpy()
        pred_wrist = robot_pos['panda_hand'].detach().numpy()
        ##normalizing those positions
        true_elbow = true_elbow/np.linalg.norm(true_elbow)
        true_wrist = true_wrist/np.linalg.norm(true_wrist)
        pred_elbow = pred_elbow/np.linalg.norm(pred_elbow)
        pred_wrist = pred_elbow/np.linalg.norm(pred_wrist)
        ##calculating the jacobians needed for the next part
        x = x.astype(float)
        J_e = self.calcJacobian(x, 'panda_link5')
        J_w = self.calcJacobian(x, 'panda_hand')
        ##derivative of loss with respect to angles
        wrist_arr = np.reshape((pred_wrist - true_wrist), (1, 3))
        elbow_arr = np.reshape((pred_elbow - true_elbow), (1, 3))
        dLossW_dAngles = 2 * np.matmul(wrist_arr, J_w[0])
        dLossE_dAngles = 2 * np.matmul(elbow_arr, J_e[0])
        dLoss_dAngles = dLossE_dAngles + dLossW_dAngles
        gradient = dLoss_dAngles
        return gradient
    
    def calcJacobian(self, angles, end_link):
        # print("angles ", angles)
        chain = pk.build_serial_chain_from_urdf(open('robot_description/panda.urdf').read(), end_link)
        ##for some reason looping through and casting is the only thing that gets rid of the error, not dtype=float
        angles_2 = []
        for items in angles:
            angles_2.append(float(items))
        th = torch.tensor(angles_2)
        # (1,6,7) tensor, with 7 corresponding to the DOF of the robot
        J = chain.jacobian(th)
        J_w = J[0][0:3]
        J_w = torch.reshape(J_w, (1, 3, 7))
        return J_w



    def constraints(self, x):
        """Returns the constraints."""
        return np.array((np.prod(x), np.dot(x, x)))

    def jacobian(self, x):
        """Returns the Jacobian of the constraints with respect to x."""
        return np.concatenate((np.prod(x)/x, 2*x))

    # def hessianstructure(self):
    #     """Returns the row and column indices for non-zero vales of the
    #     Hessian."""

    #     # NOTE: The default hessian structure is of a lower triangular matrix,
    #     # therefore this function is redundant. It is included as an example
    #     # for structure callback.

    #     return np.nonzero(np.tril(np.ones((7, 7))))

    # def hessian(self, x, lagrange, obj_factor):
    #     """Returns the non-zero values of the Hessian."""

    #     # print("lagrange ", lagrange)
    #     # print("lagrange shape ", lagrange.shape)
    #     # print(lagrange[0])

    #     H = obj_factor*np.array((
    #         (2*x[3], 0, 0, 0, 0, 0, 0),
    #         (x[3],   0, 0, 0, 0, 0, 0),
    #         (x[3],   0, 0, 0, 0, 0, 0),
    #         (x[3],   0, 0, 0, 0, 0, 0),
    #         (x[3],   0, 0, 0, 0, 0, 0),
    #         (x[3],   0, 0, 0, 0, 0, 0),
    #         (2*x[0]+x[1]+x[2], x[0], x[0], 0, 0, 0, 0)))

    #     # H += lagrange[0]*np.array((
    #     #     (0, 0, 0, 0, 0, 0, 0),
    #     #     (x[2]*x[3], 0, 0, 0, 0, 0, 0),
    #     #     (x[1]*x[3], x[0]*x[3], 0, 0, 0, 0, 0),
    #     #     (x[1]*x[3], x[0]*x[3], 0, 0, 0, 0, 0),
    #     #     (x[1]*x[3], x[0]*x[3], 0, 0, 0, 0, 0),
    #     #     (x[1]*x[3], x[0]*x[3], 0, 0, 0, 0, 0),
    #     #     (x[1]*x[2], x[0]*x[2], x[0]*x[1], 0)))

    #     # H += lagrange[1]*2*np.eye(4)

    #     row, col = self.hessianstructure()

    #     return H[row, col]

    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                     d_norm, regularization_size, alpha_du, alpha_pr,
                     ls_trials):
        """Prints information at every Ipopt iteration."""

        msg = "Objective value at iteration #{:d} is - {:g}"

        print(msg.format(iter_count, obj_value))



###using pinocchio parameters to set bounds
model = pin.buildModelFromUrdf('robot_description/panda.urdf')
pos_lim_lo = np.array(model.lowerPositionLimit)
pos_lim_hi = np.array(model.upperPositionLimit)
lb = pos_lim_lo[0:7]
ub = pos_lim_hi[0:7]
# print(lb)
# print(ub)
# cl = [25.0, 40.0]
# cu = [2.0e19, 40.0]


##initializing arm_position and robot variable
##getting initial guess 
arm_pos = wrtShoulder(getArmPosesFrame('media/sample_retarget_pose.jpg'))
panda_robot = RobotArm('robot_description/panda.urdf')
end_effector_pose = getEndEffectorPose(arm_pos)
chain = kp.build_serial_chain_from_urdf(open('robot_description/panda.urdf').read(), end_link_name = 'panda_hand')
end_effector_transform = kp.transform.Transform(rot = end_effector_pose[1], pos = end_effector_pose[0])
#end_effector_transform = kp.transform.Transform(rot = None, pos = end_effector_pose[0])
ik_results = chain.inverse_kinematics(end_effector_transform)

##setting input to ik_results
x0 = ik_results
# print(len(ik_results))
# print(ik_results)

print(len(lb))
print(len(ub))
print(len(x0))
##initialize problem parameters
nlp = cyipopt.Problem(
   n=len(x0),
   m=0,
   problem_obj=IVPAngles(),
   lb=lb,
   ub=ub,
#    cl=cl,
#    cu=cu,
)

##run optimizer
nlp.add_option('mu_strategy', 'adaptive')
nlp.add_option('tol', 0.9)

x, info = nlp.solve(x0)