import numpy as np
import time
from scipy.spatial.transform import Rotation as R
from franky import *
import threading

class FrankaController:
    def __init__(self, robot_ip: str = "192.168.1.1"):
        
        self.robot = Robot(robot_ip)

        self.robot.relative_dynamics_factor = RelativeDynamicsFactor(0.10, 0.05, 0.01)
        # self.robot.relative_dynamics_factor = 0.05

        # Set collision behavior
        lower_torque_thresholds = [20.0] * 7  # Nm
        upper_torque_thresholds = [40.0] * 7  # Nm
        lower_force_thresholds = [10.0] * 6  # N (linear) and Nm (angular)
        upper_force_thresholds = [20.0] * 6  # N (linear) and Nm (angular)
        self.robot.set_collision_behavior(
            lower_torque_thresholds,
            upper_torque_thresholds,
            lower_force_thresholds,
            upper_force_thresholds,
        )

        self.gripper = Gripper(robot_ip)

        self.HOME_JOINT_POSE = JointMotion(np.array([0. , -0.78539816,  0. , -2.35619449,  0. , 1.57079633,  0.78539816]))
        self.BOX_JOINT_POSE = JointMotion(np.array([1.12474915, -0.37809445, -0.04517126, -2.12450855,  0.00372194,  1.81733198, 0.50831428]))
        
        # Added cartesian poses to make it easier to modify
        
        T = np.array([0.50, 0.00, 0.30], dtype=np.float64)           # meters
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)         # (x,y,z,w) tool facing down
        home_affine = Affine(translation=T, quaternion=q)
        home_state  = CartesianState(home_affine)
        
        self.HOME_CARTESIAN_POSE = CartesianMotion(
            home_state,
            reference_type=ReferenceType.Absolute,
        )
        
        T = np.array([0.30, 0.50, 0.30], dtype=np.float64)           # meters
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)         # (x,y,z,w) tool facing down
        box_affine = Affine(translation=T, quaternion=q)
        box_state  = CartesianState(box_affine)
        
        self.BOX_CARTESIAN_POSE = CartesianMotion(
            box_state,
            reference_type=ReferenceType.Absolute,
        )

        # Eye-in-hand
        self.R_Cam2Gripper = np.array([
            [ 0.03457234874809523, -0.9993213945657426, 	0.012708385626025567],
            [ 0.9993995451836195, 0.03459875862875128, 0.0018641320029551822],
            [ -0.0023025613596837574, 0.012636307392949344, 0.9999175075708274]
        ])
        self.T_Cam2Gripper = np.array([[	0.07268208417508988], [	-0.01197537139497642], [-0.1161990736818651]])
        # EE 坐标系下的 Tool 偏移
        self.TOOL_IN_EE = np.array([0.00, 0.000, 0.000]) # Z越小，夹爪越向下

    def move_home(self):
        try:
            #self.robot.move(self.HOME_JOINT_POSE)
            self.robot.move(self.HOME_CARTESIAN_POSE)
        except ControlException as e: 
            print("Warning: control exception while moving to home position, trying again:", e)
            self.robot.recover_from_errors()
            time.sleep(0.5)
            self.move_home()

    def move_box(self):
        try:
            #self.robot.move(self.BOX_JOINT_POSE)
            self.robot.move(self.BOX_CARTESIAN_POSE)
        except ControlException as e: 
            print("Warning: control exception while moving to box position, trying again:", e)
            self.robot.recover_from_errors()
            time.sleep(0.5)
            self.move_box()

    def open_gripper(self, speed=0.04):
        try: 
            self.gripper.open(speed=speed)
        except ControlException as e:
            print("Warning: control exception while opening gripper, trying again:", e)
            self.robot.recover_from_errors()
            time.sleep(0.5)
            self.open_gripper(speed=speed)

    def close_gripper(self, width=0.0, speed=0.04, force=80):
        try:
            self.gripper.grasp(width, speed=speed, force=force)
        except ControlException as e:
            print("Warning: control exception while closing gripper, trying again:", e)
            self.robot.recover_from_errors()
            time.sleep(0.5)
            self.close_gripper(width=width, speed=speed, force=force)
            print("Error closing gripper")

    def hold_grasp(self, force=80, speed=0.05, stop_event=None, ready_event=None):
        while not stop_event.is_set():
            try:
                # epsilon params let grasp() succeed even if width drifts slightly
                self.gripper.grasp(width=0.0, speed=speed, force=force,
                                epsilon_inner=0.08, epsilon_outer=0.08)
                if ready_event:
                    ready_event.set()
            except (CommandException, ControlException) as e:
                print("Warning: control exception while holding grasp, trying again:", e)
                pass
            stop_event.wait(timeout=0.05)  # re-assert every 50 ms


    def compute_target_pose(self, target_gg):
        # compute the target pose in base reference frame based on the GraspNet predictions and camera extrinsics
        """根据 GraspNet 预测和相机外参计算 Base 下目标位姿"""
        # Grasp pose in camera
        R_grasp2camera = target_gg.rotation_matrix
        t_grasp2camera = target_gg.translation.reshape(3, 1)

        # Camera -> EE
        point_ee = self.R_Cam2Gripper @ t_grasp2camera + self.T_Cam2Gripper
        point_ee = point_ee.flatten()

        # EE pose in Base
        ee_pose_base = self.robot.current_cartesian_state.pose.end_effector_pose
        R_ee_base = ee_pose_base.matrix[:3, :3]

        # EE -> Base
        point_ee_in_base = ee_pose_base * Affine(point_ee, np.array([0.0, 0.0, 0.0, 1.0]))
        point_ee_in_base_pos = point_ee_in_base.translation

        # Tool 偏移在 Base 下
        tool_in_base = R_ee_base @ self.TOOL_IN_EE
        ee_target_in_base = point_ee_in_base_pos - tool_in_base

        # -----------------------------
        # 构造 EE 旋转矩阵      construct end effector rotation matrix
        # -----------------------------
        approach = R_grasp2camera[:, 0]  # GraspNet x = approach
        open_dir = R_grasp2camera[:, 1]  # GraspNet y = open
        z_ee = approach
        y_ee = open_dir
        x_ee = np.cross(y_ee, z_ee)

        R_target_ee = np.column_stack([x_ee, y_ee, z_ee])
        R_target_ee = self.R_Cam2Gripper @ R_target_ee
        R_target_base = R_ee_base @ R_target_ee

        # === 保证 z 轴朝上，消除180度的二义性 ===   ensure the gripper faces downwards
        if R_target_base[2, 2] < 0:  # 如果 z 轴朝下
            R_target_base = R_target_base @ R.from_euler("z", 180, degrees=True).as_matrix()

        # Pick the closest rotation solution to the current gripper orientation
        R_flip = R.from_euler("z", 180, degrees=True).as_matrix()
        R_target_base_alt = R_target_base @ R_flip
        def _rot_angle(Ra, Rb):
            cos_a = (np.trace(Ra.T @ Rb) - 1) / 2
            return np.arccos(np.clip(cos_a, -1, 1))
        if _rot_angle(R_ee_base, R_target_base_alt) < _rot_angle(R_ee_base, R_target_base):
            R_target_base = R_target_base_alt

        q_target_base = R.from_matrix(R_target_base).as_quat()

        return Affine(ee_target_in_base, q_target_base)

    def execute_grasp(self, target_pose_base):
        """执行一次完整抓取：移动 -> 闭合 -> 放置 -> 回 home"""
        try:
            if self.gripper.width < 0.01:
                self.open_gripper()

            time.sleep(0.2)

            # Allow a limited number of retries, give up after max (assume it is tryinng to get to an unreachable pose)
            fail_counter = 0
            fail_max = 3
            while True:
                try:
                    self.robot.move(CartesianMotion(target_pose_base, ReferenceType.Absolute))
                    break
                except ControlException as e:
                    if fail_counter < fail_max:
                        print("Warning: control exception while moving to target position, trying again:", e)
                        print("Retry " + str(fail_counter+1) + " of " + str(fail_max))
                        self.robot.recover_from_errors()
                        time.sleep(0.5)
                        fail_counter += 1
                    else:
                        print("Recovering from error and attempting a new grasp...")
                        time.sleep(0.1)
                        self.robot.recover_from_errors()
                        self.move_home()
                        return

            # Start a thread to maintain a constant grasp force
            stop_grasp = threading.Event()
            grasp_ready = threading.Event()
            hold_thread = threading.Thread(
                target=self.hold_grasp, args=(100, 0.05, stop_grasp, grasp_ready), daemon=True
            )
            hold_thread.start()

            grasp_ready.wait(timeout=5.0)   # Wait until the grasp is held

            self.move_box()

            # End the grasp hold thread
            stop_grasp.set()      
            hold_thread.join(timeout=2.0)

            self.open_gripper()

            time.sleep(0.2)

            self.move_home()
            
        except Exception as e:
            print("抓取执行出错：", e)
