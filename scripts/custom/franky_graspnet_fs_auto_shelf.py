import argparse
import numpy as np
import open3d as o3d
import sys
import os
import traceback
import cv2
import time

# Note: added to subdirectory so needs to go up two levels
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(os.path.join(ROOT_DIR, '..'))
from franka_graspnet.franka_controller import FrankaController
from franka_graspnet.graspnet_infer import GraspNetInfer
from franka_graspnet.stereo_realsense import StereoCameraIR
from foundation_stereo.fs_infer import FoundationStereoInfer

class FrankyShelfPickAndPlace:
    def __init__(self):
        # Parse the input arguments
        self.parser = argparse.ArgumentParser()
        self.args = self.parse_args()
        
        # Params
        self.robot = FrankaController(robot_ip="192.168.1.1")
        self.max_valid_depth = 0.5
        self.workspace = ((self.args.height, self.args.width), (200, 0), (1150, 600))
        self.approach_axis = 'x'
        self.open_axis = 'y'

        # Initialize the camrera
        self.rs_dev, self.R_ir2color, self.T_ir2color, self.fx_c, self.fy_c, self.ppx_c, self.ppy_c, self.fs = self.init_camera()
        # Initialize the robot
        self.init_robot()   # Move to the home position
        # Initialize graspnet
        self.graspnet_infer = graspnet_infer = GraspNetInfer(self.args)
        # Create workspace mask
        self.workspace_mask = self.create_rect_mask(self.workspace)

    def parse_args(self):
        self.parser.add_argument('--checkpoint_path', type=str, default="checkpoints/graspnet/checkpoint-rs.tar")
        self.parser.add_argument('--num_point', type=int, default=20000)
        self.parser.add_argument('--num_view', type=int, default=300)
        self.parser.add_argument('--collision_thresh', type=float, default=0.01)
        self.parser.add_argument('--angle_threshold_deg', type=float, default=30)
        self.parser.add_argument('--voxel_size', type=float, default=0.01)

        # for camera
        self.parser.add_argument('--width', type=int, default=1280)
        self.parser.add_argument('--height', type=int, default=720)

        # FS
        self.parser.add_argument('--fs_ckpt_dir', type=str, default='./checkpoints/foundation_stereo/11-33-40/model_best_bp2.pth')
        self.parser.add_argument('--baseline', type=float, default=None)
        self.parser.add_argument('--device', type=str, default='cuda')
        self.parser.add_argument('--scale', default=1, type=float)
        self.parser.add_argument('--hiera', default=0, type=int)
        self.parser.add_argument('--z_far', default=10, type=float)
        self.parser.add_argument('--valid_iters', type=int, default=32)
        self.parser.add_argument('--denoise_cloud', type=int, default=1)
        self.parser.add_argument('--denoise_nb_points', type=int, default=30)
        self.parser.add_argument('--denoise_radius', type=float, default=0.03)
        return self.parser.parse_args()

    def init_camera(self):
        print("Init realsense camera.......")
        rs_dev = StereoCameraIR(width=self.args.width, height=self.args.height, fps=30)
        try:
            rs_dev.start()
        except Exception as e:
            print("Failed to start RealSense:", e)
            traceback.print_exc()
            sys.exit(1)
        
        # Read the baseline from the camera if it is not specified
        if self.args.baseline is None:
            # Read baseline from the camera if not specified
            self.args.baseline = rs_dev.get_stereo_baseline()

        # IR to Color
        R_ir2color, T_ir2color = rs_dev.get_ir2color()
        # Color Intrinsics
        fx_c, fy_c, ppx_c, ppy_c = rs_dev.get_color_intrinsics()

        # IR to Color
        R_ir2color, T_ir2color = rs_dev.get_ir2color()
        # Color Intrinsics
        fx_c, fy_c, ppx_c, ppy_c = rs_dev.get_color_intrinsics()

        fs = FoundationStereoInfer(ckpt_dir=self.args.fs_ckpt_dir, device=self.args.device, baseline=self.args.baseline,
                valid_iters=self.args.valid_iters, hiera=self.args.hiera)

        return rs_dev, R_ir2color, T_ir2color, fx_c, fy_c, ppx_c, ppy_c, fs

    def init_robot(self):
        # === Robot Init ===
        print("Move robot to home...")
        self.robot.move_home()
        self.robot.open_gripper()

    def create_rect_mask(self, image_shape, top_left, bottom_right):
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        x1, y1 = top_left
        x2, y2 = bottom_right
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        mask[y1:y2, x1:x2] = True
        return mask

    def process_camera_frames(self):
        res = self.rs_dev.get_frames()
        if res is None:
            print("Unable to retrieve camera frames")
            time.sleep(0.1)
            return None, None
        color_image, depth_raw, left_rgb, right_rgb, K_left = res

        scale = float(self.args.scale)
        H0,W0 = left_rgb.shape[:2]
        newW,newH = int(W0*scale), int(H0*scale)
        if scale != 1.0:
            left_rgb = cv2.resize(left_rgb,(newW,newH))
            right_rgb = cv2.resize(right_rgb,(newW,newH))
            color_vis = cv2.resize(color_image,(newW,newH))
        else:
            color_vis = color_image.copy()

        depth_m, valid_mask, K_scaled = self.fs.infer_depth(left_rgb, right_rgb, K_left, scale=scale)
        
        depth_m[np.isinf(depth_m)] = 0
        depth_m[depth_m > self.max_valid_depth] = 0
        valid_mask = valid_mask & (depth_m > 0) & (depth_m <= self.args.z_far) & (self.workspace_mask > 0)
        if np.any(valid_mask):
            print(f"Depth min/max (valid only): {depth_m[valid_mask].min():.3f}/{depth_m[valid_mask].max():.3f}")

        H, W = depth_m.shape
        xx, yy = np.meshgrid(np.arange(W), np.arange(H))
        z = depth_m

        fx = K_scaled[0,0]; ppx = K_scaled[0,2]
        fy = K_scaled[1,1]; ppy = K_scaled[1,2]
        x = (xx - ppx) / fx * z
        y = (yy - ppy) / fy * z
        points_ir = np.stack([x, y, z], axis=-1).reshape(-1,3)
        mask_flat = valid_mask.reshape(-1)

        # transform LEFT IR points to Color
        points_color = (self.R_ir2color @ points_ir.T).T + self.T_ir2color

        # project to color image
        pts_z = points_color[:,2]
        valid_z = pts_z > 1e-6
        u = (points_color[:,0] / np.where(valid_z, pts_z, 1.0)) * self.fx_c + self.ppx_c
        v = (points_color[:,1] / np.where(valid_z, pts_z, 1.0)) * self.fy_c + self.ppy_c
        u_int = np.round(u).astype(np.int32)
        v_int = np.round(v).astype(np.int32)

        in_bounds = (u_int >= 0) & (u_int < color_vis.shape[1]) & (v_int >= 0) & (v_int < color_vis.shape[0]) & valid_z
        final_mask = mask_flat & in_bounds
        idxs = np.where(final_mask)[0]

        pts_keep = points_color[idxs]
        u_sel = u_int[idxs]; v_sel = v_int[idxs]
        colors_keep = cv2.cvtColor(color_vis, cv2.COLOR_BGR2RGB)[v_sel, u_sel, :].astype(np.float32) / 255.0
        
        end_points, cloud = self.graspnet_infer.process_fs_data(pts_keep, colors_keep)
        
        return end_points, cloud
    
    
    def get_grasp(self, end_points, cloud):
        try:
            target_gg = self.graspnet_infer.predict_grasps(end_points, cloud)
            return target_gg
        except IndexError:
            # Workaround to avoid modifying the function in grasp generation file
            # When there are no grasps left, it attempts to access the element at the zero index of an unpopulated list of ranked grasps
            return None

    def get_target_pose(self, target_gg, cloud):
        grippers = target_gg.to_open3d_geometry_list()
        T = np.diag([1, 1, -1, 1])
        cloud.transform(T)
        for g in grippers:
            g.transform(T)  
        target_pose_base = self.robot.compute_target_pose(target_gg[0], self.approach_axis, self.open_axis)
        return target_pose_base   


    def run(self):
        print("Starting real-time pick-and-place ...")
        try:
            while True:
                end_points, cloud = self.process_camera_frames()

                if end_points is None or cloud is None:
                    continue

                grasp = self.get_grasp(end_points, cloud)

                if not grasp: 
                    print("All objects have been placed")
                    break

                target_pose = self.get_target_pose(grasp)

                self.robot.execute_grasp(target_pose)

        except KeyboardInterrupt:
            print("Exit demo.")
            self.rs_dev.stop()
            sys.exit(0)
            
        except Exception as e: 
            print("An unexpected error, has ocurred, exiting program:", e)
            self.rs_dev.stop()
            raise

if __name__ == "__main__":
    script = FrankyShelfPickAndPlace()
    script.run()
