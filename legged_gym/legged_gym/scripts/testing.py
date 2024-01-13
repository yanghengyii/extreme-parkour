import numpy as np
import sys
from datetime import datetime

from isaacgym.torch_utils import *
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from math import sqrt
import math
import torch
import torchvision
import time
import cv2

class init_state:
    pos = [0.0, 0., 0.3] # x,y,z [m]
    rot = [0.0, 0.0, 0.0, 0.1] # x,y,z,w [quat]
    lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
    ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]

class depth:
    use_camera = True
    camera_num_envs = 192
    camera_terrain_num_rows = 10
    camera_terrain_num_cols = 20

    position = [0.275309, 0.025, 0.114282]  # front camera
    angle = [-5, 5]  # positive pitch down

    update_interval = 5  # 5 works without retraining, 8 worse

    original = (106, 60)
    resized = (87, 58)
    horizontal_fov = 87
    buffer_len = 2
    
    near_clip = 0
    far_clip = 2
    dis_noise = 0.0
    
    scale = 1
    invert = True

def attach_camera(gym, sim, env_handle, actor_handle):
    config = depth
    camera_props = gymapi.CameraProperties()
    camera_props.width = depth.original[0]
    camera_props.height = depth.original[1]
    camera_props.enable_tensors = True
    camera_horizontal_fov = depth.horizontal_fov 
    camera_props.horizontal_fov = camera_horizontal_fov

    camera_handle = gym.create_camera_sensor(env_handle, camera_props)
    #self.cam_handles.append(camera_handle)
    
    local_transform = gymapi.Transform()
    
    camera_position = np.copy(config.position)
    camera_angle = np.random.uniform(config.angle[0], config.angle[1])
    
    local_transform.p = gymapi.Vec3(*camera_position)
    
    quat = [-0.545621, 0.545621, -0.4497752, 0.4497752]
    #quat = [0.000, 0.0958, 0.0000, 0.9954]
    
    
    # local_transform.r = gymapi.Quat(*quat)
    local_transform.r = gymapi.Quat.from_euler_zyx(0,0.1920,0)
    root_handle = gym.get_actor_root_rigid_body_handle(env_handle, actor_handle)
    
    gym.attach_camera_to_body(camera_handle, env_handle, root_handle, local_transform, gymapi.FOLLOW_TRANSFORM)
    return camera_handle

def crop_depth_image(depth_image):
    # crop 30 pixels from the left and right and and 20 pixels from bottom and return croped image
    return depth_image[:-2, 4:-4]

def normalize_depth_image(depth_image):
    depth_image = depth_image * -1
    depth_image = (depth_image - depth.near_clip) / (depth.far_clip - depth.near_clip)  - 0.5
    return depth_image

def process_depth_image(depth_image, env_id):
    # These operations are replicated on the hardware
    resize_transform = torchvision.transforms.Resize((depth.resized[1], depth.resized[0]), 
                                                              interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
    
    depth_image = crop_depth_image(depth_image)
    depth_image += depth.dis_noise * 2 * (torch.rand(1)-0.5)[0]
    depth_image = torch.clip(depth_image, -depth.far_clip, -depth.near_clip)
    depth_image = resize_transform(depth_image[None, :]).squeeze()
    depth_image = normalize_depth_image(depth_image)
    return depth_image

def update_image(gym,sim,envs,cam_handles):

    gym.step_graphics(sim) # required to render in headless mode
    gym.render_all_camera_sensors(sim)
    gym.start_access_image_tensors(sim)
    
    depth_images = []
    
    for i in range(2):
        depth_image_ = gym.get_camera_image_gpu_tensor(sim, 
                                                       envs[i], 
                                                       cam_handles[i],
                                                       gymapi.IMAGE_DEPTH)
        
        depth_image_ = gymtorch.wrap_tensor(depth_image_)
        depth_images.append(process_depth_image(depth_image_, i))

    gym.end_access_image_tensors(sim)
    return depth_images




gym = gymapi.acquire_gym()

args = gymutil.parse_arguments(
    description="Testing robot dofs",
    custom_parameters=[
        {"name": "--num_envs", "type": int, "default": 2,
            "help": "Number of environments to create"}
    ]
)

sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UpAxis.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.,0.,-9.81)
if args.physics_engine == gymapi.SIM_FLEX:
    sim_params.flex.shape_collision_margin = 0.25
    sim_params.flex.num_outer_iterations = 4
    sim_params.flex.num_inner_iterations = 10
elif args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.substeps = 1
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim = gym.create_sim(args.compute_device_id,
                     args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

asset_root = "/home/yanghengyi/xuande_ws/extreme-parkour_2/legged_gym/resources/robots"

cyberdog2_asset_opt = gymapi.AssetOptions()
cyberdog2_asset_opt.default_dof_drive_mode = 1
cyberdog2_asset_opt.collapse_fixed_joints = True
cyberdog2_asset_opt.replace_cylinder_with_capsule = True
cyberdog2_asset_opt.flip_visual_attachments = True
cyberdog2_asset_opt.fix_base_link = True
cyberdog2_asset_opt.density = 0.001
cyberdog2_asset_opt.angular_damping = 0.
cyberdog2_asset_opt.linear_damping = 0.
cyberdog2_asset_opt.max_angular_velocity = 1000.
cyberdog2_asset_opt.max_linear_velocity = 1000.
cyberdog2_asset_opt.armature = 0.
cyberdog2_asset_opt.thickness = 0.01
cyberdog2_asset_opt.disable_gravity = False

cyberdog2_asset = gym.load_asset(
    sim, asset_root, "cyberdog2/urdf/cyberdog2_with_head.urdf", cyberdog2_asset_opt)
a1_asset = gym.load_asset(
    sim, asset_root, "a1/urdf/a1.urdf", cyberdog2_asset_opt)

num_envs = args.num_envs
num_per_row = 1
spacing = 1
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, 0.0, spacing)

gym.subscribe_viewer_keyboard_event(
    viewer, gymapi.KEY_ESCAPE, "QUIT")
gym.subscribe_viewer_keyboard_event(
    viewer, gymapi.KEY_V, "toggle_viewer_sync")
gym.subscribe_viewer_keyboard_event(
    viewer, gymapi.KEY_LEFT_BRACKET, "prev_id")
gym.subscribe_viewer_keyboard_event(
    viewer, gymapi.KEY_RIGHT_BRACKET, "next_id")
gym.subscribe_viewer_keyboard_event(
    viewer, gymapi.KEY_SPACE, "pause")
gym.subscribe_viewer_keyboard_event(
    viewer, gymapi.KEY_W, "vx_plus")
gym.subscribe_viewer_keyboard_event(
    viewer, gymapi.KEY_S, "vx_minus")
gym.subscribe_viewer_keyboard_event(
    viewer, gymapi.KEY_A, "left_turn")
gym.subscribe_viewer_keyboard_event(
    viewer, gymapi.KEY_D, "right_turn")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset")

np.random.seed(17)
base_init_state_list = init_state.pos + init_state.rot + init_state.lin_vel + init_state.ang_vel
base_init_state = to_torch(base_init_state_list, device="cuda", requires_grad=False)
envs = []
pose = gymapi.Transform()
pose.p = gymapi.Vec3(*base_init_state[:3])
pose.r = gymapi.Quat(*base_init_state[3:7])

env_cyberdog2 = gym.create_env(sim, env_lower, env_upper, num_per_row)
cyberdog2_handle = gym.create_actor(env_cyberdog2, cyberdog2_asset, pose, "cyberdog2", 0, 0)
cyberdog2_props = gym.get_actor_dof_properties(env_cyberdog2,cyberdog2_handle)

# print(cyberdog2_props)
cyberdog2_props["stiffness"] = (20.0,)*12
cyberdog2_props["damping"] = (0.5,)*12
gym.set_actor_dof_properties(env_cyberdog2,cyberdog2_handle,cyberdog2_props)

env_a1 = gym.create_env(sim, env_lower, env_upper, num_per_row)
a1_handle = gym.create_actor(env_a1, a1_asset, pose, "a1", 1, 0)
a1_props = gym.get_actor_dof_properties(env_a1,a1_handle)

# print(a1_props)
a1_props["stiffness"] = (30.0,)*12
a1_props["damping"] = (1,)*12
gym.set_actor_dof_properties(env_a1,a1_handle,a1_props)


envs.append(env_cyberdog2)
envs.append(env_a1)

cyb_camera = attach_camera(gym, sim, env_cyberdog2, cyberdog2_handle)
a1_camera = attach_camera(gym, sim, env_a1,        a1_handle)

a1_camera_frame = gymutil.AxesGeometry(0.1)
cyb_camera_frame = gymutil.AxesGeometry(0.2)

cameras = []
cameras.append(cyb_camera)
cameras.append(a1_camera)



gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(
    4, 1, 1), gymapi.Vec3(0, 0, 1))

initial_state = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))

pause = False

pos = np.arange(-math.pi,math.pi,0.01)
flipped_pos = pos[::-1]

joint_pos = np.concatenate((pos,flipped_pos))

pos_ind=0
dof_ind = 0

while not gym.query_viewer_has_closed(viewer):
    
    dof_ind %= 12

    # Get input actions from the viewer and handle them appropriately
    for evt in gym.query_viewer_action_events(viewer):
        if evt.action == "reset" and evt.value > 0:
            gym.set_sim_rigid_body_states(sim, initial_state, gymapi.STATE_ALL)
        if evt.action == "pause" and evt.value > 0:
            pause = True
            while pause:
                time.sleep(0.1)
                gym.draw_viewer(viewer, sim, True)
                for evt in gym.query_viewer_action_events(viewer):
                    if evt.action == "pause" and evt.value > 0:
                        pause = False
                    if gym.query_viewer_has_closed(viewer):
                        sys.exit()
                        
    for i in joint_pos:
        # step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        # update the viewer
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        
        # gym.set_dof_target_position(env_cyberdog2, abs(dof_ind-1), 0.0)
        # gym.set_dof_target_position(env_a1, abs(dof_ind-1), 0.0)
        
        # gym.set_dof_target_position(env_cyberdog2, dof_ind, i)
        # gym.set_dof_target_position(env_a1, dof_ind, i)
        a1_transform = gym.get_camera_transform(sim, env_a1, a1_camera)
        print("Camera pose is:")
        print(a1_transform.p)
        print(a1_transform.r)

        cyb_transform = gym.get_camera_transform(sim, env_cyberdog2, cyb_camera)
        print("Camera pose is:")
        print(cyb_transform.p)
        print(cyb_transform.r)
        
        depth_images = update_image(gym,sim,envs,cameras)
        
        gymutil.draw_lines(cyb_camera_frame,gym,viewer,env_cyberdog2,cyb_transform)
        gymutil.draw_lines(a1_camera_frame,gym,viewer,env_a1,a1_transform)
        
        window_name = "Depth Image cy"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow("Depth Image cy",depth_images[0].cpu().numpy() + 0.5)
        cv2.waitKey(1)
        
        window_name = "Depth Image a1"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow("Depth Image cy",depth_images[1].cpu().numpy() + 0.5)
        cv2.waitKey(1)
                
    
    dof_ind += 1
    
    
    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
