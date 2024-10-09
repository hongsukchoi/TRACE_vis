import os, sys
import torch
import tyro
import viser
import imageio
import numpy as onp
import time
from scipy.spatial.transform import Rotation as R
from collections import defaultdict
from tqdm import tqdm
    
from trace2.utils.utils import angle_axis_to_rotation_matrix


def process_idx(reorganize_idx, vids=None):
    used_org_inds = onp.unique(reorganize_idx)
    per_img_inds = [onp.where(reorganize_idx==org_idx)[0] for org_idx in used_org_inds]

    return used_org_inds, per_img_inds


def get_color_for_sid(sid):
    # Simple hash function to generate a color
    hash_value = sid * 123456789 + 111111111
    r = (hash_value & 0xFF0000) >> 16
    g = (hash_value & 0x00FF00) >> 8
    b = hash_value & 0x0000FF
    return (b, g, r)
    

def main(sid: int = 0, result_npz: str = None):
    # smpl
    smpl_model_path = '/home/hongsuk/.romp/SMPL_NEUTRAL.pth'
    smpl_faces = torch.load(smpl_model_path)['f'].numpy().astype(onp.int32)

    # load the output
    outputs = onp.load(result_npz, allow_pickle=True)['outputs'][()]
    # dict_keys(['reorganize_idx', 'j3d', 'world_cams', 'world_trans', 'world_global_rots', 'world_verts_camed_org',
    # 'pj2d_org', 'pj2d', 'cam_trans', 'pj2d_org_h36m17', 'joints_h36m17', 'center_confs', 'track_ids', 'smpl_thetas', 'smpl_betas'])

    y_up = True
    if y_up:
        yup2ydown = angle_axis_to_rotation_matrix(torch.tensor([[onp.pi, 0, 0]])).cuda().float()
        yup2ydown = yup2ydown.expand(outputs['world_verts'].shape[0], -1, -1)
        outputs['world_verts'] = (yup2ydown.mT @ torch.from_numpy(outputs['world_verts']).cuda().mT).mT
        world_trans = (yup2ydown.mT @ torch.from_numpy(outputs['world_trans']).cuda().unsqueeze(-1)).squeeze(-1)
        outputs['world_verts'] = outputs['world_verts'] + world_trans[:, None, :]
       
        world_cam_rots = angle_axis_to_rotation_matrix(torch.from_numpy(outputs['world_cam_rots']).cuda())
        outputs['world_cam_rots'] = yup2ydown.mT @ world_cam_rots
        outputs['world_cams'] = (yup2ydown.mT @ torch.from_numpy(outputs['world_cams']).cuda().unsqueeze(-1)).squeeze(-1)

    outputs['world_verts'] = outputs['world_verts'].cpu().numpy()
    outputs['world_cam_rots'] = outputs['world_cam_rots'].cpu().numpy()
    outputs['world_cams'] = outputs['world_cams'].cpu().numpy()

    data_frames = defaultdict(dict)
    used_org_inds, per_img_inds = process_idx(outputs['reorganize_idx'])
    for org_ind, img_inds in tqdm(list(zip(used_org_inds, per_img_inds))):
        for img_ind in img_inds:
            track_id = outputs['track_ids'][img_ind]
            data_frames[org_ind][track_id] = {
                'world_verts': outputs['world_verts'][img_ind],
                'world_cam_rots': outputs['world_cam_rots'][img_ind],
                'world_cams': outputs['world_cams'][img_ind],
            }



    # set the ground to be the minimum y value at the first frame
    ground_y = min([data_frames[0][track_id]['world_verts'][..., 1].min() for track_id in data_frames[0]])

    # trick scaling
    # cam_origin_world[..., 2] = cam_origin_world[..., 2] * 0.3

    timesteps = len(used_org_inds)

    # setup viser server
    server = viser.ViserServer()
    server.scene.world_axes.visible = True
    server.scene.add_grid("ground", width=10, height=10, cell_size=0.5, plane="xz")

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        print("new client!")

        client.camera.position = onp.array([1.14120013, 0.60690449, 5.17581808]) # onp.array([-1, 4, 13])
        client.camera.wxyz = onp.array([-1.75483266e-01,  9.83732196e-01 , 4.88596244e-04, 3.84233121e-02])
            
        # # This will run whenever we get a new camera!
        # @client.camera.on_update
        # def _(_: viser.CameraHandle) -> None:
        #     print(f"New camera on client {client.client_id}!")
        #     print(f"Camera pose for client {id}")
        #     print(f"\tfov: {client.camera.fov}")
        #     print(f"\taspect: {client.camera.aspect}")
        #     print(f"\tlast update: {client.camera.update_timestamp}")
        #     print(f"\twxyz: {client.camera.wxyz}")
        #     print(f"\tposition: {client.camera.position}")
        #     print(f"\tlookat: {client.camera.look_at}")
            
        # Show the client ID in the GUI.
        gui_info = client.gui.add_text("Client ID", initial_value=str(client.client_id))
        gui_info.disabled = True
    
    frame_nodes: list[viser.FrameHandle] = []
    for t in range(timesteps):
        # Add base frame.
        frame_nodes.append(server.scene.add_frame(f"/t{t}", show_axes=False))

        for track_id in data_frames[t].keys():
            vertices = onp.array(data_frames[t][track_id]['world_verts']) 
            # vertices[..., 1] = vertices[..., 1] - ground_y
            vertices = vertices - ground_y # intentional

            server.scene.add_mesh_simple(
                f"/t{t}/mesh{track_id}",
                vertices=vertices,
                faces=onp.array(smpl_faces),
                flat_shading=False,
                wireframe=False,
                color=get_color_for_sid(track_id)
            )
            
            cam_axes_matrix = data_frames[t][track_id]['world_cam_rots']
            cam_axes_quat = R.from_matrix(cam_axes_matrix).as_quat() # xyzw
            cam_axes_quat = cam_axes_quat[[3,0,1,2]] # wxyz

            cam_origin_world = data_frames[t][track_id]['world_cams']
            # cam_origin_world[..., 1] = cam_origin_world[..., 1] - ground_y
            cam_origin_world = cam_origin_world - ground_y # intentional

            server.scene.add_frame(
                f"/t{t}/cam{track_id}",
                wxyz=cam_axes_quat,
                position=cam_origin_world,
                show_axes=True,
                axes_length=0.5,
                axes_radius=0.04,
            )
            

    # Add playback UI.
    with server.gui.add_folder("Playback"):
        gui_timestep = server.gui.add_slider(
            "Timestep",
            min=0,
            max=timesteps - 1,
            step=1,
            initial_value=0,
            disabled=True,
        )
        gui_next_frame = server.gui.add_button("Next Frame", disabled=True)
        gui_prev_frame = server.gui.add_button("Prev Frame", disabled=True)
        gui_playing = server.gui.add_checkbox("Playing", True)
        gui_framerate = server.gui.add_slider(
            "FPS", min=1, max=60, step=0.1, initial_value=15
        )
        gui_framerate_options = server.gui.add_button_group(
            "FPS options", ("10", "20", "30", "60")
        )

    # Frame step buttons.
    @gui_next_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value + 1) % timesteps

    @gui_prev_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value - 1) % timesteps

    # Disable frame controls when we're playing.
    @gui_playing.on_update
    def _(_) -> None:
        gui_timestep.disabled = gui_playing.value
        gui_next_frame.disabled = gui_playing.value
        gui_prev_frame.disabled = gui_playing.value

    # Set the framerate when we click one of the options.
    @gui_framerate_options.on_click
    def _(_) -> None:
        gui_framerate.value = int(gui_framerate_options.value)

    prev_timestep = gui_timestep.value

    render_button = server.gui.add_button("Render motion", disabled=False)
    recording = False
    @render_button.on_click
    def _(event: viser.GuiEvent) -> None:
        nonlocal recording
     
        client = event.client
        if not recording:
            recording = True
            gui_playing.value = False
            gui_timestep.disabled = gui_playing.value
            gui_next_frame.disabled = gui_playing.value
            gui_prev_frame.disabled = gui_playing.value
            gui_framerate.disabled = False
            
            # images = []
            writer = imageio.get_writer(
                'output.mp4', 
                fps=gui_framerate.value, mode='I', format='FFMPEG', macro_block_size=1
            )
            while True:
                if recording:
                    gui_timestep.value = (gui_timestep.value + 1) % timesteps
                    # images.append(client.camera.get_render(height=720, width=1280))
                    img = client.camera.get_render(height=480, width=720)
                    writer.append_data(img)
                    print('recording...')
                else:
                    print("Recording stopped")
                    gui_framerate.disabled = True
                    writer.close()
                    break
        else:
            recording = False
        
    # Toggle frame visibility when the timestep slider changes.
    @gui_timestep.on_update
    def _(_) -> None:
        nonlocal prev_timestep
        current_timestep = gui_timestep.value
        with server.atomic():
            frame_nodes[current_timestep].visible = True
            frame_nodes[prev_timestep].visible = False
        prev_timestep = current_timestep
        server.flush()  # Optional!

    # Playback update loop.
    prev_timestep = gui_timestep.value
    while True:
        if gui_playing.value:
            gui_timestep.value = (gui_timestep.value + 1) % timesteps

        time.sleep(1.0 / gui_framerate.value)

if __name__ == "__main__":
    tyro.cli(main)