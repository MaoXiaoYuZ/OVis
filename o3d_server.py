# Define the path to the text file
import sys
import threading
import os
from typing import List, Tuple


def get_extract_ip():
    import socket
    st = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        st.connect(('10.255.255.255', 1))
        IP = st.getsockname()[0]
    except Exception:
        raise Exception('未知错误，无法自动获取公网ip！')
    finally:
        st.close()
    return IP

eip = get_extract_ip()
print('本机公网ip:', eip)


import colorsys
import numpy as np
import open3d as o3d

vis = o3d.visualization.Visualizer()
vis.create_window(width=1024, height=1024)


opt = vis.get_render_option()
# opt.background_color = np.asarray([43., 43., 43.]) / 255.
opt.background_color = np.asarray([255., 255., 255.]) / 255.
# vis.get_render_option().point_size = 15.



id_to_geometry = {}

def add_geometry(id, geometry, reset_bounding_box=True):
    if id in id_to_geometry:
        vis.update_geometry(geometry)
    else:
        vis.add_geometry(geometry, reset_bounding_box=reset_bounding_box)
        id_to_geometry[id] = geometry


def add_pc(sid, id, points, colors=None):
    print(f'Recive pc:{len(points)}, id:{id} from sid:{sid}')
    if id in id_to_geometry:
        pointcloud = id_to_geometry[id]
    else:
        pointcloud = o3d.geometry.PointCloud()

    pointcloud.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        colors = np.array(colors)
        if colors.shape == (1, 3) or colors.shape == (3, ):
            colors = colors.reshape(1, 3).repeat(len(points), axis=0)
        pointcloud.colors = o3d.utility.Vector3dVector(colors)

    add_geometry(id, pointcloud)

from smpl import SMPL
import torch
smpl = SMPL()

from config import SMPL_FILE
import pickle
with open(SMPL_FILE, 'rb') as f:
    smpl_model = pickle.load(f, encoding='iso-8859-1')
    face_index = smpl_model['f'].astype(int)

def add_smpl_pc(sid, id, pose, beta=None, trans=None):
    if beta is None:
        beta = torch.zeros((10,))
    if trans is None:
        trans = torch.zeros((1, 3))
    v = smpl(torch.from_numpy(np.array(pose)).float().view(1, 72),
             torch.from_numpy(np.array(beta)).float().view(1, 10)).squeeze()
    v += torch.from_numpy(np.array(trans)).float().view(1, 3)
    add_pc(sid, id, v.tolist(), None)

def add_coordinate(sid, id, origin, size):
    axis_pcd = id_to_geometry[id] if id in id_to_geometry else \
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=origin)

    add_geometry(id, axis_pcd, reset_bounding_box=True)

def add_smpl_mesh(sid, id, pose, beta=None, trans=None, colors=None):
    if beta is None:
        beta = torch.zeros((10,))
    if trans is None:
        trans = torch.zeros((1, 3))
    from time import time
    t0 = time()
    v = smpl(torch.from_numpy(np.array(pose)).float().view(1, 72),
             torch.from_numpy(np.array(beta)).float().view(1, 10)).squeeze()
    v += torch.from_numpy(np.array(trans)).float().view(1, 3)

    if id in id_to_geometry:
        m = id_to_geometry[id]
    else:
        m = o3d.geometry.TriangleMesh()

    v = v.numpy().astype('float64') #一行代码提速1000倍
    m.vertices = o3d.utility.Vector3dVector(v)
    m.triangles = o3d.utility.Vector3iVector(face_index)
    m.compute_vertex_normals()
    if colors is not None:
        if isinstance(colors, (list, tuple)):
            if colors[0] > 1:
                colors = [e/255 for e in colors]
        elif isinstance(colors, (int, float)):
            colors = colorsys.hsv_to_rgb(colors, 1, 1)
        else:
            colors = np.random.rand(3)
        m.paint_uniform_color(colors)
    add_geometry(id, m)


def add_line_set(sid, id, points, lines, color=None):
    points = pickle.loads(points) if isinstance(points, bytes) else points

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.asarray(points)),
        lines=o3d.utility.Vector2iVector(np.array(lines))
    )
    if id in id_to_geometry:
        vis.remove_geometry(id_to_geometry[id], reset_bounding_box=False)
        del id_to_geometry[id]

    if color is not None:
        line_set.paint_uniform_color(color)

    add_geometry(id, line_set)

def rm_all(sid, geometry_name=None):
    if geometry_name is None:
        vis.clear_geometries()
        id_to_geometry.clear()
        add_coordinate(None, 'default_coordinate', [0, 0, 0], 1)
    else:
        vis.remove_geometry(id_to_geometry[geometry_name], reset_bounding_box=False)
        del id_to_geometry[geometry_name]

def shapshot(sid, dirname):
    import time
    if os.path.exists(dirname):
        #read all geometry from the dirname and add them to vis
        for f in os.listdir(dirname):
            if f.endswith('.ply'):
                if f.startswith('pc_'):
                    id = f[3:-4]
                    pc = o3d.io.read_point_cloud(os.path.join(dirname, f))
                    add_geometry(id, pc)
                elif f.startswith('mesh_'):
                    id = f[5:-4]
                    mesh = o3d.io.read_triangle_mesh(os.path.join(dirname, f))
                    add_geometry(id, mesh)
                elif f.startswith('lineset_'):
                    id = f[8:-4]
                    lineset = o3d.io.read_line_set(os.path.join(dirname, f))
                    add_geometry(id, lineset)
        print('snapshot loaded from', dirname)
        return

    dirname = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + '_' + dirname
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    vis.capture_screen_image(os.path.join(dirname, 'screenshot.png'))
    for id, geometry in id_to_geometry.items():
        if isinstance(geometry, o3d.geometry.PointCloud):
            o3d.io.write_point_cloud(os.path.join(dirname, f'pc_{id}.ply'), geometry)
        elif isinstance(geometry, o3d.geometry.TriangleMesh):
            o3d.io.write_triangle_mesh(os.path.join(dirname, f'mesh_{id}.ply'), geometry)
        elif isinstance(geometry, o3d.geometry.LineSet):
            o3d.io.write_line_set(os.path.join(dirname, f'lineset_{id}.ply'), geometry)
    print('snapshot saved to', dirname)

def affine(X, matrix):
    res = np.concatenate((X, np.ones((*X.shape[:-1], 1))), axis=-1).T
    res = np.dot(matrix, res).T
    return res[..., :-1]

def add_smpl_mesh_w_rt(sid, id, pose, rt):
    beta = torch.zeros((10,))
    v = smpl(torch.from_numpy(np.array(pose)).float().view(1, 72),
             torch.from_numpy(np.array(beta)).float().view(1, 10)).squeeze()

    v = affine(v, np.array(rt).reshape(4, 4))

    if id in id_to_geometry:
        m = id_to_geometry[id]
    else:
        m = o3d.geometry.TriangleMesh()

    m.vertices = o3d.utility.Vector3dVector(v)
    m.triangles = o3d.utility.Vector3iVector(face_index)
    m.compute_vertex_normals()
    add_geometry(id, m)

def draw_bbox_3d(sid, id, 
                       bbox_3d,
                       bbox_color: List[float] = (0, 1, 0),
                       rot_axis: int = 2,
                       center_mode: str = 'lidar_bottom') -> None:
        """Draw bbox on visualizer and change the color of points inside
        bbox3d.

        Args:
            bbox_3d (:obj:`BaseInstance3DBoxes`): 3D bbox
                (x, y, z, x_size, y_size, z_size, yaw) to visualize.
            bbox_color (Tuple[float]): The color of 3D bboxes.
                Defaults to (0, 1, 0).
            points_in_box_color (Tuple[float]): The color of points inside 3D
                bboxes. Defaults to (1, 0, 0).
            rot_axis (int): Rotation axis of 3D bboxes. Defaults to 2.
            center_mode (str): Indicates the center of bbox is bottom center or
                gravity center. Available mode
                ['lidar_bottom', 'camera_bottom']. Defaults to 'lidar_bottom'.
            mode (str): Indicates the type of input points, available mode
                ['xyz', 'xyzrgb']. Defaults to 'xyz'.
        """

        # convert bboxes to numpy dtype
        bbox_3d = torch.from_numpy(np.array(bbox_3d)).flatten()

        assert bbox_3d.numel() == 7, 'ERROR:The size of bbox_3d must be 7!'

        # in_box_color = np.array(points_in_box_color)

        # for i in range(len(bbox_3d)):
        center = bbox_3d[0:3]
        dim = bbox_3d[3:6]
        yaw = np.zeros(3)
        yaw[rot_axis] = bbox_3d[6]
        rot_mat = o3d.geometry.get_rotation_matrix_from_xyz(yaw)

        if center_mode == 'lidar_bottom':
            # bottom center to gravity center
            center[rot_axis] += dim[rot_axis] / 2
        elif center_mode == 'camera_bottom':
            # bottom center to gravity center
            center[rot_axis] -= dim[rot_axis] / 2
        box3d = o3d.geometry.OrientedBoundingBox(center, rot_mat, dim)

        line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)
        line_set.paint_uniform_color(np.array(bbox_color))
        # draw bboxes on visualizer

        if id in id_to_geometry:
            vis.remove_geometry(id_to_geometry[id], reset_bounding_box=False)
            del id_to_geometry[id]

        add_geometry(id, line_set, reset_bounding_box=False)

        #     # change the color of points which are in box
        #     if self.pcd is not None and mode == 'xyz':
        #         indices = box3d.get_point_indices_within_bounding_box(
        #             self.pcd.points)
        #         self.points_colors[indices] = np.array(bbox_color[i]) / 255.

        # # update points colors
        # if self.pcd is not None:
        #     self.pcd.colors = o3d.utility.Vector3dVector(self.points_colors)
        #     self.o3d_vis.update_geometry(self.pcd)


# client.call("draw_bbox_3d", ('test', [ 14.7600,  -1.1046,  -1.5370,   3.7431,   1.5425,   1.4897,  -0.3121]))


sys.path.append('./ogrpc') 
from concurrent import futures
import grpc
import ogrpc_pb2
import ogrpc_pb2_grpc

import queue
global_events_quene = queue.Queue() 

class OService(ogrpc_pb2_grpc.OServiceServicer):
    def Ask(self, request, context):
        global global_events_quene
        kwargs = pickle.loads(request.pkl)
        global_events_quene.put(kwargs)
        return ogrpc_pb2.OReply(pkl=pickle.dumps(''))

def serve():
    port = "50051"
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    ogrpc_pb2_grpc.add_OServiceServicer_to_server(OService(), server)
    server.add_insecure_port("[::]:" + port)
    server.start()
    print("Server started, listening on " + port)
    server.wait_for_termination()


server_thread = threading.Thread(target=serve)  
server_thread.start()  


def vis_update():
    from time import time, sleep

    while(True):
        t0 = time()
        while not global_events_quene.empty():
            kwargs = global_events_quene.get()
            try:
                func = eval(kwargs.pop('func'))
                func(**kwargs)
            except Exception as e:
                import traceback
                traceback.print_exc()
            
            break #一次只执行一个

        vis.poll_events()
        vis.update_renderer()
        
        duration_t = time() - t0
        delta_t = 0.01 - duration_t
        if delta_t > 0:
            sleep(delta_t)

vis_update()