# Define the path to the text file
from collections import defaultdict
import pickle
import sys
import threading
import os
from typing import List, Tuple
from time import time, sleep, localtime, strftime

from multiprocessing import Process, Queue, Event


def get_extract_ip():
    import socket
    try:
        st = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
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


vis = None
id_to_geometry = {}

def add_geometry(id, geometry, reset_bounding_box=True):
    if id in id_to_geometry:
        vis.update_geometry(geometry)
    else:
        vis.add_geometry(geometry, reset_bounding_box=False)
        id_to_geometry[id] = geometry

def focus(sid):
    vis.reset_view_point(True)

def add_pc(sid, id, points, colors=None):
    print(f'Recive pc:{len(points)}, id:{id} from sid:{sid}')
    if id in id_to_geometry:
        pointcloud = id_to_geometry[id]
    else:
        pointcloud = o3d.geometry.PointCloud()

    pointcloud.points = o3d.utility.Vector3dVector(np.asarray(points).astype('float64'))
    if colors is not None:
        colors = np.array(colors)
        if colors.shape == (1, 3) or colors.shape == (3, ):
            colors = colors.reshape(1, 3).repeat(len(points), axis=0)
        pointcloud.colors = o3d.utility.Vector3dVector(colors)

    add_geometry(id, pointcloud)


import torch
smpl, face_index = None, None

def add_coordinate(sid, id, origin, size):
    axis_pcd = id_to_geometry[id] if id in id_to_geometry else \
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=origin)

    add_geometry(id, axis_pcd, reset_bounding_box=False)

def add_smpl_mesh(sid, id, pose=None, beta=None, trans=None, colors=None, rt=None, v=None):
    if beta is None:
        beta = torch.zeros((10,))
    if trans is None:
        trans = torch.zeros((1, 3))
    if v is None:
        v = smpl(torch.from_numpy(np.array(pose)).float().view(1, 72),
                torch.from_numpy(np.array(beta)).float().view(1, 10)).squeeze()
        v += torch.from_numpy(np.array(trans)).float().view(1, 3)

        v = v.numpy()

    if rt is not None:
        v = affine(v, np.array(rt, dtype='float16').reshape(4, 4))

    if id in id_to_geometry:
        m = id_to_geometry[id]
    else:
        m = o3d.geometry.TriangleMesh()

    v = v.astype('float64') #一行代码提速1000倍
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

    dirname = strftime("%Y-%m-%d-%H-%M-%S", localtime()) + '_' + dirname
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

class OService(ogrpc_pb2_grpc.OServiceServicer):
    def __init__(self, q):
        self.q = q

    def Ask(self, request, context):
        self.q.put(request.pkl)
        return ogrpc_pb2.OReply(pkl=pickle.dumps(''))

    def Sync(self, request, context):
        while not self.q.empty():
            sleep(0.001)
        return ogrpc_pb2.OReply(pkl=pickle.dumps(''))

def serve(q):
    port = "50051"
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    ogrpc_pb2_grpc.add_OServiceServicer_to_server(OService(q), server)
    server.add_insecure_port("[::]:" + port)
    server.start()
    print("Server started, listening on " + port)
    server.wait_for_termination()


# @profile
def vis_update():
    events_q = Queue()  
    grpc_proc = Process(target=serve, args=(events_q,))  
    print("Start gRPC Server... ")
    grpc_proc.start()  
    
    global vis
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1024, height=1024)

    opt = vis.get_render_option()
    # opt.background_color = np.asarray([43., 43., 43.]) / 255.
    opt.background_color = np.asarray([255., 255., 255.]) / 255.
    # vis.get_render_option().point_size = 15.

    global smpl, face_index
    from smpl import SMPL
    smpl = SMPL()

    from config import SMPL_FILE
    import pickle
    with open(SMPL_FILE, 'rb') as f:
        smpl_model = pickle.load(f, encoding='iso-8859-1')
        face_index = smpl_model['f'].astype(int)


    add_coordinate(None, 'default_coordinate', [0, 0, 0], 1)
    focus(None)

    frame_i = 0
    running_flag = True
    while running_flag:
        frame_i += 1
        
        t0 = time()
        func_call = defaultdict(list)
        while not events_q.empty():
            pkl = events_q.get()
            kwargs = pickle.loads(pkl)
            func_call[kwargs.pop('func')].append(kwargs)
        
        # func_call = {'add_smpl_mesh':[{'pose':np.zeros(72,), 'id': 'smpl1'}, {'pose':np.zeros(72,), 'id': 'smpl1'}]}

        if 'add_smpl_mesh' in func_call:
            batch_pose = [kwargs['pose'] for kwargs in func_call['add_smpl_mesh'] if 'v' not in kwargs]
            batch_beta = [kwargs['beta'] if 'beta' in kwargs and kwargs['beta'] else np.zeros((10,))  for kwargs in func_call['add_smpl_mesh'] if 'v' not in kwargs]
            batch_trans = [kwargs['trans'] if 'trans' in kwargs and kwargs['trans'] else np.zeros((3,))  for kwargs in func_call['add_smpl_mesh'] if 'v' not in kwargs]
            batch_v = smpl(torch.from_numpy(np.array(batch_pose)).float().view(-1, 72),
                torch.from_numpy(np.array(batch_beta)).float().view(-1, 10))
            batch_v += torch.from_numpy(np.array(batch_trans)).float().view(-1, 1, 3)

            batch_v = batch_v.numpy()

            for kwargs, v in zip([e for e in func_call['add_smpl_mesh'] if 'v' not in e], batch_v):
                kwargs['v'] = v

        for func_name, kwargs_list in func_call.items():
            for kwargs in kwargs_list:
                try:
                    func = eval(func_name)
                    func(**kwargs)
                except Exception as e:
                    import traceback
                    traceback.print_exc()

        running_flag = vis.poll_events()
        vis.update_renderer()
        
        duration_t = time() - t0
        if duration_t > 0 and 1 / duration_t < 50:
            print(f"{1/duration_t:.2f}HZ({duration_t:.3f}s)")
        delta_t = 0.01 - duration_t
        if delta_t > 0:
            sleep(delta_t)
    
    
    print("Terminating child process...")  
    grpc_proc.terminate()  # 强制结束子进程  
    grpc_proc.join()  # 等待子进程退出


if __name__ == '__main__':
    vis_update()