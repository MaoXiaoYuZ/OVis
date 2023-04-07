# Define the path to the text file
def update_server_ip_in_file(file_path, new_ip):
    import re
    # Open the file for reading and store its contents as a string
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    match = re.search(r"SEVER_IP\s*=\s*'([\d\.]+)'", content)

    assert match is not None, '未知错误！'

    # Replace the IP address with a new one (e.g. "192.168.0.1")
    new_content = content[:match.start(1)] + new_ip + content[match.end(1):]

    # Open the file for writing and write the modified content back to it
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(new_content)

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

try:
    from paramiko_util import client
    from config import RemoteIP, RemotePythonImportPath
    import os
    update_server_ip_in_file('ovis.py', eip)
    sftp_client = client.open_sftp()
    remote_ovis_py_path = os.path.join(RemotePythonImportPath, 'ovis.py').replace('\\', '/')
    print(f'update the SEVER_IP in ovis.py according to config:{eip}')
    print(f'upload the ovis.py to the remote server:{remote_ovis_py_path}')
    sftp_client.put('ovis.py', remote_ovis_py_path)
except Exception as e:
    print(e)
    print('fail to connect the remote server!')


import colorsys
import socketio

#sio = socketio.AsyncServer(async_mode='tornado', logger=True, engineio_logger=True, ping_timeout=60)
sio = socketio.AsyncServer(async_mode='tornado')

import open3d as o3d
import asyncio
vis = o3d.visualization.Visualizer()
vis.create_window(width=1024, height=1024)

import numpy as np

opt = vis.get_render_option()
opt.background_color = np.asarray([43., 43., 43.]) / 255.


@sio.on('*')
def catch_all(event, pid, data):
    print(event, pid, data)

@sio.event
def my_event(sid, data):
    print(sid, data)

@sio.on('my custom event')
def another_event(sid, data):
    print(sid, data)

@sio.event
def connect(sid, environ, auth):
    print('connect ', sid)

@sio.event
def disconnect(sid):
    print('disconnect ', sid)



id_to_geometry = {}

def add_geometry(id, geometry):
    if id in id_to_geometry:
        vis.update_geometry(geometry)
    else:

        vis.add_geometry(geometry, reset_bounding_box=True)
        id_to_geometry[id] = geometry


@sio.event
def add_pc(sid, id, points, colors=None):
    points = pickle.loads(points) if isinstance(points, bytes) else points
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

#shan = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(o3d.io.read_point_cloud('C:\\Users\\maoqh\\Documents\\temp\\shan.pcd'), 0.1)
#shan = o3d.io.read_triangle_mesh("C:\\Users\\maoqh\\Documents\\temp\\shan.obj")

@sio.event
def add_obj(sid, id, rt, radius, colors=None):
    if id in id_to_geometry:
        m = id_to_geometry[id]
    else:
        m = o3d.geometry.TriangleMesh()

    m.vertices = shan.vertices
    m.triangles = shan.triangles

    m.transform(rt)
    if colors is not None:
        m.paint_uniform_color(colors)
    add_geometry(id, m)

#add_obj(None, 'obj', np.identity(4), 0)

from smpl import SMPL
import torch
smpl = SMPL()

from config import SMPL_FILE
import pickle
with open(SMPL_FILE, 'rb') as f:
    smpl_model = pickle.load(f, encoding='iso-8859-1')
    face_index = smpl_model['f'].astype(int)

@sio.event
def add_smpl_pc(sid, id, pose, beta=None, trans=None):
    if beta is None:
        beta = torch.zeros((10,))
    if trans is None:
        trans = torch.zeros((1, 3))
    v = smpl(torch.from_numpy(np.array(pose)).float().view(1, 72),
             torch.from_numpy(np.array(beta)).float().view(1, 10)).squeeze()
    v += torch.from_numpy(np.array(trans)).float().view(1, 3)
    add_pc(sid, id, v.tolist(), None)

@sio.event
def add_coordinate(sid, id, origin, size):
    axis_pcd = id_to_geometry[id] if id in id_to_geometry else \
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=origin)

    add_geometry(id, axis_pcd)


@sio.event
def add_smpl_mesh(sid, id, pose, beta=None, trans=None, color=None):
    if beta is None:
        beta = torch.zeros((10,))
    if trans is None:
        trans = torch.zeros((1, 3))
    v = smpl(torch.from_numpy(np.array(pose)).float().view(1, 72),
             torch.from_numpy(np.array(beta)).float().view(1, 10)).squeeze()
    v += torch.from_numpy(np.array(trans)).float().view(1, 3)

    if id in id_to_geometry:
        m = id_to_geometry[id]
    else:
        m = o3d.geometry.TriangleMesh()


    m.vertices = o3d.utility.Vector3dVector(v)
    m.triangles = o3d.utility.Vector3iVector(face_index)
    m.compute_vertex_normals()
    if color is not None:
        if isinstance(color, (list, tuple)):
            if color[0] > 1:
                color = [e/255 for e in color]
        elif isinstance(color, (int, float)):
            color = colorsys.hsv_to_rgb(color, 1, 1)
        else:
            color = np.random.rand(3)
        m.paint_uniform_color(color)
    add_geometry(id, m)


@sio.event
def rm_all(sid):
    vis.clear_geometries()
    id_to_geometry.clear()
    add_coordinate(None, 'default_coordinate', [0, 0, 0], 1)

def affine(X, matrix):
    res = np.concatenate((X, np.ones((*X.shape[:-1], 1))), axis=-1).T
    res = np.dot(matrix, res).T
    return res[..., :-1]

@sio.event
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


async def vis_update():
    while(True):
        vis.poll_events()
        vis.update_renderer()
        await asyncio.sleep(0.01)




add_coordinate(None, 'default_coordinate', [0, 0, 0], 1)

import tornado
app = tornado.web.Application(
    [
        (r"/socket.io/", socketio.get_tornado_handler(sio)),
    ],
    # ... other application options
)
app.listen(5666)
tornado.ioloop.IOLoop.current().run_sync(vis_update)