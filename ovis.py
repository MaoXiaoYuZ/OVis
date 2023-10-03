import numpy as np
import socketio
import pickle
import colorsys
from scipy.spatial.transform import Rotation as R

try:
    import torch
except Exception:
    torch_available = False
else:
    torch_available = True

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

SEVER_IP = '172.18.70.38'

def oconnect(ip=None):
    global client
    client = socketio.Client()
    if ip is None:
        ip = SEVER_IP
    print(f'正在连接客户端：{ip}')
    try:
        client.connect(f'http://{ip}:5666')
        print('已连接！')
    except Exception:
        print('连接失败！')

oconnect()

def opc(*args):
    global client
    msg = {'points': None, 'trans': None, 'colors': None, 'id': None}
    for arg in args:
        if isinstance(arg, str):
            assert msg['id'] is None, '参数中只能有一个字符串变量用于指示pointcloud的id!'
            msg['id'] = arg
        elif isinstance(arg, float):
            colors = colorsys.hsv_to_rgb(arg, 1, 1)
            assert msg['colors'] is None, '参数中只能有一个float变量用于指示pointcloud的colors!'
            msg['colors'] = colors
        else:
            if torch_available and torch.is_tensor(arg):
                arg = arg.data.cpu().numpy()
            arg = np.asarray(arg, dtype=np.float32)
            if arg.size == 3:
                trans = arg.reshape(1, 3)
                assert msg['trans'] is None, '参数中只能有一个长度为3的数组类型用于指示pointcloud的trans!'
                msg['trans'] = trans
            elif arg.size == 4:
                assert msg['colors'] is None, '参数中只能有一个长度为4的数组类型用于指示pointcloud的colors!'
                if np.any(arg > 1):
                    arg = arg / 255
                msg['colors'] = arg.reshape(4)[:3].tolist()
            else:
                assert len(arg.shape) == 2 and arg.shape[1] == 3, '参数points的shape应为(*, 3)'
                assert msg['points'] is None, '参数中只能有一个shape为(*, 3)数组类型用于指示pointcloud的points!'
                msg['points'] = arg

    if msg['id'] is None:
        print('由于未指定id, opc使用默认的id:pc!')
        msg['id'] = 'pc'

    assert msg['points'] is not None, '必须有一个参数指定points!'

    if msg['trans'] is not None:
        msg['points'] = msg['points'] + msg['trans']

    points = msg['points']
    if torch_available:
        points = torch.from_numpy(points).unique(dim=0).numpy()
    points = pickle.dumps(points)

    client.emit('add_pc', (msg['id'], points, msg['colors']))

def oshapshot(dirname):
    global client
    client.emit('shapshot', (dirname, ))

def opc_pair(*args):
    global client
    msg = {'points': None, 'points2': None, 'lines': None, 'colors': None, 'id': None}
    for arg in args:
        if isinstance(arg, str):
            assert msg['id'] is None, '参数中只能有一个字符串变量用于指示pointcloud的id!'
            msg['id'] = arg
        elif isinstance(arg, float):
            colors = colorsys.hsv_to_rgb(arg, 1, 1)
            assert msg['colors'] is None, '参数中只能有一个float变量用于指示pointcloud的colors!'
            msg['colors'] = colors
        else:
            if torch_available and torch.is_tensor(arg):
                arg = arg.data.cpu().numpy()
            arg = np.asarray(arg, dtype=np.float32)
            if len(arg.shape) == 2 and arg.shape[1] == 2:
                assert msg['lines'] is None, '参数中只能有一个shape为(*, 2)数组类型用于指示pointcloud的lines!'
                msg['lines'] = arg
            else:
                assert len(arg.shape) == 2 and arg.shape[1] == 3, '参数points的shape应为(*, 3)'
                if msg['points'] is None:
                    msg['points'] = arg
                    print('pc1 shape:', arg.shape)
                else:
                    assert msg['points2'] is None, '参数中只能有两个shape为(*, 3)数组类型用于指示pointcloud的points!'
                    msg['points2'] = arg
                    print('pc2 shape:', arg.shape)

    if msg['id'] is None:
        print('由于未指定id, opc使用默认的id:pc_pair!')
        msg['id'] = 'pc_pair'

    assert msg['points'] is not None and msg['points2'] is not None, '必须有两个参数指定points!'
    if len(msg['points']) != len(msg['points2']):
        assert msg['lines'] is not None, '未指定lines!'

    points = np.vstack((msg['points'], msg['points2']))
    points = pickle.dumps(points)
    if msg['lines'] is None:
        lines = np.stack((np.arange(len(msg['points'])), np.arange(len(msg['points'])) + len(msg['points'])), axis=1)
    else:
        lines = msg['lines']
        lines[:, 1] += len(msg['points'])
    client.emit('add_line_set', (msg['id'], points, lines.tolist(), msg['colors']))


def osmpl(*args):
    global client
    msg = {'pose': None, 'beta': None, 'trans': None, 'colors': None, 'id': None}
    for arg in args:
        if isinstance(arg, str):
            assert msg['id'] is None, '参数中只能有一个字符串变量用于指示pointcloud的id!'
            msg['id'] = arg
        elif isinstance(arg, float):
            colors = colorsys.hsv_to_rgb(arg, 1, 1)
            assert msg['colors'] is None, '参数中只能有一个float变量用于指示pointcloud的colors!'
            msg['colors'] = colors
        else:
            if torch_available and torch.is_tensor(arg):
                arg = arg.data.cpu().numpy()
            arg = np.asarray(arg, dtype=np.float32)
            if arg.size == 3:
                trans = arg.flatten().tolist()
                assert msg['trans'] is None, '参数中只能有一个长度为3的数组类型用于指示smpl mesh的trans!'
                msg['trans'] = trans
            elif arg.size == 4:
                assert msg['colors'] is None, '参数中只能有一个长度为4的数组类型用于指示pointcloud的colors!'
                if np.any(arg > 1):
                    arg = arg / 255
                msg['colors'] = arg.reshape(4)[:3].tolist()
            elif arg.size == 10:
                beta = arg.flatten().tolist()
                assert msg['beta'] is None, '参数中只能有一个长度为10的数组类型用于指示smpl mesh的beta!'
                msg['beta'] = beta
            elif arg.size == 72:
                assert msg['pose'] is None, '参数中只能有一个长度为72数组类型用于指示smpl mesh的pose!'
                msg['pose'] = arg.flatten().tolist()
            elif arg.size == 24*3*3:
                assert msg['pose'] is None, '参数中只能有一个长度为24*3*3数组类型用于指示smpl mesh的rotmats!'
                msg['pose'] = (R.from_matrix(arg.reshape(-1, 3, 3))).as_rotvec().flatten().tolist()
            else:
                assert False, '无法解析参数：' + arg

    if msg['id'] is None:
        print('由于未指定id, osmpl使用默认的id:human_mesh!')
        msg['id'] = 'smpl_mesh'

    assert msg['pose'] is not None, '必须有一个参数指定pose!'

    client.emit('add_smpl_mesh', (msg['id'], msg['pose'], msg['beta'], msg['trans'], msg['colors']))


def owait(delay=0, timeout=5):
    import time
    t1 = time.time()
    client.call('empty_event', ('', ), timeout=timeout)
    delay -= time.time() - t1
    if delay > 0:
        time.sleep(delay)


def oclear(geometry_name=None):
    global client
    if geometry_name is None:
        client.emit('rm_all')
    else:
        client.emit('rm_all', (geometry_name, ))

if __name__ == '__main__':
    #opc('pc', np.random.randn(1000, 3), (0, 1, 0), 0.1)
    oconnect('127.0.0.1')
    opc_pair(np.random.rand(34, 3), np.random.rand(40, 3), 0.1, np.array([[0, 1], [3, 2]]))
    # opc(np.zeros((200, 3)), 'pc1')
    # osmpl(np.zeros(72, ))
    owait()


