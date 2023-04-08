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

SEVER_IP = '1.1.1.1'

def oconnect():
    global client
    client = socketio.Client()
    ip = SEVER_IP
    print(f'正在连接客户端：{ip}')
    client.connect(f'http://{ip}:5666')
    print('已连接！')

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


def oclear():
    global client
    client.emit('rm_all')

if __name__ == '__main__':
    #opc('pc', np.random.randn(1000, 3), (0, 1, 0), 0.1)
    opc(np.zeros((200, 3)), 'pc1')
    osmpl(np.zeros(72, ))
    owait()


