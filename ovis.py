import sys
import numpy as np
import pickle
import colorsys
import os
from scipy.spatial.transform import Rotation as R


import pickle
import numpy as np

#init_ogrpc------------------------------------------------


def init_ogrpc():
    #ogrpc.proto
    ogrpc_proto = \
"""
syntax = "proto3";

service OService {
  rpc Ask (ORequest) returns (OReply) {}
  rpc Sync (ORequest) returns (OReply) {}
}

message ORequest {
  bytes pkl = 1;
}

message OReply {
  bytes pkl = 1;
}
"""
    if os.path.exists('.ovis'):
        try:
            import ogrpc_pb2
            import ogrpc_pb2_grpc
        except Exception:
            pass
        else:
            sys.path.append('.ovis')
            return

    os.makedirs('.ovis', exist_ok=True)

    with open('.ovis/ogrpc.proto', 'w', encoding='utf-8') as file:
        file.write(ogrpc_proto)

    import subprocess  
  
    command = [  
        'python', '-m', 'grpc_tools.protoc',   
        '-I.ovis', '--python_out=.ovis', '--pyi_out=.ovis', '--grpc_python_out=.ovis', '.ovis/ogrpc.proto'  
    ]  
    
    subprocess.run(command, check=True)
    sys.path.append('.ovis')


init_ogrpc()
import grpc
import ogrpc_pb2
import ogrpc_pb2_grpc


#oclient-----------------------------------------

def oconnect(url):
    global channel
    channel = grpc.insecure_channel(url)

# @profile
def oask(request_obj):
    global channel
    stub = ogrpc_pb2_grpc.OServiceStub(channel)
    response = stub.Ask(ogrpc_pb2.ORequest(pkl=pickle.dumps(request_obj)))
    # return np.frombuffer(response.pkl, dtype='float16').reshape(100000, 3)
    return pickle.loads(response.pkl)

def oclose():
    global channel
    if channel:
        channel.close()
        channel = None


def oconnect(url):
    global channel
    channel = grpc.insecure_channel(url)

# @profile
def oask(request_obj):
    global channel
    stub = ogrpc_pb2_grpc.OServiceStub(channel)
    response = stub.Ask(ogrpc_pb2.ORequest(pkl=pickle.dumps(request_obj)))
    # return np.frombuffer(response.pkl, dtype='float16').reshape(100000, 3)
    return pickle.loads(response.pkl)

def oclose():
    global channel
    if channel:
        channel.close()
        channel = None

#wrapper for o3d server-------------------------------------------

try:
    import torch
except Exception:
    torch_available = False
else:
    torch_available = True

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
            arg = np.asarray(arg, dtype=np.float16)
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
    
    msg.pop('trans')

    points = msg['points']
    # if torch_available:
    #     points = torch.from_numpy(points).unique(dim=0).numpy()
    # points = pickle.dumps(points)

    oask({'func':'add_pc', 'sid': '', **msg})

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
    msg = {'pose': None, 'beta': None, 'trans': None, 'colors': None, 'id': None, 'rt': None}
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
            elif arg.size == 16:
                rt = arg.reshape(4, 4).tolist()
                assert msg['rt'] is None, '参数中只能有一个长度为4*4的数组类型用于指示smpl mesh的rt!'
                msg['rt'] = rt
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
    
    oask({'func':'add_smpl_mesh', 'sid': '', **msg})

def owait(delay=0):
    import time
    t1 = time.time()
    global channel
    stub = ogrpc_pb2_grpc.OServiceStub(channel)
    stub.Sync(ogrpc_pb2.ORequest(pkl=pickle.dumps('')))
    delay -= time.time() - t1
    if delay > 0:
        time.sleep(delay)


def oclear(geometry_name=None):
    if geometry_name is None:
        oask({'func':'rm_all', 'sid': ''})
    else:
        oask({'func':'rm_all', 'sid': '', 'geometry_name': geometry_name})

def ofocus():
    oask({'func':'focus', 'sid': ''})

# @profile
def test():
    from time import time
    oconnect("localhost:50051")
    # oconnect("59.77.18.8:50051")
    for i in range(100):
        #opc(np.random.rand(10, 3), 'pc1')
        # oask({'pc': np.random.rand(10, 3)})
        # result = oask([{'pid': i, 'frame': np.random.rand(256, 3).astype('float16')} for i in range(4)])
        # oask({'pc': np.random.rand(1, 3)})
        # owait()
        
        # opc(np.random.rand(100000, 3), 'pc1')
        # osmpl(np.random.rand(72), 'smpl')
        owait()
        osmpl(np.random.rand(72), 'smpl1')
        osmpl(np.random.rand(72), 'smpl2', (2, 0, 0))
        osmpl(np.random.rand(72), 'smpl3', (1, 0, 0))
    oclose()


if __name__ == '__main__':
    test()


