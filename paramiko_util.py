import numpy as np
import paramiko

from config import RemoteIP, RemoteUser, RemotePassword, RemotePort

def client_server(username=RemoteUser, hostname=RemoteIP, port=RemotePort):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname, port, username, compress=True, password=RemotePassword, timeout=5)
    return client

client = client_server()

def list_dir_remote(client, folder):
    stdin, stdout, stderr = client.exec_command('ls ' + folder)
    res_list = stdout.readlines()
    return [i.strip() for i in res_list]


def read_pcd_from_server(filepath):
    from pypcd import pypcd
    sftp_client = client.open_sftp()
    remote_file = sftp_client.open(filepath, mode='rb')  # 文件路径

    try:
        pc_pcd = pypcd.PointCloud.from_fileobj(remote_file)
        pc = np.zeros((pc_pcd.pc_data.shape[0], 3))
        pc[:, 0] = pc_pcd.pc_data['x']
        pc[:, 1] = pc_pcd.pc_data['y']
        pc[:, 2] = pc_pcd.pc_data['z']
        if pc_pcd.fields[-1] == 'rgb':
            append = pypcd.decode_rgb_from_pcl(pc_pcd.pc_data['rgb']) / 255
        else:
            append = pc_pcd.pc_data[pc_pcd.fields[-1]].reshape(-1, 1)

        return np.concatenate((pc, append), axis=1)
    except Exception as e:
        print(f"Load {filepath} error")
    finally:
        remote_file.close()

def read_img_from_server(filepath):
    from PIL import Image
    sftp_client = client.open_sftp()
    with sftp_client.open(filepath, mode='rb') as f:
        return np.asarray(Image.open(f))


if __name__ == '__main__':
    c = client_server()
