<h1>OVis: Remote Point Cloud and Human Body Mesh Visualization based on Open3D</h1>

OVis is a program that enables remote visualization of point clouds and human body meshes using Open3D. It can be run on a remote server and communicates with the local Open3D visualization program via socketio.

# Environment Configuration (Local)
```angular2html
pip install python-socketio
pip install "python-socketio[client]"
pip install numpy
pip install open3d
pip install paramiko
pip instal scipy
```
Then modify the following fields in config.py:
```angular2html
RemoteIP = '1.2.3.4'
RemoteUser = 'root'
RemotePort = 22
RemotePassword = '123456'
RemotePythonImportPath = '/~/anaconda3/envs/myenv/lib/python3.7'
```


# Environment Configuration (Remote Server)
```angular2html
pip install python-socketio
pip install "python-socketio[client]"
pip install numpy
pip instal scipy
```


# 1.Run o3d_server.py on the local machine
```angular2html
python o3d_server.py
```
This will copy ovis.py to RemotePythonImportPath. 
Make sure this doesn't overwrite any of your existing files. 
A new Open3D visualization window will open, which listens for requests on port 5666 and performs the corresponding visualization operations.


# 2.On the remote/local machine, connect to o3d_server:
```angular2html
from ovis import *
```

# 3.Visualize point clouds
```angular2html
opc(np.random.rand(1000, 3))
opc(np.random.rand(1000, 3), 0.2, (1, 0, 0), 'pc2')
```

![show_pc](./data/show_pc.png)

# 4.Visualize SMPL human bodies
```angular2html
osmpl(np.zeros(72), (1, 0, 0), 0.1)
```

![show_smpl_mesh](./data/show_smpl_mesh.png)

# 5.Create animated point cloud visualizations
```angular2html
for i in range(10):
    opc('pc1', np.random.rand(1000, 3))
    owait(0.1)
```

# 6.Reconnect to o3d_server and clear the visualization
```angular2html
oconnect()
oclear()
```

# 7.import ovis locally
To import ovis locally, manually copy ovis.py to your local Python import path and set SERVER_IP in ovis.py to 127.0.0.1.
