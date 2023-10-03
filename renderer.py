# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018-2021 www.open3d.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------
import os
import traceback
from collections import defaultdict

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import platform
import random
import threading
import time
import numpy as np

from renderer_utils import color_points_by_z, color_by_z

print(f'Current Platfrom System:{platform.system()}')
isMacOS = (platform.system() == "Darwin")

class O3DRenderer:
    MENU_SPHERE = 1
    MENU_RANDOM = 2
    MENU_QUIT = 3
    MENU_SHOT=4

    def __init__(self, win_flag, win_size=(768, 768)):
        self.win_flag = win_flag
        self.win_size = win_size

        if self.win_flag:
            self.init_windows()
        else:
            self.init_renderer()

        #self.add_coordinate('default_coordinate', [0, 0, 0], 1)

        # create sphere point pool
        self.sphere_points_poll = []
        self.geometry_name_to_sphere_point_id = defaultdict(list)
        self.global_sphere_point_increament = 0

        mat = rendering.MaterialRecord()
        mat.shader = "defaultLit"
        for i, p in enumerate(np.zeros((2024, 3))):
            sphere = self.get_sphere_point(p, radius=0.025)
            sphere_point_id = "point" + str(self.global_sphere_point_increament)
            if self.win_flag:
                self.scene.scene.remove_geometry(sphere_point_id)
                self.scene.scene.add_geometry(sphere_point_id, sphere, mat)
                self.scene.scene.show_geometry(sphere_point_id, False)
            self.sphere_points_poll.append(sphere_point_id)
            self.global_sphere_point_increament += 1

        # if self.win_flag:
        #     gui.Application.instance.run()
    
    def init_renderer(self):
        self.render = rendering.OffscreenRenderer(self.win_size[0], self.win_size[1])
        self.render.scene.set_background([1, 1, 1, 1])

        self.render.scene.scene.set_sun_light(
            [0, 0, 1],  # direction
            [1, 1, 1],  # color
            100000)  # intensity
        self.render.scene.scene.enable_sun_light(True)

        self.render.setup_camera(90, [0, 0, 1], [0, 0, 0], [0, -1, 0])

    def init_windows(self):
        gui.Application.instance.initialize()
        self.window = gui.Application.instance.create_window(
            "O3D Renderer", self.win_size[0], self.win_size[1])
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)

        self.scene.scene.set_background([1, 1, 1, 1])
        self.scene.scene.scene.set_sun_light(
            [0, 0, 1],  # direction
            [1, 1, 1],  # color
            100000)  # intensity
        self.scene.scene.scene.enable_sun_light(True)

        # plane = o3d.visualization.rendering.Scene.GroundPlane(o3d.visualization.rendering.Scene.GroundPlane.XY)
        # self.scene.scene.show_ground_plane(True, plane)
        #self.scene.scene.show_skybox(True)

        bbox = o3d.geometry.AxisAlignedBoundingBox([-10, -10, -10],[10, 10, 10])
        self.scene.setup_camera(90, bbox, [0, 0, 0])

        self.window.add_child(self.scene)

        # The menu is global (because the macOS menu is global), so only create
        # it once, no matter how many windows are created
        if gui.Application.instance.menubar is None:
            if isMacOS:
                app_menu = gui.Menu()
                app_menu.add_item("Quit", O3DRenderer.MENU_QUIT)
            debug_menu = gui.Menu()
            debug_menu.add_item("Add PCD", O3DRenderer.MENU_SPHERE)
            debug_menu.add_item("SHOT", O3DRenderer.MENU_SHOT)
            debug_menu.add_item("Add Random Spheres", O3DRenderer.MENU_RANDOM)
            if not isMacOS:
                debug_menu.add_separator()
                debug_menu.add_item("Quit", O3DRenderer.MENU_QUIT)

            menu = gui.Menu()
            if isMacOS:
                # macOS will name the first menu item for the running application
                # (in our case, probably "Python"), regardless of what we call
                # it. This is the application menu, and it is where the
                # About..., Preferences..., and Quit menu items typically go.
                menu.add_menu("Example", app_menu)
                menu.add_menu("Debug", debug_menu)
            else:
                menu.add_menu("Debug", debug_menu)
            gui.Application.instance.menubar = menu

        # The menubar is global, but we need to connect the menu items to the
        # window, so that the window can call the appropriate function when the
        # menu item is activated.
        self.window.set_on_menu_item_activated(O3DRenderer.MENU_SPHERE,
                                               self._on_menu_sphere)
        self.window.set_on_menu_item_activated(O3DRenderer.MENU_RANDOM,
                                               self._on_menu_random)
        self.window.set_on_menu_item_activated(O3DRenderer.MENU_QUIT,
                                               self._on_menu_quit)
        self.window.set_on_menu_item_activated(O3DRenderer.MENU_SHOT,
                                               self._on_menu_shot)

    def get_sphere_point(self, pos, radius=0.5, color=[1, 1, 1]):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius)
        sphere.compute_vertex_normals()
        sphere.translate(pos)
        sphere.paint_uniform_color(color)
        return sphere

    def test_add_points(self):
        pcd = o3d.io.read_point_cloud(r"C:\Users\maoqh\PycharmProjects\OVis\2023-06-01-23-42-29_occlusion\pc_pc30.ply")
        points = np.asarray(pcd.points)
        colors = np.ones((points.shape[0], ))
        try:
            self.add_sphere_pc('pc', points, colors)
        except Exception:
            traceback.print_exc()
    
    def add_sphere_pc(self, geometry_name, points: np.ndarray, colors: np.ndarray=None):
        assert isinstance(points, np.ndarray) and len(points.shape) == 2 and points.shape[1] == 3 and points.shape[0] <= 1024
        assert isinstance(colors, np.ndarray) or colors is None
        center = points.mean(axis=0)
        #camera_pos = center - center * 0.9 / np.linalg.norm(center)
        camera_pos = center - [0, 0, 1]

        if self.win_flag:
            self.scene.look_at(center, camera_pos, [0, -1, 0])
        else:
            self.render.setup_camera(90, center, camera_pos, [0, -1, 0])

        rt = np.identity(4)
        rt[:3, 3] = center

        if colors is None:
            colors = np.concatenate((color_points_by_z(points), np.ones((points.shape[0], 1))), axis=-1)
        else:
            colors = colors.squeeze()
            if len(colors.shape) == 1:
                colors_label = colors.astype('int32')
                colors_label -= colors_label.min()
                colors = np.ones((points.shape[0], 4))
                colors[:, :3] = color_by_z(np.arange(0, colors_label.max()+1, 1))[colors_label]
            elif colors.shape[1] == 3:
                colors = np.concatenate((colors, np.ones((colors.shape[0], 1))), axis=-1)

        mat = rendering.MaterialRecord()
        mat.shader = "defaultLit"

        for i, p in enumerate(points):
            rt[:3, 3] = p
            base_color = [random.random(), random.random(), random.random(), 1.0] if colors is None else list(colors[i])
            assert len(base_color) == 4 and 0. <= base_color[0] <= 1., 'colors格式不对！'
            mat.base_color = base_color

            sphere_point_id = self.sphere_points_poll.pop()
            self.geometry_name_to_sphere_point_id[geometry_name].append(sphere_point_id)

            if self.win_flag:
                self.scene.scene.show_geometry(sphere_point_id, True)
                self.scene.scene.set_geometry_transform(sphere_point_id, rt)
                self.scene.scene.modify_geometry_material(sphere_point_id, mat)
            else:
                self.render.scene.show_geometry(sphere_point_id, True)
                self.render.scene.set_geometry_transform(sphere_point_id, rt)
                self.render.scene.modify_geometry_material(sphere_point_id, mat)

    def save_sphere_point(self, geometry_name, filename, color=(1, 1, 1)):
        from operator import add
        from functools import reduce
        try:
            sphere_pc = reduce(add, [self.get_sphere_point(self.scene.scene.get_geometry_transform(e)[:3, 3], radius=0.025, color=color) for e
                                in self.geometry_name_to_sphere_point_id[geometry_name]])
            o3d.io.write_triangle_mesh(filename, sphere_pc)
        except Exception:
            traceback.print_exc()
            print('save sphere point failed!')

    def remove_geometry(self, geometry_name):
        if geometry_name in self.geometry_name_to_sphere_point_id:
            for sphere_point_id in self.geometry_name_to_sphere_point_id[geometry_name]:
                if self.win_flag:
                    self.scene.scene.show_geometry(sphere_point_id, False)
                else:
                    self.render.scene.show_geometry(sphere_point_id, False)
                self.sphere_points_poll.append(sphere_point_id)
            del self.geometry_name_to_sphere_point_id[geometry_name]
        else:
            self.scene.scene.remove_geometry(geometry_name)

    def add_coordinate(self, geometry_name, origin, size):
        axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=origin)
        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        if self.win_flag:
            self.scene.scene.remove_geometry(geometry_name)
            self.scene.scene.add_geometry(geometry_name, axis_pcd, mat)
        else:
            self.render.scene.remove_geometry(geometry_name)
            self.render.scene.add_geometry(geometry_name, axis_pcd, mat)

    def add_ground(self):
        ground_plane = o3d.geometry.TriangleMesh.create_box(
            50.0, 0.1, 50.0, create_uv_map=True, map_texture_to_each_face=True)
        ground_plane.compute_triangle_normals()
        import math
        rotate_180 = o3d.geometry.get_rotation_matrix_from_xyz((-math.pi, 0, 0))
        ground_plane.rotate(rotate_180)
        ground_plane.translate((-25.0, 0.1, -25.0))
        ground_plane.paint_uniform_color((1, 1, 1))
        mat = rendering.MaterialRecord()
        mat.base_color = [1.0, 1.0, 1.0, 1.0]
        mat.shader = "defaultLit"
        if self.win_flag:
            self.scene.scene.add_geometry("ground", ground_plane, mat)
        else:
            self.render.scene.add_geometry("ground", ground_plane, mat)

    def _on_menu_sphere(self):
        self.test_add_points()
        print()

    def _set_mouse_mode_sun(self):
        self.scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_SUN)

    def _set_mouse_mode_ibl(self):
        self.scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_IBL)

    def _set_mouse_mode_model(self):
        self.scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_MODEL)

    def add_geometry(self, _id, geometry, mat=None):
        if mat is None:
            mat = rendering.MaterialRecord()
            mat.shader = "defaultLit"
        if self.win_flag:
            self.scene.scene.remove_geometry(_id)
            self.scene.scene.add_geometry(_id, geometry, mat)
        else:
            self.render.scene.remove_geometry(_id)
            self.render.scene.add_geometry(_id, geometry, mat)

    def _on_menu_random(self):
        dirname = r"C:\Users\maoqh\PycharmProjects\OVis\2023-04-18-17-33-32_UWV"
        if os.path.exists(dirname):
            # read all geometry from the dirname and add them to vis
            for f in os.listdir(dirname):
                if f.endswith('.ply'):
                    if f.startswith('pc_'):
                        id = f[3:-4]
                        pc = o3d.io.read_point_cloud(os.path.join(dirname, f))
                        self.add_sphere_pc(id, np.asarray(pc.points), np.asarray(pc.colors))
                    elif f.startswith('mesh_'):
                        id = f[5:-4]
                        mesh = o3d.io.read_triangle_mesh(os.path.join(dirname, f))
                        self.add_geometry(id, mesh)
                    elif f.startswith('lineset_'):
                        id = f[8:-4]
                        lineset = o3d.io.read_line_set(os.path.join(dirname, f))
                        self.add_geometry(id, lineset)
            print('snapshot loaded from', dirname)
            return
        #
        # def thread_main():
        #     for _ in range(0, 20):
        #         gui.Application.instance.post_to_main_thread(
        #             self.window, self.add_sphere)
        #         time.sleep(1)
        #
        # threading.Thread(target=thread_main).start()

    def _on_menu_quit(self):
        gui.Application.instance.quit()

    def _on_menu_shot(self):
        print("Saving image at test.png")
        self.save_sphere_point('pc1', 'pc1.ply', (0.9, 0.1, 0.1))
        #self.save_image("test.png")

    def save_image(self, path):
        def on_image(image):
            img = image
            quality = 9  # png
            if path.endswith(".jpg"):
                quality = 100
            o3d.io.write_image(path, img, quality)

        if self.win_flag:
            img = self.scene.scene.scene.render_to_image(on_image)
        else:
            img = self.render.render_to_image()
            o3d.io.write_image(path, img, 9)




"""
ground_plane = o3d.geometry.TriangleMesh.create_box(
            50.0, 0.1, 50.0, create_uv_map=True, map_texture_to_each_face=True)
ground_plane.compute_triangle_normals()
import math
rotate_180 = o3d.geometry.get_rotation_matrix_from_xyz((-math.pi, 0, 0))
ground_plane.rotate(rotate_180)
ground_plane.translate((-25.0, 0.8, -25.0))
ground_plane.paint_uniform_color((1, 1, 1))
mat = rendering.MaterialRecord()
mat.base_color = [1.0, 1.0, 1.0, 1.0]
mat.shader = "defaultLit"
if self.win_flag:
    self.scene.scene.add_geometry("ground", ground_plane, mat)
self.render.scene.add_geometry("ground", ground_plane, mat)

object_color = np.array([0.5, 0.7626245339585409, 1.0, 1])
human_color = np.array([int(e) for e in "0 238 0".split(" ")] + [255, ]) / 255
noise_color = np.array([int(e) for e in "30 144 255".split(" ")] + [255, ]) / 255

i = 70
pcd = gt['ids_metric'][61806]['point_clouds'][i].copy()
pcd = affine(pcd, extrinsic_matrix)
label = gt['ids_metric'][61806]['pred_segment'][i].copy()
pcd -= pcd.mean(axis=0, keepdims=True)

replace_choice = np.random.choice(512, 200, replace=False)
pcd[replace_choice] = np.random.rand(replace_choice.shape[0], 3) * np.array([1, 1.8, 1]) - np.array([0.5, 1, 0.5])

colors = np.zeros((512, 4))

colors[label==0] = human_color
colors[label==1] = human_color
colors[replace_choice] = noise_color
self.add_points(pcd, colors=colors)
"""


def main():
    if platform.system() == 'Windows':
        renderer = O3DRenderer(True)
    else:
        renderer = O3DRenderer(False)
        renderer.test_add_points()
        renderer.save_image("test.png")


if __name__ == "__main__":
    main()
