"""
mavsimPy: path drawing function
    - Beard & McLain, PUP, 2012
    - Update history:  
        1/8/2019 - RWB
"""
import sys
sys.path.append("..")
import numpy as np

import pyqtgraph as pg
import pyqtgraph.opengl as gl
import pyqtgraph.Vector as Vector

from tools.tools import RotationBody2Vehicle as Euler2Rotation

class path_viewer():
    def __init__(self):
        self.scale = 4000
        # initialize Qt gui application and window
        self.app = pg.QtGui.QApplication([])  # initialize QT
        self.window = gl.GLViewWidget()  # initialize the view object
        self.window.setWindowTitle('Path Viewer')
        self.window.setGeometry(0, 0, 1000, 1000)  # args: upper_left_x, upper_right_y, width, height
        grid = gl.GLGridItem() # make a grid to represent the ground
        grid.scale(self.scale/20, self.scale/20, self.scale/20) # set the size of the grid (distance between each line)
        self.window.addItem(grid) # add grid to viewer
        self.window.setCameraPosition(distance=self.scale, elevation=90, azimuth=0)
        self.window.setBackgroundColor('k')  # set background color to black
        self.window.show()  # display configured window
        self.window.raise_() # bring window to the front
        self.plot_initialized = False # has the mav been plotted yet?
        # get points that define the non-rotated, non-translated mav and the mesh colors
        self.points, self.meshColors = self._get_mav_points()

    ###################################
    # public functions
    def update(self, path, state):
        """
        Update the drawing of the mav.

        The input to this function is a (message) class with properties that define the state.
        The following properties are assumed:
            state.pn  # north position
            state.pe  # east position
            state.h   # altitude
            state.phi  # roll angle
            state.theta  # pitch angle
            state.psi  # yaw angle
        """
        mav_position = np.array([[state.pn], [state.pe], [-state.h]])  # NED coordinates
        # attitude of mav as a rotation matrix R from body to inertial
        R = Euler2Rotation(state.phi, state.theta, state.psi)
        # rotate and translate points defining mav
        rotated_points = self._rotate_points(self.points, R)
        translated_points = self._translate_points(rotated_points, mav_position)
        # convert North-East Down to East-North-Up for rendering
        R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        translated_points = R @ translated_points
        # convert points to triangular mesh defined as array of three 3D points (Nx3x3)
        mesh = self._points_to_mesh(translated_points)

        # initialize the drawing the first time update() is called
        if not self.plot_initialized:
            if path.flag=='line':
                straight_line_object = self.straight_line_plot(path)
                self.window.addItem(straight_line_object)  # add straight line to plot
            else:  # path.flag=='orbit
                orbit_object = self.orbit_plot(path)
                self.window.addItem(orbit_object)
            # initialize drawing of triangular mesh.
            self.body = gl.GLMeshItem(vertexes=mesh,  # defines the triangular mesh (Nx3x3)
                                      vertexColors=self.meshColors, # defines mesh colors (Nx1)
                                      drawEdges=True,  # draw edges between mesh elements
                                      smooth=False,  # speeds up rendering
                                      computeNormals=False)  # speeds up rendering
            self.window.addItem(self.body)  # add body to plot
            self.plot_initialized = True

        # else update drawing on all other calls to update()
        else:
            # reset mesh using rotated and translated points
            self.body.setMeshData(vertexes=mesh, vertexColors=self.meshColors)

        # update the center of the camera view to the mav location
        view_location = Vector(state.pe, state.pn, state.h)  # defined in ENU coordinates
        self.window.opts['center'] = view_location
        # redraw
        self.app.processEvents()

    ###################################
    # private functions
    def _rotate_points(self, points, R):
        "Rotate points by the rotation matrix R"
        rotated_points = R @ points
        return rotated_points

    def _translate_points(self, points, translation):
        "Translate points by the vector translation"
        translated_points = points + np.dot(translation, np.ones([1,points.shape[1]]))
        return translated_points

    def _get_mav_points(self):
        """"
            Points that define the mav, and the colors of the triangular mesh
            Define the points on the aircraft following diagram in Figure 2.14
        """

        fuse_l1 = 2
        fuse_l2 = 1
        fuse_l3 = 4
        fuse_h = 1
        fuse_w = 1
        wing_l = 1
        wing_w = 5
        tail_h = 1
        tailwing_l = 1
        tailwing_w = 3

        #points are in NED coordinates
        points = np.array([[fuse_l1, 0, 0],  # point 1
                           [fuse_l2, fuse_w/2, -fuse_h/2],  # point 2
                           [fuse_l2, -fuse_w/2, -fuse_h/2],  # point 3
                           [fuse_l2, -fuse_w/2, fuse_h/2],  # point 4
                           [fuse_l2, fuse_w/2, fuse_h/2],  # point 5
                           [-fuse_l3, 0, 0],  # point 6
                           [0, wing_w/2, 0],  # point 7
                           [-wing_l, wing_w/2, 0],  # point 8
                           [-wing_l, -wing_w/2, 0],  # point 9
                           [0, -wing_w/2, 0],  # point 10
                           [-fuse_l3 + tailwing_l, tailwing_w/2, 0],  # point 11
                           [-fuse_l3, tailwing_w/2, 0],  # point 12
                           [-fuse_l3, -tailwing_w/2, 0],  # point 13
                           [-fuse_l3 + tailwing_l, -tailwing_w/2, 0],  # point 14
                           [-fuse_l3 + tailwing_l, 0, 0],  # point 15
                           [-fuse_l3, 0, -tail_h],  # point 16
                           ]).T
        # scale points for better rendering
        scale = 10
        points = scale * points

        #   define the colors for each face of triangular mesh
        red = np.array([1., 0., 0., 1])
        green = np.array([0., 1., 0., 1])
        blue = np.array([0., 0., 1., 1])
        yellow = np.array([1., 1., 0., 1])
        meshColors = np.empty((13, 3, 4), dtype=np.float32)
        meshColors[0] = yellow  # nose 1
        meshColors[1] = yellow  # nose 2
        meshColors[2] = yellow  # nose 3
        meshColors[3] = yellow  # nose 4
        meshColors[4] = blue  # body 1
        meshColors[5] = blue  # body 2
        meshColors[6] = blue  # body 3
        meshColors[7] = blue  # body 4
        meshColors[8] = red  # wing 1
        meshColors[9] = red  # wing 2
        meshColors[10] = green  # tailwing 1
        meshColors[11] = green  # tailwing 2
        meshColors[12] = red  # rudder
        return points, meshColors

    def _points_to_mesh(self, points):
        """"
        Converts points to triangular mesh
        Each mesh face is defined by three 3D points
          (a rectangle requires two triangular mesh faces)
        """
        points=points.T
        mesh = np.array([[points[0], points[1], points[2]],  # nose 1
                         [points[0], points[2], points[3]],  # nose 2
                         [points[0], points[3], points[4]],  # nose 3
                         [points[0], points[4], points[1]],  # nose 4
                         [points[5], points[1], points[2]],  # body 1
                         [points[5], points[2], points[3]],  # body 2
                         [points[5], points[3], points[4]],  # body 3
                         [points[5], points[4], points[1]],  # body 4
                         [points[6], points[7], points[8]],  # wing 1
                         [points[8], points[9], points[6]],  # wing 2
                         [points[10], points[11], points[12]],  # tailwing 1
                         [points[12], points[13], points[10]],  # tailwing 2
                         [points[5], points[14], points[15]],  # rudder
                         ])
        return mesh

    def straight_line_plot(self, path):
        points = np.array([[path.line_origin.item(0),
                            path.line_origin.item(1),
                            path.line_origin.item(2)],
                           [path.line_origin.item(0) + self.scale * path.line_direction.item(0),
                            path.line_origin.item(1) + self.scale * path.line_direction.item(1),
                            path.line_origin.item(2) + self.scale * path.line_direction.item(2)]])
        # convert North-East Down to East-North-Up for rendering
        R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        points = points @ R.T
        red = np.array([[1., 0., 0., 1]])
        path_color = np.concatenate((red, red), axis=0)
        object = gl.GLLinePlotItem(pos=points,
                                   color=path_color,
                                   width=2,
                                   antialias=True,
                                   mode='lines')
        return object

    def orbit_plot(self, path):
        N = 100
        red = np.array([[1., 0., 0., 1]])
        theta = 0
        points = np.array([[path.orbit_center.item(0) + path.orbit_radius,
                            path.orbit_center.item(1),
                            path.orbit_center.item(2)]])
        path_color = red
        for i in range(0, N):
            theta += 2 * np.pi / N
            new_point = np.array([[path.orbit_center.item(0) + path.orbit_radius * np.cos(theta),
                                   path.orbit_center.item(1) + path.orbit_radius * np.sin(theta),
                                   path.orbit_center.item(2)]])
            points = np.concatenate((points, new_point), axis=0)
            path_color = np.concatenate((path_color, red), axis=0)
        # convert North-East Down to East-North-Up for rendering
        R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        points = points @ R.T
        object = gl.GLLinePlotItem(pos=points,
                                   color=path_color,
                                   width=2,
                                   antialias=True,
                                   mode='line_strip')
        return object

