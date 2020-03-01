import numpy as np
import sys
sys.path.append('..')
from chap11.dubins_parameters import dubins_parameters
from message_types.msg_path import msg_path

class path_manager:
    def __init__(self):
        # message sent to path follower
        self.path = msg_path()
        # pointers to previous, current, and next waypoints
        self.ptr_previous = 0
        self.ptr_current = 1
        self.ptr_next = 2
        # flag that request new waypoints from path planner
        self.flag_need_new_waypoints = True
        self.num_waypoints = 0
        self.halfspace_n = np.inf * np.ones((3,1))
        self.halfspace_r = np.inf * np.ones((3,1))
        # state of the manager state machine
        self.manager_state = 1  # 1 == straight line, 2 == orbit
        # dubins path parameters
        self.dubins_path = dubins_parameters()
        # additional setup
        self.path_changed = True
        self.init = True
        self.initialize_pointers()

    def update(self, waypoints, radius, state):
        # this flag is set for one time step to signal a redraw in the viewer
        self.num_waypoints = waypoints.num_waypoints
        if self.path.flag_path_changed == True:
            self.path.flag_path_changed = False
        if waypoints.num_waypoints == 0:
            waypoints.flag_manager_requests_waypoints = True
        else:
            if waypoints.type == 'straight_line':
                self.line_manager(waypoints, state)
            elif waypoints.type == 'fillet':
                self.fillet_manager(waypoints, radius, state)
            elif waypoints.type == 'dubins':
                self.dubins_manager(waypoints, radius, state)
            else:
                print('Error in Path Manager: Undefined waypoint type.')
        return self.path

    def line_manager(self, waypoints, state):
        if waypoints.flag_waypoints_changed:
            self.initialize_pointers()
        p = np.array([[state.pn], [state.pe], [-state.h]])
        w_prev = waypoints.ned[:, self.ptr_previous].reshape(-1, 1)
        w_curr = waypoints.ned[:, self.ptr_current].reshape(-1, 1)
        w_next = waypoints.ned[:, self.ptr_next].reshape(-1, 1)
        self.halfspace_r = w_prev
        q_im = (w_curr - w_prev)/np.linalg.norm((w_curr - w_prev))
        q_i = (w_next - w_curr)/np.linalg.norm((w_next - w_curr))
        self.halfspace_n = ((q_im + q_i)/np.linalg.norm((q_im + q_i)))
        if self.inHalfSpace(p):
            if self.ptr_current <= (self.num_waypoints - 1):
                self.increment_pointers()

        w_prev = waypoints.ned[:, self.ptr_previous].reshape(-1, 1)
        w_curr = waypoints.ned[:, self.ptr_current].reshape(-1, 1)
        q_im = (w_curr - w_prev) / np.linalg.norm((w_curr - w_prev))
        self.path.flag = 'line'
        self.path.line_origin = w_prev  # r in book
        self.path.line_direction = q_im  # q in book

    def fillet_manager(self, waypoints, radius, state):
        if waypoints.flag_waypoints_changed:
            self.initialize_pointers()
        p = np.array([[state.pn], [state.pe], [-state.h]])
        w_prev = waypoints.ned[:, self.ptr_previous].reshape(-1, 1)
        w_curr = waypoints.ned[:, self.ptr_current].reshape(-1, 1)
        w_next = waypoints.ned[:, self.ptr_next].reshape(-1, 1)
        q_im = (w_curr - w_prev) / np.linalg.norm((w_curr - w_prev))
        q_i = (w_next - w_curr) / np.linalg.norm((w_next - w_curr))
        g = np.arccos(-q_im.T @ q_i)
        if self.manager_state == 1:
            self.path.flag = 'line'  # 1 in book
            self.path.line_origin = w_prev  # r in book
            self.path.line_direction = q_im  # q in book
            self.halfspace_r = waypoints.ned[:, self.ptr_current].reshape(-1, 1) + (radius/(np.sin(g/2.0)))*q_i  # z in book
            self.halfspace_n = q_im
            if self.inHalfSpace(p):
                self.manager_state = 2
        elif self.manager_state == 2:
            self.path.flag = 'orbit'  # 2 in book
            self.path.orbit_center = waypoints.ned[:, self.ptr_current].reshape(-1, 1) + (radius/np.tan(g/2.0)) @ (q_im - q_i)/np.linalg.norm(q_im - q_i)  # c in book
            self.path.orbit_radius = radius  # rho in book
            self.path.orbit_direction = self.OrbitDirType(np.sign(q_im.item(0)*q_i.item(1) - q_im.item(1)*q_i.item(0)))  # lambda in book
            self.halfspace_r = waypoints.ned[:, self.ptr_current].reshape(-1, 1) - (radius/(np.sin(g/2.0)))*q_im  # z in book
            self.halfspace_n = q_i
            if self.inHalfSpace(p):
                if self.ptr_current <= (self.num_waypoints - 1):
                    self.increment_pointers()
                    self.manager_state = 1

    def dubins_manager(self, waypoints, radius, state):
        p = np.array([[state.pn], [state.pe], [-state.h]])
        w_prev = waypoints.ned[:, self.ptr_previous].reshape(-1, 1)
        chi_prev = waypoints.course.item(self.ptr_previous)
        w_curr = waypoints.ned[:, self.ptr_current].reshape(-1, 1)
        chi_curr = waypoints.course.item(self.ptr_current)
        self.dubins_path.update(w_prev, chi_prev, w_curr, chi_curr, radius)
        if self.manager_state == 1:
            self.path.flag = 'orbit'
            self.path.orbit_center = self.dubins_path.center_s  # c=c_s in book
            self.path.orbit_radius = self.dubins_path.radius  # rho=R in book
            self.path.orbit_direction = self.OrbitDirType(self.dubins_path.dir_s)  # lambda=lambda_S in book
            self.halfspace_r = self.dubins_path.r1  # z1 in book
            self.halfspace_n = -self.dubins_path.n1  # -q1 in book
            if self.init == True:
                self.init = False
                self.path.flag_path_changed = True
            if self.inHalfSpace(p):
                self.manager_state = 2
                self.init = True
        elif self.manager_state == 2:
            self.halfspace_r = self.dubins_path.r1  # z1 in book
            self.halfspace_n = self.dubins_path.n1  # q1 in book
            if self.init == True:
                self.init = False
                self.path.flag_path_changed = True
            if self.inHalfSpace(p):
                self.path.flag_path_changed = True
                self.manager_state = 3
                self.init = True
        elif self.manager_state == 3:
            self.path.flag = 'line'
            self.path.line_origin = self.dubins_path.r1 # r=z1 in book
            self.path.line_direction = self.dubins_path.n1 # q=q1 in book
            self.halfspace_r = self.dubins_path.r2 # z2 in book
            self.halfspace_n = self.dubins_path.n1 # q1 in book
            if self.init == True:
                self.init = False
                self.path.flag_path_changed = True
            if self.inHalfSpace(p):
                self.manager_state = 4
                self.init = True
        elif self.manager_state == 4:
            self.path.flag = 'orbit'
            self.path.orbit_center = self.dubins_path.center_e # c=c_e in book
            self.path.orbit_radius = self.dubins_path.radius  # rho=R in book
            self.path.orbit_direction = self.OrbitDirType(self.dubins_path.dir_e) # lambda=lambda_e in book
            self.halfspace_r = self.dubins_path.r3 # z3 in book
            self.halfspace_n = -self.dubins_path.n3 # -q3 in book
            if self.init == True:
                self.init = False
                self.path.flag_path_changed = True
            if self.inHalfSpace(p):
                self.manager_state = 5
                self.init = True
        elif self.manager_state == 5:
            self.halfspace_r = self.dubins_path.r3 # z3 in book
            self.halfspace_n = self.dubins_path.n3 # q3 in book
            if self.init == True:
                self.init = False
                self.path.flag_path_changed = True
            if self.inHalfSpace(p):
                self.manager_state = 1
                self.init = True
                if self.ptr_current <= (self.num_waypoints - 1):
                    self.increment_pointers()
                # w_prev = waypoints.ned[:, self.ptr_previous].reshape(-1, 1)
                # chi_prev = waypoints.course[0, self.ptr_previous]
                # w_curr = waypoints.ned[:, self.ptr_current].reshape(-1, 1)
                # chi_curr = waypoints.course[0, self.ptr_current]
                # self.dubins_path.update(w_prev, chi_prev, w_curr, chi_curr, radius)

    def initialize_pointers(self):
        self.ptr_previous = 0
        self.ptr_current = 1
        self.ptr_next = 2

    def increment_pointers(self):
        self.ptr_previous += 1
        self.ptr_current += 1
        self.ptr_next += 1

    def inHalfSpace(self, pos):
        if (pos-self.halfspace_r).T @ self.halfspace_n >= 0:
            return True
        else:
            return False

    def OrbitDirType(self, type):
        if type == 1:
            res = 'CW'
        elif type == -1:
            res = 'CCW'
        return res

