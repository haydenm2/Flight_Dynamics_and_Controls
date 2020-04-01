import numpy as np
from math import sin, cos, atan, atan2
import sys

sys.path.append('..')
from message_types.msg_autopilot import msg_autopilot
from tools.tools import Rzp

class path_follower:
    def __init__(self):
        self.chi_inf = np.radians(60)  # approach angle for large distance from straight-line path
        self.k_path = 0.02  # proportional gain for straight-line path following
        self.k_orbit = 3.0  # proportional gain for orbit following
        self.gravity = 9.8
        self.autopilot_commands = msg_autopilot()  # message sent to autopilot

    def update(self, path, state):
        if path.flag=='line':
            self._follow_straight_line(path, state)
        elif path.flag=='orbit':
            self._follow_orbit(path, state)
        return self.autopilot_commands

    def _follow_straight_line(self, path, state):
        chi_q = np.arctan2(path.line_direction.item(1), path.line_direction.item(0))
        chi = state.chi
        chi_q = self._wrap(chi_q, chi)
        p_i = np.array([[state.pn], [state.pe], [-state.h]])
        r_i = path.line_origin
        R_iP = Rzp(chi_q)
        e_p = R_iP @ (p_i - r_i)
        e_ip = p_i - r_i
        q = path.line_direction
        k = np.array([[0], [0], [1]])
        n = (np.cross(q.flatten(), k.flatten()) / np.linalg.norm(np.cross(q.flatten(), k.flatten()))).reshape(-1, 1)
        s = e_ip - (n.T @ e_ip) * n
        wn = state.wn
        we = state.we
        wd = 0.0
        Va = state.Va
        g = 9.81
        R = path.orbit_radius
        if path.orbit_direction == 'CW':
            lbda = 1
        else:
            lbda = -1

        self.autopilot_commands.airspeed_command = path.airspeed
        self.autopilot_commands.course_command = chi_q - self.chi_inf*2.0/np.pi*np.arctan(self.k_path*e_p.item(1))
        self.autopilot_commands.altitude_command = -r_i.item(2) - np.sqrt(s.item(0)**2 + s.item(1)**2)*(q.item(2)/np.sqrt(q.item(0)**2 + q.item(1)**2))
        self.autopilot_commands.phi_feedforward = 0

    def _follow_orbit(self, path, state):
        vpsi = np.arctan2(state.pe - path.orbit_center.item(1), state.pn - path.orbit_center.item(0))
        p_i = np.array([[state.pn], [state.pe], [-state.h]])
        d = np.linalg.norm(p_i - path.orbit_center)
        if path.orbit_direction == 'CW':
            lbda = 1
        else:
            lbda = -1
        rho = path.orbit_radius
        wn = state.wn
        we = state.we
        wd = 0.0
        chi = state.chi
        Va = state.Va
        g = 9.81
        R = rho

        self.autopilot_commands.airspeed_command = path.airspeed
        self.autopilot_commands.course_command = vpsi + lbda*(np.pi/2.0 + np.arctan2(self.k_orbit*(d-rho), rho))
        self.autopilot_commands.altitude_command = -path.orbit_center.item(2)
        self.autopilot_commands.phi_feedforward = lbda*np.arctan2(((wn*np.cos(chi) + we*np.sin(chi)) + np.sqrt(Va**2 - (wn*np.sin(chi) - we*np.cos(chi))**2 - wd**2))**2, g*R*np.sqrt((Va**2 - (wn*np.sin(chi) - we*np.cos(chi))**2 - wd**2)/(Va**2 - wd**2)))

    def _wrap(self, chi_c, chi):
        while chi_c-chi > np.pi:
            chi_c = chi_c - 2.0 * np.pi
        while chi_c-chi < -np.pi:
            chi_c = chi_c + 2.0 * np.pi
        return chi_c

