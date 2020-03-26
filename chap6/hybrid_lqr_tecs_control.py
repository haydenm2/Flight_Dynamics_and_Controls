"""
hybrid_lqr_te_control
    - Hayden Morgan
    - Last Update:
    03/21/2020 - HMM
"""
import sys
import numpy as np
import scipy as scp
from parameters import aerosonde_parameters as MAV
sys.path.append('..')

# lateral lqr controller
class lqr_control:
    def __init__(self, A, B, Q, R, limit, Ts):
        self.Ts = Ts
        self.limit = limit
        self.Q = Q
        self.R = R
        self.A = A
        self.B = B
        self.P = scp.linalg.solve_continuous_are(self.A, self.B, self.Q, self.R)
        self.K_lqr = np.linalg.inv(self.R) @ self.B.T @ self.P
        self.x_I = 0
        self.e_I_prev = 0

    def update(self, x, e_I):
        self.x_I += self.Ts/2*(e_I + self.e_I_prev)
        self.e_I_prev = e_I
        xi = np.vstack([x, self.x_I])
        u = -self.K_lqr @ xi
        u_sat = self._saturate(u)
        return u_sat

    def _saturate(self, u):
        # saturate u at +- self.limit
        u_sat = u
        for i in range(len(u)):
            if u.item(i) >= self.limit.item(i):
                u_sat[i, 0] = self.limit.item(i)
            elif u.item(i) <= -self.limit.item(i):
                u_sat[i, 0] = -self.limit.item(i)
            else:
                u_sat[i, 0] = u.item(i)
        return u_sat

# longitudinal Total Energy Control
class tecs_control:
    def __init__(self, Ts=0.01, limit=1.0):
        self.Ts = Ts
        self.limit = limit

    def update(self, y_ref, y):
        # control calculations
        u = 0 #TODO
        u_sat = self._saturate(u)
        return u_sat

    def _saturate(self, u):
        # saturate u at +- self.limit
        if u >= self.limit:
            u_sat = self.limit
        elif u <= -self.limit:
            u_sat = -self.limit
        else:
            u_sat = u
        return u_sat