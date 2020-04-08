"""
hybrid_lqr_te_control
    - Hayden Morgan
    - Last Update:
    03/21/2020 - HMM
"""
import sys
import numpy as np
import scipy as scp
from chap6.pid_control import pid_control, pd_control_with_rate, pi_control
import parameters.aerosonde_parameters as MAV
import parameters.control_parameters as AP
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
        self.init = True

    def update(self, x, e_I):
        # integrator anti-windup
        if self.init:
            self.x_I = e_I*0
            self.e_I_prev = e_I*0
            self.init = False
        for i in range(np.alen(e_I)):
            if np.alen(e_I) > 1:
                if np.abs(e_I.item(i)-self.e_I_prev.item(i))/self.Ts < 0.2:
                    self.x_I[i, 0] += self.Ts / 2 * (e_I.item(i) + self.e_I_prev.item(i))
            else:
                if np.abs(e_I-self.e_I_prev)/self.Ts < 0.2:
                    self.x_I += self.Ts / 2 * (e_I + self.e_I_prev)
        self.e_I_prev = e_I
        xi = np.vstack([x, self.x_I])
        u = -self.K_lqr @ xi
        u_sat = self._saturate(u)
        return u_sat

    def _saturate(self, u):
        # saturate u at +- self.limit
        u_sat = u
        for i in range(len(u)):
            if u.item(i) >= self.limit.item(2 * i):
                u_sat[i, 0] = self.limit.item(2 * i)
            elif u.item(i) <= self.limit.item(2 * i + 1):
                u_sat[i, 0] = self.limit.item(2 * i + 1)
            else:
                u_sat[i, 0] = u.item(i)
        return u_sat


# Longitudinal Total Energy Control
# (from Nonlinear Total Energy Control for the Longitudinal Dynamics of an Aircraft by M. E. Argyle and R. W. Beard)
class tecs_control:
    def __init__(self, K, limit, Ts, MAV):
        self.Ts = Ts
        self.limit = limit
        self.MAV = MAV
        self.k_T = K.item(0)
        self.k_D = K.item(1)
        self.k_h = K.item(2)
        self.k_Va = K.item(3)
        self.h_d = -self.MAV.pd0
        self.h_d_dot = 0
        self.Va_d = self.MAV.Va0
        self.Va_d_dot = 0

        self.thrust_from_throttle = pi_control(
            kp=AP.thrust_throttle_kp,
            ki=AP.thrust_throttle_ki,
            Ts=Ts,
            limit=1.0)
        self.flight_path_angle_from_elevator = pid_control(
            kp=AP.fpa_elevator_kp,
            ki=AP.fpa_elevator_ki,
            kd=AP.fpa_elevator_kd,
            # Ts=Ts,
            limit=1.0)

    def update(self, state, command):
        # MAV states
        m = self.MAV.mass
        g = self.MAV.gravity
        h = state.item(0)           # altitude
        Va = state.item(1)          # airspeed
        theta = state.item(2)       # pitch angle
        q = state.item(3)           # pitch rate
        T = state.item(4)           # thrust
        T_D = state.item(5)         # thrust to counteract drag
        alpha = state.item(6)       # angle of attack
        h_c = command.item(0)       # commanded altitude
        Va_c = command.item(1)      # commanded airspeed

        # Intermediate Desired Outputs
        self.h_d_dot = self.k_h * (h_c - self.h_d)          # Desired altitude rate (eq 21)
        self.Va_d_dot = self.k_Va * (Va_c - self.Va_d)      # Desired airspeed rate (eq 22)
        self.h_d = self.h_d_dot * self.Ts + self.h_d        # Desired altitude
        self.Va_d = self.Va_d_dot * self.Ts + self.Va_d     # Desired airspeed

        ## ------------ ENERGY ------------ ##
        # Energy error calculations
        E_tilde_K = 1/2*m*(self.Va_d**2 - Va**2)  # Kinetic energy error
        E_tilde_P = m*g*(self.h_d - h)  # Potential energy error
        E_tilde_T = E_tilde_P + E_tilde_K  # Total energy error

        E_dot_Td = self.Va_d*(T - T_D)     # Desired total energy rate (eq. 9)

        # High Level Control
        delta_T = E_dot_Td/Va + self.k_T*E_tilde_T/Va     # Thrust add-on to trim
        T_c = T_D + delta_T                                 # Commanded thrust
        k1 = np.abs(self.k_T - self.k_D)
        k2 = self.k_T + self.k_D
        # Commanded Flight Path Angle
        theta_max = np.sin(np.radians(40))
        temp = self.saturate(self.h_d_dot/Va + 1/(2.0*m*g*Va)*(-k1*E_tilde_K + k2*E_tilde_P), -theta_max, theta_max)
        gamma_c = np.arcsin(temp)
        theta_c = gamma_c + alpha

        delta_e = self.flight_path_angle_from_elevator.update_with_rate(theta_c, theta, q)
        delta_t = self.thrust_from_throttle.update(T_c, T)

        # Control Outputs
        u = np.array([[delta_e, delta_t]])
        u_sat = self._saturate(u)
        return [u_sat, theta_c]

    def saturate(self, u, lower, upper):
        # saturate u at +- self.limit
        u_sat = u
        if u >= upper:
            u_sat = upper
        elif u <= lower:
            u_sat = lower
        else:
            u_sat = u
        return u_sat

    def _saturate(self, u):
        # saturate u at +- self.limit
        u_sat = u
        for i in range(len(u[0])):
            if u.item(i) >= self.limit.item(2*i):
                u_sat[0, i] = self.limit.item(2*i)
            elif u.item(i) <= self.limit.item(2*i+1):
                u_sat[0, i] = self.limit.item(2*i+1)
            else:
                u_sat[0, i] = u.item(i)
        return u_sat