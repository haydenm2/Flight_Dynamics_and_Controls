"""
mav_dynamics
    - this file implements the dynamic equations of motion for MAV
    - use unit quaternion for the attitude state
    
"""
import sys
sys.path.append('..')
import numpy as np

# load message types
from message_types.msg_state import msg_state

import parameters.aerosonde_parameters as MAV
from tools.tools import Quaternion2Rotation, Quaternion2Euler, Euler2Rotation

class mav_dynamics:
    def __init__(self, Ts):
        self._ts_simulation = Ts
        # set initial states based on parameter file
        # _state is the 13x1 internal state of the aircraft that is being propagated:
        # _state = [pn, pe, pd, u, v, w, e0, e1, e2, e3, p, q, r]
        # We will also need a variety of other elements that are functions of the _state and the wind.
        # self.true_state is a 19x1 vector that is estimated and used by the autopilot to control the aircraft:
        # true_state = [pn, pe, h, Va, alpha, beta, phi, theta, chi, p, q, r, Vg, wn, we, psi, gyro_bx, gyro_by, gyro_bz]
        self._state = np.array([[MAV.pn0],  # (0)
                               [MAV.pe0],   # (1)
                               [MAV.pd0],   # (2)
                               [MAV.u0],    # (3)
                               [MAV.v0],    # (4)
                               [MAV.w0],    # (5)
                               [MAV.e0],    # (6)
                               [MAV.e1],    # (7)
                               [MAV.e2],    # (8)
                               [MAV.e3],    # (9)
                               [MAV.p0],    # (10)
                               [MAV.q0],    # (11)
                               [MAV.r0]])   # (12)
        # store wind data for fast recall since it is used at various points in simulation
        self._wind = np.array([[0.], [0.], [0.]])  # wind in NED frame in meters/sec
        self._update_velocity_data()
        # store forces to avoid recalculation in the sensors function
        self._forces = np.array([[0.], [0.], [0.]])
        self._Va = MAV.u0
        self._alpha = 0
        self._beta = 0
        # initialize true_state message
        self.msg_true_state = msg_state()

    ###################################
    # public functions
    def update_state(self, delta, wind):
        '''
            Integrate the differential equations defining dynamics, update sensors
            delta = (delta_a, delta_e, delta_r, delta_t) are the control inputs
            wind is the wind vector in inertial coordinates
            Ts is the time step between function calls.
        '''
        # get forces and moments acting on rigid bod
        forces_moments = self._forces_moments(delta)

        # Integrate ODE using Runge-Kutta RK4 algorithm
        time_step = self._ts_simulation
        k1 = self._derivatives(self._state, forces_moments)
        k2 = self._derivatives(self._state + time_step/2.*k1, forces_moments)
        k3 = self._derivatives(self._state + time_step/2.*k2, forces_moments)
        k4 = self._derivatives(self._state + time_step*k3, forces_moments)
        self._state += time_step/6 * (k1 + 2*k2 + 2*k3 + k4)

        # normalize the quaternion
        e0 = self._state.item(6)
        e1 = self._state.item(7)
        e2 = self._state.item(8)
        e3 = self._state.item(9)
        normE = np.sqrt(e0**2+e1**2+e2**2+e3**2)
        self._state[6][0] = self._state.item(6)/normE
        self._state[7][0] = self._state.item(7)/normE
        self._state[8][0] = self._state.item(8)/normE
        self._state[9][0] = self._state.item(9)/normE

        # update the airspeed, angle of attack, and side slip angles using new state
        self._update_velocity_data(wind)

        # update the message class for the true state
        self._update_msg_true_state()

    ###################################
    # private functions
    def _derivatives(self, state, forces_moments):
        """
        for the dynamics xdot = f(x, u), returns f(x, u)
        """
        # extract the states
        pn = state.item(0)
        pe = state.item(1)
        pd = state.item(2)
        u = state.item(3)
        v = state.item(4)
        w = state.item(5)
        e0 = state.item(6)
        e1 = state.item(7)
        e2 = state.item(8)
        e3 = state.item(9)
        p = state.item(10)
        q = state.item(11)
        r = state.item(12)
        #   extract forces/moments
        fx = forces_moments.item(0)
        fy = forces_moments.item(1)
        fz = forces_moments.item(2)
        l = forces_moments.item(3)
        m = forces_moments.item(4)
        n = forces_moments.item(5)

        # position kinematics
        pn_dot = (e1 ** 2 + e0 ** 2 - e2 ** 2 - e3 ** 2) * u + 2 * (e1 * e2 - e3 * e0) * v + 2 * (e1 * e3 + e2 * e0) * w
        pe_dot = 2 * (e1 * e2 + e3 * e0) * u + (e2 ** 2 + e0 ** 2 - e1 ** 2 - e3 ** 2) * v + 2 * (e2 * e3 - e1 * e0) * w
        pd_dot = 2 * (e1 * e3 - e2 * e0) * u + 2 * (e2 * e3 + e1 * e0) * v + (e3 ** 2 + e0 ** 2 - e1 ** 2 - e2 ** 2) * w

        # position dynamics
        mass = MAV.mass
        u_dot = (r * v - q * w) + 1 / mass * fx
        v_dot = (p * w - r * u) + 1 / mass * fy
        w_dot = (q * u - p * v) + 1 / mass * fz

        # rotational kinematics
        e0_dot = 0.5 * (-p * e1 - q * e2 - r * e3)
        e1_dot = 0.5 * (p * e0 + r * e2 - q * e3)
        e2_dot = 0.5 * (q * e0 - r * e1 + p * e3)
        e3_dot = 0.5 * (r * e0 + q * e1 - p * e2)

        # rotatonal dynamics
        p_dot = (MAV.gamma1 * p * q - MAV.gamma2 * q * r) + (MAV.gamma3 * l + MAV.gamma4 * n)
        q_dot = (MAV.gamma5 * p * r - MAV.gamma6 * (p ** 2 - r ** 2)) + (1 / MAV.Jy * m)
        r_dot = (MAV.gamma7 * p * q - MAV.gamma1 * q * r) + (MAV.gamma4 * l + MAV.gamma8 * n)

        # collect the derivative of the states
        x_dot = np.array([[pn_dot, pe_dot, pd_dot, u_dot, v_dot, w_dot,
                           e0_dot, e1_dot, e2_dot, e3_dot, p_dot, q_dot, r_dot]]).T
        return x_dot

    def _update_velocity_data(self, wind=np.zeros((6,1))):
        # compute relative airspeed components
        self._ur = self._state.item(3) - wind.item(0)
        self._vr = self._state.item(4) - wind.item(1)
        self._wr = self._state.item(5) - wind.item(2)

        e0 = self._state.item(6)
        e1 = self._state.item(7)
        e2 = self._state.item(8)
        e3 = self._state.item(9)
        phi, theta, psi = Quaternion2Euler(np.array([e0, e1, e2, e3]))
        self._Vg = Euler2Rotation(phi, theta, psi).transpose() @ self._state[3:6]

        # compute airspeed
        self._Va = np.sqrt(self._ur**2 + self._vr**2 + self._wr**2)
        # compute angle of attack
        self._alpha = np.arctan2(self._wr, self._ur)
        # compute sideslip angle
        self._beta = np.arcsin(self._vr/self._Va)

    def _forces_moments(self, delta):
        """
        return the forces on the UAV based on the state, wind, and control surfaces
        :param delta: np.matrix(delta_a, delta_e, delta_r, delta_t)
        :return: Forces and Moments on the UAV np.matrix(Fx, Fy, Fz, Ml, Mn, Mm)
        """
        mass = MAV.mass
        g = MAV.gravity
        rho = MAV.rho
        S = MAV.S_wing
        a = self._alpha
        b = self._beta
        c = MAV.c
        p = self._state.item(10)
        q = self._state.item(11)
        r = self._state.item(12)

        # control surface offsets
        delta_a = delta[0]
        delta_e = delta[1]
        delta_r = delta[2]
        delta_t = delta[3]

        # drag coefficients
        C_D_q = MAV.C_D_q
        C_L_q = MAV.C_L_q
        C_D_de = MAV.C_D_delta_e
        C_L_de = MAV.C_L_delta_e
        C_Y_0 = MAV.C_Y_0
        C_Y_b = MAV.C_Y_beta
        C_Y_p = MAV.C_Y_p
        C_Y_r = MAV.C_Y_r
        C_Y_da = MAV.C_Y_delta_a
        C_Y_dr = MAV.C_Y_delta_r
        C_l_0 = MAV.C_ell_0
        C_l_b = MAV.C_ell_beta
        C_l_p = MAV.C_ell_p
        C_l_r = MAV.C_ell_r
        C_l_da = MAV.C_ell_delta_a
        C_l_dr = MAV.C_ell_delta_r
        C_m_0 = MAV.C_m_0
        C_m_a = MAV.C_m_alpha
        C_m_q = MAV.C_m_q
        C_m_de = MAV.C_m_delta_e
        C_n_0 = MAV.C_n_0
        C_n_b = MAV.C_n_beta
        C_n_p = MAV.C_n_p
        C_n_r = MAV.C_n_r
        C_n_da = MAV.C_n_delta_a
        C_n_dr = MAV.C_n_delta_r

        f_p = 0.5*MAV.rho*MAV.S_prop*MAV.C_prop*np.array([(MAV.k_motor*delta_t)**2 - self.msg_true_state.Va**2, 0, 0])
        m_p = np.array([-MAV.kTp*(MAV.kOmega*delta_t)**2, 0, 0])

        # Calculate forces/moments
        # Longitudinal Aerodynamics
        fx = f_p[0] + 0.5 * rho * self._Va**2 * S * (-self.C_D(a)*np.cos(a) + self.C_L(a)*np.sin(a)) + (-C_D_q*np.cos(a) + C_L_q*np.sin(a))*c/(2*self._Va)*q + (-C_D_de*np.cos(a) + C_L_de*np.sin(a))*delta_e
        fz = f_p[2] + 0.5 * rho * self._Va**2 * S * (-self.C_D(a)*np.sin(a) - self.C_L(a)*np.cos(a)) + (-C_D_q*np.sin(a) - C_L_q*np.cos(a))*c/(2*self._Va)*q + (-C_D_de*np.sin(a) - C_L_de*np.cos(a))*delta_e
        My = m_p[1] + 0.5 * rho * self._Va**2 * S * c * (C_m_0 + C_m_a*a + C_m_q*(c/(2*self._Va))*q + C_m_de*delta_e)
        # Lateral Aerodynamics
        fy = f_p[1] + 0.5 * rho * self._Va**2 * S * (C_Y_0 + C_Y_b*b + C_Y_p*(b/(2*self._Va))*p + C_Y_r*(b/(2*self._Va))*r + C_Y_da*delta_a + C_Y_dr*delta_r)
        Mx = m_p[0] + 0.5 * rho * self._Va**2 * S * b * (C_l_0 + C_l_b*b + C_l_p*(b/(2*self._Va))*p + C_l_r*(b/(2*self._Va))*r + C_l_da*delta_a + C_l_dr*delta_r)
        Mz = m_p[2] + 0.5 * rho * self._Va**2 * S * b * (C_n_0 + C_n_b*b + C_n_p*(b/(2*self._Va))*p + C_n_r*(b/(2*self._Va))*r + C_n_da*delta_a + C_n_dr*delta_r)

        return np.array([[fx, fy, fz, Mx, My, Mz]]).T

    def C_D(self, alpha):
        return MAV.C_D_p + (MAV.C_L_0 + MAV.C_L_alpha*alpha)**2/(np.pi*MAV.e*MAV.AR)

    def C_L(self, alpha):
        sig = (1 + np.exp(-MAV.M*(alpha-MAV.alpha0))+np.exp(MAV.M*(alpha+MAV.alpha0)))/((1+np.exp(-MAV.M*(alpha-MAV.alpha0)))*(1+np.exp(MAV.M*(alpha+MAV.alpha0))))
        return (1-sig)*(MAV.C_L_0+MAV.C_L_alpha*alpha)+sig*(2*np.sign(alpha)*np.sin(alpha)**2*np.cos(alpha))

    def _update_msg_true_state(self):
        # update the class structure for the true state:
        #   [pn, pe, h, Va, alpha, beta, phi, theta, chi, p, q, r, Vg, wn, we, psi, gyro_bx, gyro_by, gyro_bz]
        phi, theta, psi = Quaternion2Euler(self._state[6:10])
        self.msg_true_state.pn = self._state.item(0)
        self.msg_true_state.pe = self._state.item(1)
        self.msg_true_state.h = -self._state.item(2)
        self.msg_true_state.Va = self._Va
        self.msg_true_state.alpha = self._alpha
        self.msg_true_state.beta = self._beta
        self.msg_true_state.phi = phi
        self.msg_true_state.theta = theta
        self.msg_true_state.psi = psi
        self.msg_true_state.Vg = np.sqrt(self._state.item(3)**2 + self._state.item(4)**2 + self._state.item(5)**2)
        self.msg_true_state.gamma = np.arctan2(self._Vg[2], np.sqrt(self._Vg[0]**2 + self._Vg[1]**2))
        self.msg_true_state.chi = np.arctan2(self._Vg[1], self._Vg[0])
        self.msg_true_state.p = self._state.item(10)
        self.msg_true_state.q = self._state.item(11)
        self.msg_true_state.r = self._state.item(12)
        self.msg_true_state.wn = self._wind.item(0)
        self.msg_true_state.we = self._wind.item(1)