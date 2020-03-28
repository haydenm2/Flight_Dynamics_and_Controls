"""
autopilot block for mavsim_python
    - Beard & McLain, PUP, 2012
    - Last Update:
        2/6/2019 - RWB
"""
import sys
import numpy as np
sys.path.append('..')
import parameters.control_parameters as AP
import parameters.aerosonde_parameters as MAV
from tools.transfer_function import transfer_function
from tools.wrap import wrap
from chap6.pid_control import pid_control, pi_control, pd_control_with_rate
from chap6.hybrid_lqr_tecs_control import lqr_control, tecs_control
from message_types.msg_state import msg_state


class autopilot:
    def __init__(self, ts_control):
        self.lqr_tecs = True
        if self.lqr_tecs:
            self.lateral_control = lqr_control(AP.A_lqr, AP.B_lqr, AP.Q, AP.R, AP.limit_lqr, ts_control)
            # self.longitudinal_control = tecs_control(tecs.A, tecs.B, tecs.C, tecs.K, tecs.x, tecs.y, tecs.u, tecs.limit, tecs.Ki, ts_control)

        # instantiate lateral controllers
        self.roll_from_aileron = pd_control_with_rate(
                        kp=AP.roll_kp,
                        kd=AP.roll_kd,
                        limit=np.radians(45))
        self.course_from_roll = pi_control(
                        kp=AP.course_kp,
                        ki=AP.course_ki,
                        Ts=ts_control,
                        limit=np.radians(30))
        self.sideslip_from_rudder = pi_control(
                        kp=AP.sideslip_kp,
                        ki=AP.sideslip_ki,
                        Ts=ts_control,
                        limit=np.radians(45))
        self.yaw_damper = transfer_function(
                        num=np.array([[AP.yaw_damper_kp, 0]]),
                        den=np.array([[1, 1/AP.yaw_damper_tau_r]]),
                        Ts=ts_control)

        # instantiate lateral controllers
        self.pitch_from_elevator = pd_control_with_rate(
                        kp=AP.pitch_kp,
                        kd=AP.pitch_kd,
                        limit=np.radians(45))
        self.altitude_from_pitch = pi_control(
                        kp=AP.altitude_kp,
                        ki=AP.altitude_ki,
                        Ts=ts_control,
                        limit=np.radians(30))
        self.airspeed_from_throttle = pi_control(
                        kp=AP.airspeed_throttle_kp,
                        ki=AP.airspeed_throttle_ki,
                        Ts=ts_control,
                        limit=1.0)
        # common
        self.commanded_state = msg_state()

    def update(self, cmd, state):
        if self.lqr_tecs:
            # LQR lateral autopilot
            x_lat = np.array([[state.beta, state.p, state.r, state.phi, state.chi]]).T
            chi_c = cmd.course_command
            chi = wrap(state.chi, cmd.course_command)
            e_I = chi - chi_c
            phi_c = 0
            x_lat[4, 0] = chi
            u_lateral = self.lateral_control.update(x_lat, e_I)
            delta_a = u_lateral.item(0)
            delta_r = u_lateral.item(1)

            # # longitudinal autopilot
            # h_c = cmd.altitude_command
            # u_longitudinal = self.longitudinal_control.update()
            # delta_e = u_longitudinal.item(0)
            # delta_t = u_longitudinal.item(1)
            # theta_c = MAV.alpha_star  # TODO?

            # longitudinal autopilot (TEMPORARY FOR TESTING LATERAL)
            h_c = cmd.altitude_command
            theta_c = self.altitude_from_pitch.update(h_c, state.h)
            delta_e = self.pitch_from_elevator.update(theta_c, state.theta, state.q)
            delta_t = self.airspeed_from_throttle.update(cmd.airspeed_command, state.Va)
        else:
            # lateral autopilot
            chi_c = wrap(cmd.course_command, state.chi)
            phi_c = cmd.phi_feedforward + self.course_from_roll.update(chi_c, state.chi)
            delta_a = self.roll_from_aileron.update(phi_c, state.phi, state.p)
            delta_r = self.yaw_damper.update(state.r)

            # longitudinal autopilot
            h_c = cmd.altitude_command
            theta_c = self.altitude_from_pitch.update(h_c, state.h)
            delta_e = self.pitch_from_elevator.update(theta_c, state.theta, state.q)
            delta_t = self.airspeed_from_throttle.update(cmd.airspeed_command, state.Va)

        # construct output and commanded states
        delta = np.array([[delta_a], [delta_e], [delta_r], [delta_t]])
        self.commanded_state.h = cmd.altitude_command
        self.commanded_state.Va = cmd.airspeed_command
        self.commanded_state.phi = phi_c
        self.commanded_state.theta = theta_c
        self.commanded_state.chi = cmd.course_command
        return delta, self.commanded_state

    def saturate(self, input, low_limit, up_limit):
        if input <= low_limit:
            output = low_limit
        elif input >= up_limit:
            output = up_limit
        else:
            output = input
        return output
