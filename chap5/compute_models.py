"""
compute_ss_model
    - Chapter 5 assignment for Beard & McLain, PUP, 2012
    - Update history:  
        2/4/2019 - RWB
"""
import sys
sys.path.append('..')
import numpy as np
from scipy.optimize import minimize
from tools.tools import Euler2Quaternion, Quaternion2Euler
from tools.transfer_function import transfer_function
import parameters.aerosonde_parameters as MAV
from parameters.simulation_parameters import ts_simulation as Ts

def compute_tf_model(mav, trim_state, trim_input):

    # euler states
    e_state = euler_state(trim_state)
    pn = e_state.item(0)
    pe = e_state.item(1)
    pd = e_state.item(2)
    u = e_state.item(3)
    v = e_state.item(4)
    w = e_state.item(5)
    phi = e_state.item(6)
    theta = e_state.item(7)
    psi = e_state.item(8)
    p = e_state.item(9)
    q = e_state.item(10)
    r = e_state.item(11)

    # control surface inputs
    delta_a_star = trim_input[0]
    delta_e_star = trim_input[1]
    delta_r_star = trim_input[2]
    delta_t_star = trim_input[3]

    # additional mav variables
    Va = mav._Va
    Va_star = mav._Va
    Vg = mav._Vg
    alpha_star = mav._alpha
    b = MAV.b
    rho = MAV.rho
    S = MAV.S_wing
    mass = MAV.mass
    c = MAV.c
    Jy = MAV.Jy
    g = MAV.gravity

    # -------------- Transfer Functions --------------
    # phi to delta_a
    a_phi_1 = -0.5*rho*Va**2*S*b*MAV.C_p_p*b/(2.0*Va)
    a_phi_2 = 0.5*rho*Va**2*S*b*MAV.C_p_delta_a
    T_phi_delta_a = transfer_function(num=np.array([[a_phi_2]]),
                                     den=np.array([[1, a_phi_1, 0]]),
                                     Ts=Ts)
    # chi to phi
    T_chi_phi = transfer_function(num=np.array([[g/Vg]]),
                                  den=np.array([[1, 0]]),
                                  Ts=Ts)

    # beta to delta_r
    a_b_1 = -rho*Va*S/(2*mass)*MAV.C_Y_beta
    a_b_2 = rho*Va*S/(2*mass)*MAV.C_Y_delta_r
    T_beta_delta_r = transfer_function(num=np.array([[a_b_2]]),
                                       den=np.array([[1, a_b_1]]),
                                       Ts=Ts)

    #theta to delta_e
    a_t_1 = -rho*Va**2*c*S/(2*Jy)*MAV.C_m_q*c/(2.0*Va)
    a_t_2 = -rho*Va**2*c*S/(2*Jy)*MAV.C_m_alpha
    a_t_3 = rho*Va**2*c*S/(2*Jy)*MAV.C_m_delta_e
    T_theta_delta_e = transfer_function(num=np.array([[a_t_3]]),
                                        den=np.array([[1, a_t_1, a_t_2]]),
                                        Ts=Ts)

    # h to theta
    T_h_theta = transfer_function(num=np.array([[Va]]),
                                  den=np.array([[1, 0]]),
                                  Ts=Ts)

    # h to Va
    T_h_Va = transfer_function(num=np.array([[theta]]),
                               den=np.array([[1, 0]]),
                               Ts=Ts)

    # Va to delta_t
    a_v_1 = rho*Va_star*S/mass*(MAV.C_D_0 + MAV.C_D_alpha*alpha_star + MAV.C_D_delta_e*delta_e_star) + rho*MAV.S_prop/mass*MAV.C_prop*Va_star
    a_v_2 = rho*MAV.S_prop/mass*MAV.C_prop*MAV.k_motor**2*delta_t_star
    a_v_3 = g
    T_Va_delta_t = transfer_function(num=np.array([[a_v_2]]),
                                     den=np.array([[1, a_v_1]]),
                                     Ts=Ts)

    # Va to theta
    T_Va_theta = transfer_function(num=np.array([[-a_v_3]]),
                                   den=np.array([[1, a_v_1]]),
                                   Ts=Ts)

    return T_phi_delta_a, T_chi_phi, T_theta_delta_e, T_h_theta, T_h_Va, T_Va_delta_t, T_Va_theta, T_beta_delta_r

def compute_ss_model(mav, trim_state, trim_input):
    # euler states
    e_state = euler_state(trim_state)
    u_star = e_state.item(3)
    v_star = e_state.item(4)
    w_star = e_state.item(5)
    phi_star = e_state.item(6)
    theta_star = e_state.item(7)
    p_star = e_state.item(9)
    q_star = e_state.item(10)
    r_star = e_state.item(11)

    # control surface inputs
    delta_a_star = trim_input[0]
    delta_e_star = trim_input[1]
    delta_r_star = trim_input[2]
    delta_t_star = trim_input[3]

    Va_star = mav._Va
    b = MAV.b
    rho = MAV.rho
    S = MAV.S_wing
    S_prop = MAV.S_prop
    C_prop = MAV.C_prop
    mass = MAV.mass
    alpha = mav._alpha

    c = MAV.c
    Jy = MAV.Jy
    g = MAV.gravity
    k = MAV.k_motor

    # Aerodynamic Coefficients
    C_D = lambda alpha: MAV.C_D_p + (MAV.C_L_0 + MAV.C_L_alpha * alpha) ** 2 / (np.pi * MAV.e * MAV.AR)
    sig = lambda alpha: (1 + np.exp(-MAV.M * (alpha - MAV.alpha0)) + np.exp(MAV.M * (alpha + MAV.alpha0))) / ((1 + np.exp(-MAV.M * (alpha - MAV.alpha0))) * (1 + np.exp(MAV.M * (alpha + MAV.alpha0))))
    C_L = lambda alpha: (1 - sig(alpha)) * (MAV.C_L_0 + MAV.C_L_alpha * alpha) + sig(alpha) * (2 * np.sign(alpha) * np.sin(alpha) ** 2 * np.cos(alpha))
    C_X_0 = lambda alpha: -C_D(alpha) * np.cos(alpha) + C_L(alpha) * np.sin(alpha)
    C_X_q = lambda alpha: -MAV.C_D_q * np.cos(alpha) + MAV.C_L_q * np.sin(alpha)
    C_X_a = lambda alpha: -MAV.C_D_alpha * np.cos(alpha) + MAV.C_L_alpha * np.sin(alpha)
    C_X_de = lambda alpha: -MAV.C_D_delta_e * np.cos(alpha) + MAV.C_L_delta_e * np.sin(alpha)
    C_Z_0 = lambda alpha: -C_D(alpha) * np.sin(alpha) - C_L(alpha) * np.cos(alpha)
    C_Z_q = lambda alpha: -MAV.C_D_q * np.sin(alpha) - MAV.C_L_q * np.cos(alpha)
    C_Z_a = lambda alpha: -MAV.C_D_alpha * np.sin(alpha) - MAV.C_L_alpha * np.cos(alpha)
    C_Z_de = lambda alpha: -MAV.C_D_delta_e * np.sin(alpha) - MAV.C_L_delta_e * np.cos(alpha)

    C_Y_p = MAV.C_Y_p
    C_Y_r = MAV.C_Y_r
    C_Y_0 = MAV.C_Y_0
    C_Y_b = MAV.C_Y_beta
    C_Y_dr = MAV.C_Y_delta_r
    C_Y_da = MAV.C_Y_delta_a
    C_p_p = MAV.C_p_p
    C_p_r = MAV.C_p_r
    C_p_0 = MAV.C_p_0
    C_p_b = MAV.C_p_beta
    C_p_dr = MAV.C_p_delta_r
    C_p_da = MAV.C_p_delta_a
    C_r_p = MAV.C_r_p
    C_r_r = MAV.C_r_r
    C_r_0 = MAV.C_r_0
    C_r_b = MAV.C_r_beta
    C_r_dr = MAV.C_r_delta_r
    C_r_da = MAV.C_r_delta_a
    C_X_0 = C_X_0(alpha)
    C_X_a = C_X_a(alpha)
    C_X_q  = C_X_q(alpha)
    C_X_de = C_X_de(alpha)
    C_Z_0 = C_Z_0(alpha)
    C_Z_a = C_Z_a(alpha)
    C_Z_q = C_Z_q(alpha)
    C_Z_de = C_Z_de(alpha)
    C_m_0 = MAV.C_m_0
    C_m_a = MAV.C_m_alpha
    C_m_q = MAV.C_m_q
    C_m_de = MAV.C_m_delta_e

    # Trim Variables
    alpha_star = mav._alpha
    beta_star = mav._beta

    # Lateral State Space Model Coefficients (Table 5.1)
    Y_v = rho*S*b*v_star/(4.0*mass*Va_star)*(C_Y_p*p_star + C_Y_r*r_star) + rho*S*v_star/mass*(C_Y_0 + C_Y_b*beta_star + C_Y_da*delta_a_star + C_Y_dr*delta_r_star) + rho*S*C_Y_b/(2.0*mass)*np.sqrt(u_star**2 + w_star**2)
    Y_p = w_star + rho*Va_star*S*b/(4.0*mass)*C_Y_p
    Y_r = -u_star + rho*Va_star*S*b/(4.0*mass)*C_Y_r
    Y_da = rho*Va_star**2*S/(4.0*mass)*C_Y_da
    Y_dr = rho*Va_star**2*S/(4.0*mass)*C_Y_dr
    L_v = rho*S*b**2*v_star/(4.0*Va_star)*(C_p_p*p_star + C_p_r*r_star) + rho*S*b*v_star*(C_p_0 + C_p_b*beta_star + C_p_da*delta_a_star + C_p_dr*delta_r_star) + rho*S*b*C_p_b/2.0*np.sqrt(u_star**2 + w_star**2)
    L_p = MAV.gamma1*q_star + rho*Va_star*S*b**2/4.0*C_p_p
    L_r = -MAV.gamma2*q_star + rho*Va_star*S*b**2/4.0*C_p_r
    L_da = rho*Va_star**2*S*b/2.0*C_p_da
    L_dr = rho*Va_star**2*S*b/2.0*C_p_dr
    N_v = rho*S*b**2*v_star/(4.0*Va_star)*(C_r_p*p_star + C_r_r*r_star) + rho*S*b*v_star*(C_r_0 + C_r_b*beta_star + C_r_da*delta_a_star + C_r_dr*delta_r_star) + rho*S*b*C_r_b/2.0*np.sqrt(u_star**2 + w_star**2)
    N_p = MAV.gamma7*q_star + rho*Va_star*S*b**2/4.0*C_r_p
    N_r = -MAV.gamma1*q_star + rho*Va_star*S*b**2/4.0*C_r_r
    N_da = rho*Va_star**2*S*b/2.0*C_r_da
    N_dr = rho*Va_star**2*S*b/2.0*C_r_dr

    # Longitudinal State Space Model Coefficients (Table 5.2)
    X_u = u_star*rho*S/mass*(C_X_0 + C_X_a*alpha_star + C_X_de*delta_e_star) - rho*S*w_star*C_X_a/(2.0*mass) + rho*S*c*C_X_q*u_star*q_star/(4.0*mass*Va_star) - rho*S_prop*C_prop*u_star/mass
    X_w = -q_star + w_star*rho*S/mass*(C_X_0 + C_X_a*alpha_star + C_X_de*delta_e_star) + rho*S*c*C_X_q*w_star*q_star/(4.0*mass*Va_star) + rho*S*C_X_a*u_star/(2.0*mass) - rho*S_prop*C_prop*w_star/mass
    X_q = -w_star + rho*Va_star**2*S*C_X_q*c/(4.0*mass)
    X_de = rho*Va_star**2*S*C_X_de/(2.0*mass)
    X_dt = rho*S_prop*C_prop*k**2*delta_t_star/mass
    Z_u = q_star + u_star*rho*S/mass*(C_Z_0 + C_Z_a*alpha_star + C_Z_de*delta_e_star) - rho*S*C_Z_a*w_star/(2.0*mass) + u_star*rho*S*C_Z_q*c*q_star/(4.0*mass*Va_star)
    Z_w = w_star*rho*S/mass*(C_Z_0 + C_Z_a*alpha_star + C_Z_de*delta_e_star) + rho*S*C_Z_a*u_star/(2.0*mass) + rho*w_star*S*c*C_Z_q*q_star/(4.0*mass*Va_star)
    Z_q = u_star + rho*Va_star*S*C_Z_q*c/(4.0*mass)
    Z_de = rho*Va_star**2*S*C_Z_de/(2.0*mass)
    M_u = u_star*rho*S*c/Jy*(C_m_0 + C_m_a*alpha_star + C_m_de*delta_e_star) - rho*S*c*C_m_a*w_star/(2.0*Jy) + rho*S*c**2*C_m_q*q_star*u_star/(4.0*Jy*Va_star)
    M_w = w_star*rho*S*c/Jy*(C_m_0 + C_m_a*alpha_star + C_m_de*delta_e_star) + rho*S*c*C_m_a*u_star/(2.0*Jy) + rho*S*c**2*C_m_q*q_star*w_star/(4.0*Jy*Va_star)
    M_q = rho*Va_star*S*c**2*C_m_q/(4.0*Jy)
    M_de = rho*Va_star**2*S*c*C_m_de/(2.0*Jy)

    A_lat = np.array([[Y_v, Y_p, Y_r, g*np.cos(theta_star)*np.cos(phi_star), 0],
                      [L_v, L_p, L_r, 0, 0],
                      [N_v, N_p, N_r, 0, 0],
                      [0, 1, np.cos(phi_star)*np.tan(theta_star), q_star*np.cos(phi_star)*np.tan(theta_star)-r_star*np.sin(phi_star)*np.tan(theta_star), 0],
                      [0, 0, np.cos(phi_star)*1/np.cos(theta_star), q_star*np.cos(phi_star)*1/np.cos(theta_star)-r_star*np.sin(phi_star)*np.tan(theta_star), 0]])
    B_lat = np.array([[Y_da, Y_dr],
                      [L_da, L_dr],
                      [N_da, N_dr],
                      [0, 0],
                      [0, 0]])
    A_lon = np.array([[X_u, X_w, X_q, -g*np.cos(theta_star)],
                      [Z_u, Z_w, Z_q, -g*np.sin(theta_star)],
                      [M_u, M_w, M_q, 0, 0],
                      [0, 0, 1, 0, 0],
                      [np.sin(theta_star), -np.cos(theta_star), 0, u_star*np.cos(theta_star) + w_star*np.sin(theta_star), 0]])
    B_lon = np.array([[X_de, X_dt],
                      [Z_de, 0],
                      [M_de, 0],
                      [0, 0],
                      [0, 0]])
    return A_lon, B_lon, A_lat, B_lat

def euler_state(x_quat):
    # convert state x with attitude represented by quaternion
    # to x_euler with attitude represented by Euler angles
    e = np.array([[x_quat[6], x_quat[7], x_quat[8], x_quat[9]]])
    [phi, theta, psi] = Quaternion2Euler(e)
    x_euler = np.array([[MAV.pn0],
                            [MAV.pe0],
                            [MAV.pd0],
                            [MAV.u0],
                            [MAV.v0],
                            [MAV.w0],
                            [phi],
                            [theta],
                            [psi],
                            [MAV.p0],
                            [MAV.q0],
                            [MAV.r0]])
    return x_euler

def quaternion_state(x_euler):
    # convert state x_euler with attitude represented by Euler angles
    # to x_quat with attitude represented by quaternions
    e = Euler2Quaternion(x_euler[6], x_euler[7], x_euler[8])
    x_quat = np.array([[MAV.pn0],  # (0)
                            [MAV.pe0],  # (1)
                            [MAV.pd0],  # (2)
                            [MAV.u0],  # (3)
                            [MAV.v0],  # (4)
                            [MAV.w0],  # (5)
                            [e[0]],  # (6)
                            [e[1]],  # (7)
                            [e[2]],  # (8)
                            [e[3]],  # (9)
                            [MAV.p0],  # (10)
                            [MAV.q0],  # (11)
                            [MAV.r0]])  # (12)
    return x_quat

def f_euler(mav, x_euler, input):
    # return 12x1 dynamics (as f state were Euler state)
    # compute f at euler_state
    # TODO
    f_euler_ = np.zeros((12, 1))
    return f_euler_

def df_dx(mav, x_euler, input):
    # take partial of f_euler with respect to
    # TODO
    A = np.zeros((12, 12))
    return A

def df_du(mav, x_euler, delta):
    # take partial of f_euler with respect to delta
    # TODO
    B = np.zeros((12, 4))
    return B

def dT_dVa(mav, Va, delta_t):
    # returns the derivative of motor thrust with respect to Va
    # TODO
    dThrust = 0
    return dThrust

def dT_ddelta_t(mav, Va, delta_t):
    # returns the derivative of motor thrust with respect to delta_t
    # TODO
    dThrust = 0
    return dThrust