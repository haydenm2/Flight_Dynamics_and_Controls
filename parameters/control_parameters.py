import sys
sys.path.append('..')
import numpy as np
# import chap5.transfer_function_coef as TF
import parameters.aerosonde_parameters as MAV

gravity = MAV.gravity
sigma = 0.05
Va0 = MAV.Va0

#----------roll loop------------- (UAV book pg. 98)
wn_phi = 10.0
zeta_phi = 0.707

a_phi_1 = -0.5*MAV.rho*Va0**2*MAV.S_wing*MAV.b*MAV.C_p_p*MAV.b/(2.0*Va0)
a_phi_2 = 0.5*MAV.rho*Va0**2*MAV.S_wing*MAV.b*MAV.C_p_delta_a
roll_kp = wn_phi**2/a_phi_2
roll_kd = (2.*zeta_phi*wn_phi - a_phi_1)/a_phi_2

#----------course loop------------- (UAV book pg. 101)
W_phi_chi = 10.0
wn_chi = wn_phi/W_phi_chi
zeta_chi = 0.707

course_kp = 2.*zeta_chi*wn_chi*Va0/gravity  #should be Vg instead of Va0
course_ki = wn_chi**2*Va0/gravity  #should be Vg instead of Va0

#----------sideslip loop------------- (UAV book pg. 103)
wn_beta = 10.0
zeta_beta = 0.707

a_b_1 = -MAV.rho*Va0*MAV.S_wing/(2*MAV.mass)*MAV.C_Y_beta
a_b_2 = MAV.rho*Va0*MAV.S_wing/(2*MAV.mass)*MAV.C_Y_delta_r
sideslip_ki = wn_beta**2/a_b_2
sideslip_kp = (2.*zeta_beta*wn_beta - a_b_1)/a_b_2

#----------yaw damper------------- (UAV book pg. ??) #TODO implement this correctly
yaw_damper_tau_r = 0.01
yaw_damper_kp = 0.01

#----------pitch loop------------- (UAV book pg. 105)
wn_theta = 10.0
zeta_theta = 0.707

a_t_1 = -MAV.rho*Va0**2*MAV.c*MAV.S_wing/(2.*MAV.Jy)*MAV.C_m_q*MAV.c/(2.0*Va0)
a_t_2 = -MAV.rho*Va0**2*MAV.c*MAV.S_wing/(2.*MAV.Jy)*MAV.C_m_alpha
a_t_3 = MAV.rho*Va0**2*MAV.c*MAV.S_wing/(2.*MAV.Jy)*MAV.C_m_delta_e
pitch_kp = (wn_theta**2 - a_t_2)/a_t_3
pitch_kd = (2.*zeta_theta*wn_theta - a_t_1)/a_t_3
K_theta_DC = pitch_kp*a_t_3/(a_t_2 + pitch_kp*a_t_3)

#----------altitude loop------------- (UAV book pg. 107)
W_h_theta = 10.0
wn_h = wn_theta/W_h_theta
zeta_h = 0.707

altitude_kp = 2.*zeta_h*wn_h/(K_theta_DC*Va0)
altitude_ki = wn_h**2/(K_theta_DC*Va0)
# altitude_zone =

#---------airspeed hold using throttle--------------- (UAV book pg. 109)
wn_v = 10.0
zeta_v = 0.707

delta_t_star = 0.78144714
delta_e_star = -0.00388821
alpha_star = MAV.alpha0
a_v_1 = MAV.rho*MAV.Va0*MAV.S_prop/MAV.mass*(MAV.C_D_0 + MAV.C_D_alpha*alpha_star + MAV.C_D_delta_e*delta_e_star) + MAV.rho*MAV.S_prop/MAV.mass*MAV.C_prop*Va0
a_v_2 = MAV.rho*MAV.S_prop/MAV.mass*MAV.C_prop*MAV.k_motor**2*delta_t_star
airspeed_throttle_kp = wn_v**2/a_v_2
airspeed_throttle_ki = (2.*zeta_v*wn_v - a_v_1)/a_v_2
