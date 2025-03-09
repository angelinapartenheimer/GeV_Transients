import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pylab as pl
import scipy
import sympy as sp

eV2erg = 1.6e-12

#cgs
m_p = 1.67e-24 #g
m_e = 9.1e-28 #g
c_speed = 3e10 #cm/s
e_charge = 4.8e-10 #statcoulomb 
sigma_pgamma = 5e-28 * 0.15 #cm^2, and the 0.15 is an implied kappa_pgamma
sigma_T = 6.65e-25 #cm^2

#not cgs
h_planck = 4.135e-15 #eV*s
E_delta_prime = 0.3e9 #eV
mpc2_eV = 938.6e6 #eV


### All equations from https://iopscience.iop.org/article/10.3847/2041-8213/abbb88/pdf ###

#FRB observables
t = sp.Symbol('t', positive=True, real=True)
fxi_3, nuGHz, tms, Erad40 = sp.symbols('f_xi_-3 nu_GHz t_ms E_rad40', positive=True, real=True) 


#Shock properties at the moment of the FRB
r_FRB = 5e12 * fxi_3 ** (-2./15) * nuGHz ** (-7./15) * tms ** (1./5) * Erad40 ** (1./3) #cm
n_FRB = 420 * fxi_3 ** (-4./15) * nuGHz ** (31/15.) * tms ** (2./5) * Erad40 ** (-1./3) #cm^-3
Gamma_FRB = 287 * fxi_3 ** (-1./15) * nuGHz ** (-7/30.) * tms ** (-2./5) * Erad40 ** (1./6) 
L_FRB = 4 * np.pi * r_FRB ** 2 * m_p * n_FRB * c_speed ** 3 * Gamma_FRB ** 4 * (2.98/2) #erg/s


# time evolution of FRB properties
t_FRB = sp.Symbol('t_FRB', positive=True, real=True)
Gamma_sh = Gamma_FRB * (t / t_FRB) ** (-3/8)
r_sh = r_FRB * (t / t_FRB) ** (1./4)
n_ext = n_FRB
L_sh = L_FRB * (t / t_FRB) ** (-1)


#Peak energy of thermal synchrotron emission
#Equation 12
gamma_m = m_p / m_e * Gamma_sh / 2 #additional boost for thermal particles in shock frame 
sigma_B_1 = sp.Symbol('sigma_-1', positive=True, real=True) #magnetic density 
B_prime = (64 * np.pi * Gamma_sh ** 2 * m_p * c_speed ** 2 * n_ext) ** 0.5 * (0.1 * sigma_B_1) ** 0.5 #magnetic field (G)
E_ph_m = h_planck * e_charge * B_prime * gamma_m **2 * Gamma_sh / (m_e * c_speed * 2 * np.pi) #thermal photon energy eps_pk (eV)


#Energy of electrons that cool on the expansion timescale 
#Equation 14
gamma_c = (6 * np.pi * m_e * c_speed / (sigma_T * Gamma_sh * B_prime ** 2 * t))
E_ph_c = h_planck * e_charge * B_prime * gamma_c **2 * Gamma_sh / (m_e * c_speed * 2 * np.pi)


# Proton energy at which Delta resonance with E_ph_m peaks
#Equation 20
eps_m_prime = E_ph_m / Gamma_sh #Paper calls this eps_pk_prime
E_thr_prime = E_delta_prime / 2 / eps_m_prime * mpc2_eV #Proton energy where peak Delta-res occurs


# photon energy density
#Equation 30
U_gamma_prime = L_sh / 2 / (4 * np.pi * r_sh **2 * c_speed * Gamma_sh**2)



def get_transition_times(Erad_erg, tFRB_s): 
    
    shock_values = {fxi_3 : 1, nuGHz : 1, tms : tFRB_s/(1e-3), Erad40 : Erad_erg/(1e40), sigma_B_1: 1, t_FRB : tFRB_s}
    
    #Transition time from interactions by thermal to nonthermal protons
    #Determined by condition mean(Ep_thermal_prime) >= E_Delta_prime
    t_nth = sp.solvers.solve(E_thr_prime / (0.5 * Gamma_sh * mpc2_eV) - 1, t) #solve for transition time
    t_nth_value = t_nth[-1].subs(shock_values).evalf() #s

    #Timescale on which electrons remain fast-cooling
    #Determined by condition eps_cool < eps_pk
    t_c = sp.solvers.solve(gamma_c - gamma_m, t)
    t_c_value = t_c[-1].subs(shock_values).evalf()
    
    return t_nth_value, t_c_value


def get_time_evolution(Erad_erg, tFRB_s):

    shock_values = {fxi_3 : 1, nuGHz : 1, tms : tFRB_s/(1e-3), Erad40 : Erad_erg/(1e40), sigma_B_1: 1, t_FRB : tFRB_s}
    
    Lsh_t = sp.lambdify(t, L_sh.subs(shock_values).evalf(), 'numpy')
    Gamma_sh_t = sp.lambdify(t, Gamma_sh.subs(shock_values).evalf(), 'numpy')
    r_sh_t = sp.lambdify(t, r_sh.subs(shock_values).evalf(), 'numpy')
    E_ph_c_t = sp.lambdify(t, E_ph_c.subs(shock_values).evalf(), 'numpy')
    E_ph_m_t = sp.lambdify(t, E_ph_m.subs(shock_values).evalf(), 'numpy')
    U_gamma_prime_t = sp.lambdify(t, U_gamma_prime.subs(shock_values).evalf(), 'numpy')

    return Lsh_t, Gamma_sh_t, r_sh_t, E_ph_c_t, E_ph_m_t, U_gamma_prime_t


#Relativistic Maxwell-Boltzmann distribution for thermal protons 
def dNdgamma_p_th(gamma_p, Gamma, eps_p_th = 0.5):
    gammap_th_mean = eps_p_th * Gamma
    Theta_th = gammap_th_mean / 3.
    return (gamma_p / Gamma)**2 * np.exp(- gamma_p / Gamma / Theta_th) / 2 / Theta_th ** 3 / Gamma


#Neutrino energy bins
energy = np.logspace(9, 14, 1000) #eV
dlogenergy = np.mean(np.log10(energy)[1:]-np.log10(energy)[0:-1])
energy_p = energy * 20.


def E_nu2dN_dEnu(t, dt, Erad_erg = 1e40, tFRB_s = 1e-3):

    t_nth_value, t_c_value = get_transition_times(Erad_erg, tFRB_s)
    Lsh_t, Gamma_sh_t, r_sh_t, E_ph_c_t, E_ph_m_t, U_gamma_prime_t = get_time_evolution(Erad_erg, tFRB_s)
    _Gamma, _r_sh, _U_gamma_prime, _Eph_m, _Eph_c = Gamma_sh_t(t), r_sh_t(t), U_gamma_prime_t(t), E_ph_m_t(t), E_ph_c_t(t)
    
    #Neutrino energy from proton at Delta resonance energy
    Enu_m = E_delta_prime * mpc2_eV / 2 / (_Eph_m / _Gamma) * _Gamma * 0.05
    Enu_c = E_delta_prime * mpc2_eV / 2 / (_Eph_c / _Gamma) * _Gamma * 0.05
    
    
    #Thermal proton spectrum E^2 dN/dE
    Ep2dN_dEp = dNdgamma_p_th(energy_p / mpc2_eV, _Gamma) * energy_p ** 2 / mpc2_eV
    Ep2dN_dEp *= Lsh_t(t) * dt * 0.5 / (np.sum(Ep2dN_dEp) * dlogenergy * np.log(10.)) #normalize to Lsh/2

    
    #Optical depth to protons
    tau_pgamma = np.zeros_like(energy_p)
    tau_pgamma_c = _U_gamma_prime / eV2erg / _Eph_m * _r_sh * (t / t_c_value)**(-1./2) * (sigma_pgamma * 0.2 / 0.3)
    tau_pgamma[np.where(energy_p > Enu_c * 20)] = tau_pgamma_c * (Enu_c / energy[np.where(energy_p > Enu_c * 20)])**(1./3)
    tau_pgamma[np.where((energy_p < Enu_c * 20) &  (energy_p > Enu_m * 20))] = tau_pgamma_c * (Enu_c / energy[np.where( (energy_p < Enu_c * 20) &  (energy_p > Enu_m * 20))])**(-0.5)
    tau_pgamma[np.where(energy_p < Enu_m * 20)] = 0.
    
    return 3./8 * Ep2dN_dEp * tau_pgamma #neutrino E^2 dN/dE as a function of time



def FRB_E2dNdEdA(Erad_erg = 1e40, tFRB_s = 1e-3, dlogt = 0.01, Distance = 10 * 3.09e21):
    
    t_nth_value, t_c_value = get_transition_times(Erad_erg, tFRB_s)
    erg2GeVcm_2 = 1./eV2erg * 1e-9 / (4 * np.pi * Distance**2)
    
    logt = np.arange(np.log10(0.1e-3), np.log10(float(t_nth_value)), dlogt)
    E_nu2dN_dEnu_ti = np.zeros_like(energy)
    
    time = 10.**logt
    dt = time * dlogt * np.log(10.)
    for idx_t, _time in enumerate(time):
        E_nu2dN_dEnu_ti += E_nu2dN_dEnu(_time, dt[idx_t], Erad_erg, tFRB_s)
        
    return t_nth_value, energy / 1e9, E_nu2dN_dEnu_ti * erg2GeVcm_2
