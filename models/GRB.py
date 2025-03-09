import matplotlib.pyplot as plt
import numpy as np
from numpy import sqrt, log10
from aafragpy import get_cross_section, get_spectrum, get_cross_section_Kamae2006
import pylab as pl


eV2erg = 1.6e-12
GeV2erg = eV2erg * (1e9)
pc2meter = 3.086e+16

mp = 1.67e-24 #g
c_cgs = 3e10 #cm/s
mpc2_eV = 938.6e6 #eV
mpc2 = 0.9386 #GeV


E_thr = 4  # "threshold" energy in GeV above which Aafrag is used
z = 0.05  # approximation for nearest GRB (used in spectrum estimate) 
E_nu=np.logspace(-5, 4, 130) # neutrino energy binning


def Maxwell_Boltzmann(E_p): #Thermal distribution 
    kT = 2 * mpc2 / 3
    spec = 2 * np.sqrt(E_p/np.pi) * (1/kT)**(3/2)*np.exp(-E_p/kT)
    return spec


def monoenergetic(E_p, mean_energy = mpc2): #Narrow logparabola to represent monoenergetic proton spectrum
    piv = mean_energy
    beta = 1000
    spec = np.power(E_p/piv,- beta*np.log(E_p/piv))
    return spec 


### Equations from https://arxiv.org/pdf/2210.15625 ###


#Collisional model 
def collision_spectrum(Gamma, Egamma, dL_cm, get_err = False, sigma_G = None, sigma_E = None, sigma_d = None):
    
    E_p = np.logspace(-4, 3, 100) 
    dNp_dEp = Maxwell_Boltzmann(E_p)

    # standard pp cross section
    cs_matrix_above = get_cross_section ('nu_all','p-p', E_primaries = E_p[E_p>E_thr], E_secondaries = E_nu)
    spec_above = get_spectrum(E_p[E_p > E_thr], E_nu, cs_matrix=cs_matrix_above[0], prim_spectrum = dNp_dEp[E_p>E_thr])
    cs_matrix_below = get_cross_section_Kamae2006('nu_all', E_primaries = E_p[E_p<=E_thr], E_secondaries = E_nu)
    spec_below = get_spectrum(E_p[E_p<=E_thr], E_nu, cs_matrix=cs_matrix_below[0], prim_spectrum = dNp_dEp[E_p<=E_thr])
    spec = spec_below + spec_above

    #Boosting and normalization
    Gamma_rel = 2
    tau_pn = 1
    xi_N_range = (3, 30) #nucleon loading factor
    xi_N = np.mean(xi_N_range)
    sigma_xi = np.std(xi_N_range)
    
    diff_flux = spec * (E_nu**2) #E^2 dN/dE
    ref_Enu = E_nu[np.where(diff_flux == np.max(diff_flux))[0]] #Spectral peak
    Enu_peak = 0.1 * Gamma * Gamma_rel * mpc2 / (1+z) #Expected spectral peak (from Murase 2022)
    
    #Shift spectral peak to expected peak energy (boosting)
    Enu_boosted = E_nu * (Enu_peak/ref_Enu) 
    dlog_Enu = np.mean(np.log(Enu_boosted)[1:]-np.log(Enu_boosted)[0:-1])
    
    #Neutrino expected fluence; x3 because Murase (2022) predicts per-flavor
    nu_fluence = 3 * (1/12) * (1 + z) * tau_pn * xi_N * Egamma / (4*np.pi*dL_cm**2) # erg / cm^2
    nu_fluence /= GeV2erg
    norm = nu_fluence / np.sum((Enu_boosted) ** 2 * spec * dlog_Enu)
    spec *= norm
    
    #Uncertainty
    if get_err: 
        Fnu_err =  (3 * (1/12) * (1 + z) * tau_pn /  (4*np.pi)) * np.sqrt((sigma_d * Egamma * xi_N * 2 * dL_cm**(-3))**2 + (sigma_E * xi_N * dL_cm**(-2))**2 + (sigma_xi * Egamma * dL_cm **(-2))**2)
        Fnu_err /= GeV2erg
        norm = Fnu_err / np.sum((Enu_boosted) ** 2 * spec * dlog_Enu)
        spec *= norm
    
    return Enu_boosted, spec



#Decoupling model
def decoupling_spectrum(Gamma, Egamma, dL_cm, get_err = False, sigma_G = None, sigma_E = None, sigma_d = None, duration = 100):

    E_p = np.logspace(-4, 3, 100)
    dNp_dEp = monoenergetic(E_p)

    # standard pp cross section
    cs_matrix_above = get_cross_section ('nu_all','p-p', E_primaries = E_p[E_p>E_thr], E_secondaries = E_nu)
    spec_above = get_spectrum(E_p[E_p > E_thr], E_nu, cs_matrix=cs_matrix_above[0], prim_spectrum = dNp_dEp[E_p>E_thr])
    cs_matrix_below = get_cross_section_Kamae2006('nu_all', E_primaries = E_p[E_p<=E_thr], E_secondaries = E_nu)
    spec_below = get_spectrum(E_p[E_p<=E_thr], E_nu, cs_matrix=cs_matrix_below[0], prim_spectrum = dNp_dEp[E_p<=E_thr])
    spec = spec_below + spec_above
    
    # boosting and normalization
    zeta = 1
    sigma_np = 3e-26 #cm^2
    Gammastar = 10 
    Rstar = 1e11 #cm
    
    xi_N_range = (3, 30) #nucleon loading factor
    xi_N = np.mean(xi_N_range)
    sigma_xi = np.std(xi_N_range)
    
    #proton energy and luminosity 
    Eproton = xi_N * Egamma
    Lproton = Eproton/duration
    
    #decoupling Lorentz factor
    #this is just Equation 7 rewritten and with the constants multiplied out
    Gamma_n_dec = (3/4)*((Lproton*sigma_np*Gammastar)/(4*np.pi*Rstar*Gamma*mp*c_cgs**3))**(1./3) 
    
    diff_flux = spec * (E_nu**2) #E^2 dN/dE
    ref_Enu = E_nu[np.where(diff_flux == np.max(diff_flux))[0]] #Spectral peak
    Enu_peak = 0.1 * Gamma_n_dec * mpc2 / (1+z) #Expected spectral peak (from Murase 2022)
    
    #shift spectral peak to expected peak energy (boosting)
    Enu_boosted = E_nu * (Enu_peak/ref_Enu) 
    dlog_Enu = np.mean(np.log(Enu_boosted)[1:]-np.log(Enu_boosted)[0:-1])
    
    #Neutrino expected fluence, x3 because Murase (2022) predicts per-flavor
    nu_fluence = 3 * (1/12) * (1 + z) * zeta * (Gamma_n_dec/Gamma) * xi_N * Egamma / (4*np.pi*dL_cm**2) #erg/cm^2
    nu_fluence /= GeV2erg
    norm = nu_fluence / np.sum((Enu_boosted) ** 2 * spec * dlog_Enu)
    spec *= norm
    
    #Uncertainty
    if get_err: 
        sigma_Ep = np.sqrt((sigma_xi * Egamma)**2 + (sigma_E * xi_N)**2)
        sigma_Lp = sigma_Ep/duration
        
        sigma_Gamma_nd = (3/4)*(sigma_np*Gammastar/(4*np.pi*Rstar*mp*c_cgs**3))**(1./3) * np.sqrt((sigma_Lp * Lproton**(-2./3) * Gamma**(-1./3)/3)**2 + (sigma_G * Gamma **(-4./3) * Lproton**(1./3)/3)**2)
        
        Fnu_err = (3 * (1/12) * (1 + z) * zeta / (4*np.pi)) * np.sqrt((sigma_d * xi_N * Egamma * Gamma_n_dec / Gamma * 2 * dL_cm**(-3))**2 + (sigma_G * Egamma * xi_N * Gamma_n_dec * dL_cm**(-2) * Gamma**(-2))**2 + (sigma_Gamma_nd * Egamma * xi_N / Gamma * dL_cm**(-2))** 2 + (sigma_xi * Egamma * Gamma_n_dec / Gamma * dL_cm**(-2))** 2 + (sigma_E * xi_N * Gamma_n_dec / Gamma * dL_cm**(-2))**2)
        
        Fnu_err /= GeV2erg
        norm = Fnu_err / np.sum((Enu_boosted) ** 2 * spec * dlog_Enu)
        spec *= norm
    
    return Enu_boosted, spec

    
    
#Fluence estimate on a given energy interval
def GRB_E2dNdEdA(tobs_years, Gamma_range = (100, 1000), Egamma_range = (1e52, 1e54), R0_range = (0.5, 2), collision = True, T90 = 100): 
    
    R0 = np.mean(R0_range)
    Gamma = np.mean(Gamma_range)
    Egamma = np.mean(Egamma_range)
    
    #nearest expected event
    dmin = (tobs_years*4*np.pi*R0/3)**(-1/3) #Gpc
    dmin *= (10**9)*(pc2meter)*(100) #cm
    
    if collision: 
        Ebins, spectrum = collision_spectrum(Gamma, Egamma, dmin)
    else: 
        Ebins, spectrum = decoupling_spectrum(Gamma, Egamma, dmin, duration = T90)
    
    return Ebins, Ebins**2 * spectrum 



#Uncertainty in fluence estimate
def GRB_E2dNdEdA_err(tobs_years, Gamma_range = (100, 1000), Egamma_range = (1e52, 1e54), R0_range = (0.5, 2), collision = True, T90 = 100):

    R0 = np.mean(R0_range)
    Gamma = np.mean(Gamma_range)
    Egamma = np.mean(Egamma_range)
    
    dmin = (tobs_years*4*np.pi*R0/3)**(-1/3) #Gpc
    dmin *= (10**9)*(pc2meter)*(100) #cm
    
    sigG = np.std(Gamma_range)
    sigE = np.std(Egamma_range)
    sigR = np.std(R0_range)
    sigd = sigR * (tobs_years*4*np.pi/3)**(-1/3) * (R0**(-4./3)/3)
    sigd *= (10**9)*(pc2meter)*(100)
    
    if collision: 
        Ebins, err_spec = collision_spectrum(Gamma, Egamma, dmin, get_err = True, sigma_G = sigG, sigma_E = sigE, sigma_d = sigd)
    else: 
        Ebins, err_spec = decoupling_spectrum(Gamma, Egamma, dmin, get_err = True, sigma_G = sigG, sigma_E = sigE, sigma_d = sigd, duration = T90)
    
    return Ebins, Ebins**2 * err_spec