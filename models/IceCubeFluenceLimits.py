import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pylab as pl
import pandas as pd
import scipy
from scipy.integrate import trapz, quad


#Atmospheric neutrino flux following https://arxiv.org/abs/1502.03916
#Downloaded from http://www-rccn.icrr.u-tokyo.ac.jp/mhonda/public/nflx2014/index.html
def get_atmospheric_flux(region = 'north'): 
    
    if region == 'allsky':
        
        AtmosphericHonda = np.loadtxt('data/IceCube/Honda_atmospheric.d') #all-direction averaged flux
        Eatm_GeV = AtmosphericHonda[:, 0] # energy bins
        dNdEdAdtdOmega_atm = np.sum(AtmosphericHonda[:, 1:], axis = 1) #sum for all-flavor dN/dE
        
    elif (region == 'north') or (region == 'south'): #flux averaged over northern or southern sky
        
        Eatm_GeV = np.loadtxt('data/IceCube/Honda_{}ern/0_10.d'.format(region))[:, 0]
        zenith_bins = np.arange(0, 100, 10) #(files named with implied < 0 for northern sky)
        total_flux = np.zeros(len(Eatm_GeV))
        
        for start_bin in zenith_bins: 
            #azimuth angle averaged flux
            temp_data = np.loadtxt('data/IceCube/Honda_{}ern/{}_{}.d'.format(region, start_bin, start_bin + 10)) 
            temp_flux = np.sum(temp_data[:, 1:], axis = 1)
            total_flux += temp_flux 

        dNdEdAdtdOmega_atm = total_flux/len(zenith_bins) #average over zenith bins
        Eatm_GeV = np.loadtxt('data/IceCube/Honda_{}ern/0_10.d'.format(region))[:, 0]
    
    return Eatm_GeV, dNdEdAdtdOmega_atm



#GRECO Aeff, dPsi from arXiv:2212.06810 Appendix B 
def get_GRECO_data(spec = 'Aeff', region = 'north'): 
    
    GRECO_Aeff_data = np.loadtxt('data/IceCube/GRECO_Aeff_{}ern.txt'.format(region), delimiter = ',')
    Ebins_Aeff, Aeff = GRECO_Aeff_data[:, 0], GRECO_Aeff_data[:, 1]*2  #x2 to give nu + nubar instead of (nu + nubar)/2
    
    GRECO_dPsi_data = np.loadtxt('data/IceCube/GRECO_dPsi.txt', delimiter = ',')
    Ebins_dPsi, dPsi = GRECO_dPsi_data[:, 0], GRECO_dPsi_data[:, 1] 
    
    if spec == 'Aeff': 
        return Ebins_Aeff, Aeff

    elif spec == 'dPsi': 
        return Ebins_dPsi, dPsi

    

#Upgrade Aeff, dPsi from https://icecube.wisc.edu/data-releases/2020/04/icecube-upgrade-neutrino-monte-carlo-simulation/
def get_Upgrade_data(): 
    
    flavor_files = ['nuebar_nc.csv', 'nuebar_cc.csv', 'numubar_nc.csv', 'numubar_cc.csv', 'numu_nc.csv', 'nue_cc.csv', 'nue_nc.csv', 'numu_cc.csv']
    
    Ebinedges = np.logspace(0, 2, 11) #10 energy bins on 0 - 100 GeV
    Ebins = Ebinedges[:-1] + (Ebinedges[1:] - Ebinedges[:-1])/2 #bincenters
    
    Aeff = np.zeros_like(Ebins)
    dPsi = np.zeros_like(Ebins)
    
    for file in flavor_files: 
        temp_data = pl.genfromtxt('data/IceCube/IceCubeUpgradeNeutrinoMCDataRelease/binned/'+file, names = True, delimiter = ',')
        
        temp_Aeff = temp_data['Aeff_m2']
        Aeff += temp_Aeff #all-flavor total 
        
        temp_dPsi = temp_data['dPsi_deg']
        dPsi += temp_dPsi    
    dPsi /= len(flavor_files) #flavor-averaged dPsi
    
    return Ebins, Aeff, dPsi # /2 would give (nu + nubar) average, but we care about all-flavor here



#Estimate 90% upper limit necessary for detection
#Following https://iopscience.iop.org/article/10.1088/0004-637X/719/1/900/pdf
def FluenceLimitEstimation(Emin, Emax, region = 'north', detector = 'GRECO', sindex = 2, duration = np.logspace(-4, 8, 41)):  
    
    if detector == 'Upgrade':
        Eatm_GeV, dNdEdAdtdOmega_atm = get_atmospheric_flux('allsky') #all-sky background for Upgrade
        Ebins, _Aeff, _dPsi = get_Upgrade_data()
    
    
    if detector == 'GRECO':
        Eatm_GeV, dNdEdAdtdOmega_atm = get_atmospheric_flux(region) #background averaged over northern or southern sky for GRECO
        Ebins, _Aeff = get_GRECO_data(spec = 'Aeff', region = region)
        GRECO_dPsi_Ebins, dPsi_GRECO = get_GRECO_data(spec = 'dPsi')
        _dPsi = np.interp(Ebins, GRECO_dPsi_Ebins, dPsi_GRECO, left = 0, right = 0) #cast to same energy bins as Aeff
    
    #Background only on the desired energy range
    _Eatm_GeV = Eatm_GeV[np.where((Eatm_GeV >= Emin) & (Eatm_GeV <= Emax))[0]]
    _dNdEdAdtdOmega_atm = dNdEdAdtdOmega_atm[np.where((Eatm_GeV >= Emin) & (Eatm_GeV <= Emax))[0]]
    
    #Cast Aeff, angular resolution to atmospheric energy bins
    Aeff = np.interp(_Eatm_GeV, Ebins, _Aeff, left = 0, right = 0) # m^2
    Omega = np.interp(_Eatm_GeV, Ebins, np.radians(_dPsi)**2, left = 0, right = 0)
    
    #Background rate
    Natm = trapz(_dNdEdAdtdOmega_atm * Aeff * Omega, _Eatm_GeV) * duration

    #Probability of false positive (alpha) = 0.1
    Ns_alpha = np.zeros_like(Natm)
    nalpha = 0
    for i, natm in enumerate(Natm):
        while scipy.special.gammainc(nalpha + 1, natm) >= 0.1:
            nalpha += 0.1
        Ns_alpha[i] = nalpha
    
    #Probability of false negative (beta) = 0.9
    Ns_beta = np.zeros_like(Ns_alpha)
    for i, nalpha in enumerate(Ns_alpha): 
        nbeta = scipy.special.gammaincinv(nalpha + 1, 0.9)
        Ns_beta[i] = nbeta
     
    #Number excess events necessary to exceed upper limit
    Ns_90 = Ns_beta - Natm 
    
    #Normalization for which source reaches 90% upper limit
    Ns = trapz(_Eatm_GeV ** (-sindex) * Aeff, _Eatm_GeV) #unnormalized source with dNs/dE = As E^-sindex
    Norm90 = (Ns_90 / Ns) 

    #Integrate over chosen energy range to get fluence [GeV/m^2]
    Fluence_lim = Norm90 * trapz(_Eatm_GeV*_Eatm_GeV**(-sindex), _Eatm_GeV)
    Fluence_lim *= 1e-4 # m^2 -> cm^2
    
    return duration, Fluence_lim



######### Figures ###############


#https://arxiv.org/pdf/2412.05087 Figure 1
def plot_detector_specs(): 
    
    #Upgrade
    Upgrade_Ebins, Aeff_Upgrade, dPsi_Upgrade = get_Upgrade_data()

    #GRECO
    GRECO_Aeff_Ebins, Aeff_GRECO = get_GRECO_data(spec = 'Aeff')
    GRECO_dPsi_Ebins, dPsi_GRECO = get_GRECO_data(spec = 'dPsi')
    
    #Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (30, 14), tight_layout = True)
   
    ax1.grid()
    ax1.plot(Upgrade_Ebins, Aeff_Upgrade, linewidth = 12, color = 'navy', label = r'Upgrade $\nu_{all}$') 
    ax1.plot(GRECO_Aeff_Ebins, Aeff_GRECO, linewidth = 12, color = 'seagreen', label = r'GRECO $\nu_{all}$', linestyle = 'dashed')
    ax1.plot()
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel(r"$E_{\nu,\rm{true}}$ [GeV]", fontsize = 40)
    ax1.set_ylabel(r"$A_{\rm{eff}}$ [$\rm{m^2}$]", fontsize = 40)
    ax1.set_xlim(1e0, 1e3)
    ax1.set_ylim(1e-7, 1e-1)
    ax1.tick_params(labelsize = 32)
    ax1.legend(fontsize = 32)
    
    ax2.grid()
    ax2.plot(Upgrade_Ebins, dPsi_Upgrade, linewidth = 12, color = 'navy')
    ax2.plot(GRECO_dPsi_Ebins, dPsi_GRECO, linewidth = 12, color = 'seagreen', linestyle = 'dashed')
    ax2.set_xscale("log")
    ax2.set_xlabel(r"$E_{\nu,\rm{true}}$ [GeV]", fontsize = 40)
    ax2.set_ylabel(r"$d\Psi$ [$\rm{deg}^2$]", fontsize = 40)
    ax2.tick_params(labelsize = 32)

    
    
#https://arxiv.org/pdf/2412.05087 Figure 3 sensitivities     
def plot_sensitivities(Emin, Emax): 
    
    duration = np.logspace(-4, 8, 41) #Observational time bins
    _, Upgrade_limit = FluenceLimitEstimation(Emin, Emax, detector = 'Upgrade')
    _, GRECO_limit = FluenceLimitEstimation(Emin, Emax, detector = 'GRECO')
    
    #Plotting
    plt.figure(figsize = (8,6))
    plt.plot(duration, Upgrade_limit, linewidth = 3, color = 'forestgreen', linestyle = 'dashed', label = 'IceCube-Upgrade')
    plt.plot(duration, GRECO_limit, linewidth = 3, color = 'lightcoral', linestyle = 'dashed', label = 'GRECO')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlim(1e-4, 1e8)
    plt.ylim(1e-4, 1e3)
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.legend(fontsize = 20, ncol=1, frameon=False )
    plt.xlabel(r'$\Delta \,t$ [s]', fontsize = 16)
    plt.ylabel(r'$F_\nu \, [\mathrm{GeV}\,\mathrm{cm}^{-2}]$', fontsize = 16)
    
    plt.text(2e-4, 4e2, r'$E_{\nu}$'+' = {} - {} GeV'.format(Emin, Emax), fontsize = 16)
    plt.tight_layout()