import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pylab as pl
import scipy


#Constants and conversions
day2s = 24 * 3600
eV2erg = 1.60218e-12
c = 2.9979E10 #cgs
mp = 1.67262E-24 #cgs
k = 1.380649e-16 #erg/K
eCharge = 4.803204E-10 #esu
sigma_boltzman = 5.6704E-5 #erg/cm^2/s/K^4 #cgs
mec2 = 9.1093E-28 * c**2
mec2_eV = 0.511E6
Lsun = 3.846e33 #erg/s
pc2meter = 3.086e+16
erg2GeV = 624.151


### Model and equations following arXiv:2007.15742 ###

mu_SN = 0.62 #Mean molecular weight of ionized gas of solar composition
sigma_pp = 5e-26 #Proton-proton cross-section around 1 PeV

class TimeDependentShock_Lpk_tpk():
    
    def __init__(self, L_opt_pk, t_pk_day, v_ej_km_s, sigma_Lpk, sigma_vej, kappa_opt = 0.3, epsilon_B = 0.01):

        self.L_sh_pk = L_opt_pk 
        self.t_pk = t_pk_day * day2s #s
        self.epsilon_B = epsilon_B
        self.kappa_opt = kappa_opt
        self.v_ej = v_ej_km_s * 1e5 #cm/s
      
        #uncertainties
        self.sigma_Lpk = sigma_Lpk 
        self.sigma_vej = sigma_vej * 1e5 #cm/s
        
        #Equation 1
        self.v_sh_fOmega1 = (8 * self.L_sh_pk * kappa_opt / (9 * np.pi * c * self.t_pk)) ** (1./3)
        
        
        #If velocity of ejecta greater than shock velocity
        if self.v_sh_fOmega1 < self.v_ej: 
            self.v_sh_pk_H = self.v_ej
            # f_Omega = f_Omega_min, corresponding to v_sh_pk_H
            self.f_Omega = 8 * self.L_sh_pk * kappa_opt / (9 * np.pi * c * self.t_pk) / self.v_ej ** 3
            
            #uncertainties
            self.sigma_vsh_pk = self.sigma_vej
            self.sigma_fOmega = 8 * kappa_opt / (9 * np.pi * c * self.t_pk) * np.sqrt((self.sigma_Lpk/self.v_ej**3)**2 + (self.sigma_vej * self.L_sh_pk * 3 / self.v_ej**4)**2)
        
        #Otherwise forget about v_ej
        else: 
            self.v_sh_pk_H = self.v_sh_fOmega1
            self.f_Omega = 1.
            
            #uncertainties
            self.sigma_vsh_pk = (8 * kappa_opt / (9 * np.pi * c * self.t_pk))**(1./3) * self.sigma_Lpk * self.L_sh_pk**(-2./3)/3
            self.sigma_fOmega = 0.
        
    
    #Time evolution of the shock
    def evolution(self, L_sh_norm_Func, t_end= 300 * day2s, dt = 1 * day2s):
        
        #Range of particle acceleration efficiencies
        eps_rel_range = (0.01, 0.1) 
        self.eps_rel = np.mean(eps_rel_range)
        self.sigma_eps_rel = np.std(eps_rel_range)
        
        #Time stuff 
        self.dt = dt 
        self.t_start = 1. * day2s
        self.t_bin = int( (t_end - self.t_start) / dt ) #number of bins 
        self.t = self.t_start + dt * np.arange(self.t_bin) #time in seconds
        self.t_day = self.t / day2s #time in days
        self.t_pk_bin = int( (self.t_pk - self.t_start) / self.dt ) #which bin is tpk
        
        #Shock properties
        self.L_sh = L_sh_norm_Func(self.t)*self.L_sh_pk #normalize to Lpk
        self.v_sh = self.v_sh_pk_H * np.ones_like(self.t) #Assume constant velocity
        self.Mdot_over_vw = self.L_sh * 32. / 9. / self.v_sh ** 3 #Equation 1
        
        #Radius and number density 
        self.R = np.cumsum(self.v_sh) * self.dt 
        self.n = self.Mdot_over_vw / (4 * np.pi * self.f_Omega * self.R ** 2 * mu_SN * mp)
        
        #Section 2.5
        self.tau_pp = self.n * sigma_pp * self.R #proton-proton optical depth 
        self.f_pp = 1 - np.exp(-self.tau_pp * c / self.v_sh)
        
        #Proton and neutrino luminosities
        self.L_nu = self.L_sh * 1. / 2 * self.f_pp * self.eps_rel
        
        #Uncertainties
        self.sigma_L = L_sh_norm_Func(self.t)*self.sigma_Lpk
        self.sigma_vsh = self.sigma_vsh_pk * np.ones_like(self.t)
        self.sigma_Mdot = (32./9)*np.sqrt((self.sigma_L/self.v_sh**3)**2 + (self.sigma_vsh * self.L_sh * 3 / self.v_sh**4)**2)
        self.sigma_R = np.cumsum(self.sigma_vsh) * self.dt 
        self.sigma_n = 1/(4 * np.pi * mu_SN * mp) * np.sqrt((self.sigma_Mdot/self.f_Omega/self.R**2)**2 + (self.sigma_fOmega *self.Mdot_over_vw /self.f_Omega**2/self.R**2)**2 + (self.sigma_R * 2 * self.Mdot_over_vw/self.f_Omega/self.R**3)**2)
        self.sigma_tau = sigma_pp * np.sqrt((self.sigma_n * self.R)**2 + (self.sigma_R * self.n)**2)
        self.sigma_fpp = self.sigma_tau * np.exp(-self.tau_pp)
        self.sigma_Lnu = 1. / 2 * np.sqrt((self.sigma_L * self.f_pp * self.eps_rel)**2 + (self.sigma_fpp * self.L_sh * self.eps_rel)**2 + (self.sigma_eps_rel * self.L_sh * self.f_pp)**2)


        
#Shock model applied to a specific transient type
class SN_shock():

    def __init__(self, shock_type, L_opt_pk, t_pk_day, v_ej_km_s, sigma_Lpk = 0, sigma_vej = 0, nova_mass = "Msun"):
        
        self.L_opt_pk = L_opt_pk
        self.t_pk_day = t_pk_day
        self.v_ej_km_s = v_ej_km_s
        
        self.sigma_Lpk = sigma_Lpk
        self.sigma_vej = sigma_vej

        #time to peak luminosity 
        t_rise = t_pk_day * day2s #s
        
        #For each type of transient, use a light curve "template"
        if shock_type == "SN": #supernova
            SN_file = pl.genfromtxt("data/light_curves/typical_SLSNII.txt")
            logL_opt = scipy.interpolate.interp1d(SN_file[:, 0], SN_file[:, 1]-44, fill_value="extrapolate")
        elif shock_type == "TDE": #tidal disruption event
            TDE_file = pl.genfromtxt("data/light_curves/TDE_PTF09ge.txt")
            logL_opt = scipy.interpolate.interp1d(TDE_file[:, 0], TDE_file[:, 1]-44.7, fill_value="extrapolate")
        elif shock_type == "FBOT": #fast blue optical transient
            FBOT_file = pl.genfromtxt("data/light_curves/FBOT_AT2018cow.txt")
            logL_opt = scipy.interpolate.interp1d(FBOT_file[:, 0], FBOT_file[:, 1]-43.9, fill_value="extrapolate")
        elif shock_type == "NOVA": #nova
            NOVA_file = pl.genfromtxt("data/light_curves/novae_light_curves.txt", names = True)
            magnitude = NOVA_file["magnitude"]
            absolute_magnitude = magnitude - 14.6
            log_luminosity = np.log10(Lsun*10**(0.4*(4.85 - absolute_magnitude))) 
            logL_opt = scipy.interpolate.interp1d(NOVA_file['Msun'], log_luminosity - 39.765, fill_value="extrapolate")
        elif shock_type == "LRN": #luminous red nova
            LRN_file = pl.genfromtxt("data/light_curves/V1309_Sco.txt")
            logL_opt = scipy.interpolate.interp1d(LRN_file[:, 0], np.log10(LRN_file[:, 1])-np.log10(1.171242291677622e+38), fill_value="extrapolate")
     
        self.L_optFunc = lambda t: 10. ** logL_opt( (t - t_rise) / day2s ) 
        self.shock = TimeDependentShock_Lpk_tpk(L_opt_pk, t_pk_day, v_ej_km_s, sigma_Lpk, sigma_vej)
        self.shock.evolution(self.L_optFunc)

    
    #Total energy emitted in neutrinos over observational time window
    def get_totalE(self, return_unc = False):
        
        shock = self.shock 
        
        #Define observation time to be tpk -> time when neutrino luminosity is half of peak
        tpk_bin = np.where(shock.t_day >= shock.t_pk/day2s)[0][0]
        t_half_bin = np.where(shock.L_nu >= shock.L_nu[shock.t_pk_bin]/2)[0][-1]
        trange = shock.t_day[tpk_bin:t_half_bin]
        delta_t = (trange[-1] - trange[0]) * day2s
        trange = [int(time) for time in trange]
        
        Etot = np.sum(shock.L_nu[trange]*shock.dt) #Total neutrino energy
        sigma_Etot = np.sum(shock.sigma_Lnu[trange]*shock.dt)
        
        if return_unc: 
            return sigma_Etot
        else: 
            return delta_t, Etot #erg
    
    
### Neutrino spectra ###

from aafragpy import get_cross_section, get_spectrum, get_cross_section_Kamae2006
E_thr = 4 #"threshold" energy in GeV above which Aafrag is used
# pc2meter = 3.086e+16
# erg2GeV = 624.151

#powerlaw spectrum
def powerlaw(E_p, alpha): 
    return E_p**(-alpha)


def shockpowered_E2dNdEdA(trans_type, alpha_range, Lpk_range, tpk_range, vej_range, R0_range, tobs_years = 3, Galactic = False, get_err = False):
    
    #Mean values
    Lpk = np.mean(Lpk_range)
    tpk = np.mean(tpk_range)
    vej = np.mean(vej_range)
    R0 = np.mean(R0_range)
    alpha = np.mean(alpha_range)
    
    #Uncertainties
    sigma_L = float(np.std(Lpk_range))
    sigma_v = np.std(vej_range)

    #Initiate model
    shock = SN_shock(shock_type = trans_type, L_opt_pk = Lpk, t_pk_day = tpk, v_ej_km_s = vej, sigma_Lpk = sigma_L, sigma_vej = sigma_v)

    #Estimate nearest expected event
    if Galactic: #assume flat disk of radius 20 kpc
        dmin = (R0 * tobs_years * np.pi)**(-1/2) #Gpc
        dmin *= (10**9)*(pc2meter)*(100) #cm
        sigma_R = float(np.std(R0_range)) 
        sigma_d = sigma_R * (tobs_years*np.pi)**(-1/2) * (R0**(-3/2)/2)
        sigma_d *= (10**9)*(pc2meter)*(100)
        
    else: #assume uniform volume
        dmin = (tobs_years*4*np.pi*R0/3)**(-1/3) #Gpc
        dmin *= (10**9)*(pc2meter)*(100) #cm
        sigma_R = float(np.std(R0_range))
        sigma_d = sigma_R * (tobs_years*4*np.pi/3)**(-1/3) * (R0**(-4./3)/3)
        sigma_d *= (10**9)*(pc2meter)*(100)
    
    #Energy bins
    E_nu=np.logspace(-2, 8, 130) #neutrino energy binning
    dlog_Enu = np.mean(np.log(E_nu)[1:]-np.log(E_nu)[0:-1])
    E_p = np.logspace(0, 10, 100)
    
    #Proton spectrum following a power law
    dNp_dEp = powerlaw(E_p, alpha)
    
    #Resulting neutrino spectrum
    cs_matrix_above = get_cross_section ('nu_all','p-p', E_primaries = E_p[E_p>E_thr], E_secondaries = E_nu)
    spec_above = get_spectrum(E_p[E_p > E_thr], E_nu, cs_matrix=cs_matrix_above[0], prim_spectrum = dNp_dEp[E_p>E_thr])
    cs_matrix_below = get_cross_section_Kamae2006('nu_all', E_primaries = E_p[E_p<=E_thr], E_secondaries = E_nu)
    spec_below = get_spectrum(E_p[E_p<=E_thr], E_nu, cs_matrix=cs_matrix_below[0], prim_spectrum = dNp_dEp[E_p<=E_thr])
    spec = spec_below + spec_above
    
    #Observed fluence
    delta_t, Etot = shock.get_totalE()
    Etot *= erg2GeV
    Fnu = Etot/(4*np.pi*dmin**2)
    
    #Normalize spectrum to total fluence
    norm = Fnu / np.sum((E_nu) ** 2 * spec * dlog_Enu)
    spec *= norm
    
    #Uncertainty
    if get_err: 
        sigma_E = shock.get_totalE(return_unc = True) * erg2GeV #GeV
        sigma_Fnu = np.sqrt((sigma_E/(4*np.pi*(dmin**2)))**2 + (sigma_d*Etot/(2*np.pi*(dmin**3)))**2)
        norm = sigma_Fnu / np.sum((E_nu) ** 2 * spec * dlog_Enu)
        spec *= norm

    return delta_t, E_nu, (E_nu**2) * spec
