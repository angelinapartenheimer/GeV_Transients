import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pylab as pl
import scipy
from scipy.integrate import trapz, quad
import pandas as pd
from aafragpy import get_cross_section, get_spectrum, get_cross_section_Kamae2006
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import importlib
from importlib import reload
import json


#Custom models
from models.SPT import *
import models
from models.FRB import *
from models.GRB import *
from models.IceCubeFluenceLimits import *


#Unit conversions
erg2GeV = 624.151
day2s = 24*60*60
pc2meter = 3.086e+16


#Transient data from arXiv:2007.15742 Table 1
with open('SPT_values.txt') as f: 
    data = f.read() 
shockpowered_vals = json.loads(data) 


#Integral over E2dNdE to give total fluence
def int_over_spectrum(Ebins, E2dNdE, Emin, Emax): 
    
    int_range = np.where((Emin <= Ebins) & (Ebins <= Emax))[0] #integration limits
    Ebins = Ebins[int_range]
    E2dNdE = E2dNdE[int_range]
    
    fluence = trapz(E2dNdE, np.log(Ebins)) #E2dNdE * dlogE = EdNdE * dE
    
    return fluence


#Shock-powered transients following https://arxiv.org/abs/2007.15742
def shockpowered(Emin, Emax, trans_type, log_Lpk_erg_range, tpk_day_range, vej_km_s_range, R0_range, tobs_years, alpha_range = (2., 2.7), Galactic = False):
    
    Lpk_range = 10**np.array(log_Lpk_erg_range)
    
    #Fluence
    Delta_t, Ebins, E2_spec = shockpowered_E2dNdEdA(trans_type, alpha_range, Lpk_range, tpk_day_range, vej_km_s_range, R0_range, tobs_years, Galactic)
    Fnu = int_over_spectrum(Ebins, E2_spec, Emin, Emax) #total fluence
    
    #Uncertainty
    _, Ebins, E2_err = shockpowered_E2dNdEdA(trans_type, alpha_range, Lpk_range, tpk_day_range, vej_km_s_range, R0_range, tobs_years, Galactic, get_err = True)
    norm_err = int_over_spectrum(Ebins, E2_err, Emin, Emax) #Uncertainty in total fluence
    spec_err = int_over_spectrum(Ebins, E2_spec * np.log(Ebins), Emin, Emax) #Uncertainty from spectral shape
    
    #joint error caused by uncertainty in total fluence and spectral index
    dalpha = np.std(alpha_range)
    Fnu_err = np.sqrt(norm_err**2 + (dalpha * spec_err)**2)
    uperr = np.reshape(np.array([0., Fnu_err]), (2, 1))
    
    return Delta_t, Fnu, uperr


#FRB following https://arxiv.org/abs/2008.12318
def FRB(Emin, Emax, Erad_erg_range = (1e37, 1e41), tFRB_ms_range = (0.1, 10)):
    
    #Mean values
    Erad = np.mean(Erad_erg_range)
    tFRB = np.mean(tFRB_ms_range) * 1e-3 #ms -> s
    
    #Maximum possible values
    Erad_max = np.max(Erad_erg_range)
    tFRB_max = np.max(tFRB_ms_range) * 1e-3
    
    #Fluence
    Delta_t, Ebins, E2_spec = FRB_E2dNdEdA(Erad_erg = Erad, tFRB_s = tFRB, dlogt = 0.1)
    Fnu = int_over_spectrum(Ebins, E2_spec, Emin, Emax)
    
    #Uncertainty (in this case, range of maximum/minimum values)
    _, Ebins_err, E2_maxspec = FRB_E2dNdEdA(Erad_erg = Erad_max, tFRB_s = tFRB_max, dlogt = 0.1)
    maxF = int_over_spectrum(Ebins, E2_maxspec, Emin, Emax)
    minF = 0. #FRB can have parameters st there is no neutrino emission 
    uperr = np.reshape(np.array([minF, maxF - Fnu]), (2, 1))
    
    return Delta_t, Fnu, uperr


#GRB following https://arxiv.org/abs/2210.15625
def GRB(Emin, Emax, tobs_years, model, Gamma_range = (100, 1000), Egamma_range = (1e52, 1e54), R0_range = (1.5, 2), T90 = 100):  
    
    Delta_t = T90
    
    if model == 'collision': 
        Ebins, E2_spec = GRB_E2dNdEdA(tobs_years, Gamma_range, Egamma_range, R0_range)
        Ebins_err, E2_err = GRB_E2dNdEdA_err(tobs_years, Gamma_range, Egamma_range, R0_range)
        
    elif model == 'decoupling': 
        Ebins, E2_spec = GRB_E2dNdEdA(tobs_years, Gamma_range, Egamma_range, R0_range, collision = False, T90 = T90)
        Ebins_err, E2_err = GRB_E2dNdEdA(tobs_years, Gamma_range, Egamma_range, R0_range, collision = False, T90 = T90)
     
    #Fluence
    Fnu = int_over_spectrum(Ebins, E2_spec, Emin, Emax)
    
    #Uncertainty 
    Fnu_err = int_over_spectrum(Ebins_err, E2_err, Emin, Emax)
    uperr = np.reshape(np.array([0, Fnu_err]), (2, 1))
    
    return Delta_t, Fnu, uperr


#Convert number of Galactic events/year to event rate
def Galnum2rate(min_num_year, max_num_year):
    
    Gal_area = np.pi * (20/1e6)**2 #Assume disk w/radius = 20 kpc 
    min_rate = min_num_year/Gal_area
    max_rate = max_num_year/Gal_area
    
    return min_rate, max_rate # events/Gpc^2/year



### Plotting ### 


dummy_call = plt.figure() #for some reason matplotlib only responds to formatting after a call is made

font = { 'family': 'Arial', 'weight' : 'normal', 'size'   : 40}
matplotlib.rc('font', **font)
legendfont = {'fontsize' : 36, 'frameon' : False}
matplotlib.rc('legend', **legendfont)
matplotlib.rc('lines', linewidth = 4)
matplotlib.rc('errorbar', capsize = 10)

colors = {'NOVA': 'dodgerblue', 'SN': 'orange', 'TDE': 'violet', 'FBOT': 'rosybrown', 'FRB': 'purple', 
          'GRB': 'mediumaquamarine', 'GRB_LL': 'tomato', 'LRN': 'darkred'}

markers = {'Nova': 'o', 'LRNe': 'D', 'SLSNeI': 'o', 'SLSNeII': 'D', 'SNeIIn': 'X', 'CCSNe': 'p', 'IaCSM': 's', 
           'TDE': 'o', 'FBOT': 'o', 'LumFBOT': 'D',  'FRB': 'o', 'GRB_collision': 'o', 'GRB_decoupling': 'D',
           'GRB_LL': 'X','Galactic nova': '*', 'Galactic LRNe': 'p', 'Galactic supernova': 'x'}

#only put lower arrows on some transients to avoid crowding
arrows = {'Nova': 1, 'LRNe': 1, 'SLSNeI': 1, 'SLSNeII': 0, 'SNeIIn': 0, 'CCSNe': 0, 'IaCSM': 0, 'TDE': 1, 
          'FBOT': 1, 'LumFBOT': 0,'Galactic nova': 1, 'Galactic LRNe': 1, 'Galactic supernova': 1}


#https://arxiv.org/pdf/2412.05087 Figure 3
def plot_model(obs_Emin, obs_Emax, region = 'north', tobs_years = 10, show_GRECO = True, show_Upgrade = True, show_Combined = False): 
    
    fig, ax = plt.subplots(figsize = (28, 14))
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel(r'Fluence [GeV/cm$^2$]')
    ax.set_xlabel(r'$\Delta$t [s]')
    ax.set_xlim(1e-4, 1e8)
    ax.set_ylim(1e-12, 1e4)
    ax.set_title('{} - {} GeV'.format(int(obs_Emin), int(obs_Emax)))
    
    #FRB
    FRB_dt, FRB_F, FRB_uperr = FRB(obs_Emin, obs_Emax)
    ax.scatter(FRB_dt, FRB_F, color = colors['FRB'], s = 100)
    ax.errorbar(np.float64(FRB_dt), FRB_F, yerr = FRB_uperr, color =  colors['FRB'], fmt = markers['FRB'], markersize = 20, label = 'FRB')
    ax.errorbar(np.float64(FRB_dt), FRB_F, yerr = 0.8*FRB_F, color = colors['FRB'], uplims = True) #lower arrow
    
    #GRB
    #high-luminosity
    GRB_dt, GRB_F_col, GRB_yerr_col = GRB(obs_Emin, obs_Emax, tobs_years, R0_range = (0.5, 2), model = 'collision')
    _, GRB_F_dec, GRB_yerr_dec = GRB(obs_Emin, obs_Emax, tobs_years, R0_range = (0.5, 2), model = 'decoupling')

    #low-luminosity
    GRB_dt_LL, GRB_F_LL, GRB_yerr_LL = GRB(obs_Emin, obs_Emax, tobs_years, Gamma_range = (10, 100), Egamma_range = (10**49., 10**51.), R0_range = (200, 500), model = 'decoupling', T90 = 20)
    
    ax.errorbar(GRB_dt, GRB_F_col, yerr = GRB_yerr_col, color = colors['GRB'], fmt = markers['GRB_collision'], markersize = 20, label = 'GRB collision')
    ax.errorbar(GRB_dt, GRB_F_dec, yerr = GRB_yerr_dec, color = colors['GRB'], fmt = markers['GRB_decoupling'], markersize = 20, label = 'GRB decoupling')
    
    ax.errorbar(GRB_dt, GRB_F_col, yerr = 0.7*GRB_F_col, color = colors['GRB'], uplims = True) 
    ax.errorbar(GRB_dt, GRB_F_dec, yerr = 0.7*GRB_F_dec, color = colors['GRB'], uplims = True) 
    
    ax.errorbar(GRB_dt_LL, GRB_F_LL, yerr = GRB_yerr_LL, color = colors['GRB_LL'], fmt = markers['GRB_LL'], markersize = 20, label = 'LL GRB')
    ax.errorbar(GRB_dt_LL, GRB_F_LL, yerr = 0.7*GRB_F_LL, color = colors['GRB_LL'], uplims = True)
    
    #Shockpowered
    show_sources = ['SLSNeI', 'SLSNeII', 'SNeIIn', 'CCSNe', 'IaCSM', 'Nova', 'LRNe', 'TDE', 'FBOT', 'LumFBOT', 'Galactic nova', 'Galactic LRNe', 'Galactic supernova']

    for source_name in show_sources:
        source = shockpowered_vals[source_name] 
        sourcetype = source['type']
        is_galactic = (source_name in ['Galactic nova', 'Galactic LRNe', 'Galactic supernova'])
        
        _dt, _Fnu, _yerr = shockpowered(obs_Emin, obs_Emax, source['type'], source['Lpk_log10'], source['tpk_day'], source['vej_km_s'], source['R0'], tobs_years, Galactic = is_galactic)
        
        ax.errorbar(_dt, _Fnu, _yerr, fmt = markers[source_name], markersize = 20, color = colors[sourcetype], label = source_name)
        if arrows[source_name]: 
            ax.errorbar(np.float64(_dt), _Fnu, yerr = 0.8*_Fnu, color = colors[sourcetype], uplims = True) #lower arrow    
    
    #key
    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] for h in handles]
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        
    #IceCube sensitivity  
    duration = np.logspace(-4, 8, 41) # from ms to ~year
    sindices = [1.5, 2., 2.5, 3.]
    linestyles = ['dotted', 'dashed', 'dashdot', 'solid']
    GRECO_limits = []
    Upgrade_limits = []
    Combined_limits = []
    
    for i, s in enumerate(sindices):
        
        detector_colors = []
        
        if show_GRECO: 
            _, _GRECOlim = FluenceLimitEstimation(obs_Emin, obs_Emax, region = region, detector = 'GRECO', sindex = s)
            ax.plot(duration, _GRECOlim, color = 'gray', linestyle = linestyles[i])
            GRECO_limits.append(_GRECOlim)
            detector_colors.append(Patch(facecolor = 'gray', alpha = 0.4, label = 'GRECO'))

        if show_Upgrade: 
            _, _Upgradelim = FluenceLimitEstimation(obs_Emin, obs_Emax, region = region, detector = 'Upgrade', sindex = s)
            ax.plot(duration, _Upgradelim, color = 'deepskyblue', linestyle = linestyles[i])
            Upgrade_limits.append(_Upgradelim)
            detector_colors.append(Patch(facecolor = 'deepskyblue', alpha = 0.6, label = 'IceCube-Upgrade (projected)'))
    
        if show_Combined: 
            _, _Combinedlim = FluenceLimitEstimation(obs_Emin, obs_Emax, region = region, detector = 'Combined', sindex = s)
            ax.plot(duration, _Combinedlim, color = 'lightgreen', linestyle = linestyles[i])
            Combined_limits.append(_Combinedlim)
            detector_colors.append(Patch(facecolor = 'lightgreen', alpha = 0.6, label = 'Combined GRECO + Upgrade sensitivity'))
    
    index_lines = [Line2D([0], [0], color = 'black', label = 's = 3'),
                   Line2D([0], [0], color = 'black', linestyle = 'dashdot', label = 's = 2.5'),
                   Line2D([0], [0], color = 'black', linestyle = 'dashed', label = 's = 2'),
                   Line2D([0], [0], color = 'black', linestyle = 'dotted', label = 's = 1.5')]
    
    legend3 = ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))
    legend2 = plt.legend(handles = index_lines, loc = 4)
    legend = plt.legend(handles = detector_colors, loc=2)

    plt.gca().add_artist(legend2)
    plt.gca().add_artist(legend3)
    plt.grid()
    plt.show()


plot_model(1e1, 1e3, region = "north", show_Upgrade = False, tobs_years = 10)
