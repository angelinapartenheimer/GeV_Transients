import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pylab as pl
import scipy
from scipy.integrate import trapz, quad
import pandas as pd
from aafragpy import get_cross_section, get_spectrum, get_cross_section_Kamae2006
import json

#Custom models
from models.SPT import *
from models.FRB import *
from models.GRB import *


#Transient data from arXiv:2007.15742 Table 1
with open('SPT_values.txt') as f: 
    data = f.read()  
shockpowered_vals = json.loads(data) 


#https://arxiv.org/pdf/2412.05087 Figure 2
def plot_model_spectra(tobs_years = 3): 
    
    plt.figure(figsize = (18, 12))
    plt.xlim(1e-1, 1e6)
    plt.ylim(1e-10, 1e3)
    plt.xticks(fontsize = 40)
    plt.yticks(fontsize = 40)
    plt.xlabel(r'$E_{\nu}$ [GeV]', fontsize = 40)
    plt.ylabel(r'$E^2 dN/dEdA$ [GeV/cm$^2$]', fontsize = 40)
    
    #FRB
    _, FRB_E, FRB_spec = FRB_E2dNdEdA()
    plt.loglog(FRB_E, FRB_spec, label = 'FRB', color = 'darkcyan', linewidth = 6)

    #GRB
    GRB_E_col, GRB_spec_col = GRB_E2dNdEdA(tobs_years, Gamma_range = 550, Egamma_range = 5e53, R0_range = 1.25, collision = True)
    GRB_E_dec, GRB_spec_dec = GRB_E2dNdEdA(tobs_years, Gamma_range = 550, Egamma_range = 5e53, R0_range = 1.25, collision = False)
    GRB_LL_E, GRB_LL_spec = GRB_E2dNdEdA(tobs_years, Gamma_range = 55, Egamma_range = 5e50, R0_range = 350, collision = False, T90 = 20)
    
    plt.loglog(GRB_E_col, GRB_spec_col, label = 'GRB collision', color = 'blueviolet', linewidth = 6, linestyle = 'dashdot')
    plt.loglog(GRB_E_dec, GRB_spec_dec, label = 'GRB decoupling', color = 'darkblue', linewidth = 6, linestyle = 'dashed')
    plt.loglog(GRB_LL_E, GRB_LL_spec, label = 'Low-luminosity GRB', color = 'violet', linewidth = 6)
    
    #Shockpowered
    linestyles = ['solid', 'dashdot', 'dashed', 'dotted']
    alpha = 2.4
    
    for i, sourcetype in enumerate(['TDE', 'SLSNeI', 'FBOT', 'Nova']): 
        source = shockpowered_vals[sourcetype]
        _Lpk_range = np.power(10, source['Lpk_log10']) #erg/s
        _tpk_range = source['tpk_day'] #days
        _vej_range = source['vej_km_s'] #km/s
        _R0_range = source['R0'] #/GeV^3/year
        _, _Ebins, _spec = shockpowered_E2dNdEdA(source['type'], alpha,  _Lpk_range, _tpk_range, _vej_range, _R0_range, tobs_years = tobs_years)
        plt.loglog(_Ebins, _spec, label = sourcetype, color = 'goldenrod', linestyle = linestyles[i])

    plt.legend(loc = 1, fontsize = 30)
    plt.show()


plot_model_spectra(tobs_years = 10)