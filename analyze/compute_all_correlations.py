#!/usr/bin/python3
import numpy as np
import coremagmodels as cm
import coreflows as cf
sf = cf.advect.SteadyFlow()
wv = cf.advect.Waves()
anl = cf.analyze
import dill
import os
import sys

delta_ths = [8]
wt_type = 'emp'
wt_dth = 10
ls = [0,1]
outdir = 'corrs'
l_max = 14
Nth = 60
T_start = 2001
T_end = 2016

if len(sys.argv) > 1:
    for i,arg in enumerate(sys.argv):
        if arg[0] == '-':
            if arg[1:] == 'dth':
                delta_ths = [float(x) for x in sys.argv[i+1].split(',')]
                print('dths={}'.format(delta_ths))
            elif arg[1:] == 'wt':
                wt_type = sys.argv[i+1]
                print('wt='+wt_type)
            elif arg[1:] == 'wtdth':
                wt_dth = float(sys.argv[i+1])
                print('wtdth={}'.format(wt_dth))
            elif arg[1:] == 'ls':
                ls = [int(x) for x in sys.argv[i+1].split(',')]
                print('ls={}'.format(ls))
            elif arg[1:] == 'outdir':
                outdir = sys.argv[i+1]
                print('outdir={}'.format(outdir))
            elif arg[1:] == 'lmax':
                l_max = sys.argv[i+1]
                print('l_max={}'.format(l_max))
            elif arg[1:] == 'Nth':
                Nth = int(sys.argv[i+1])
                print('Nth={}'.format(Nth))
            elif arg[1:] == 'ts':
                T_start = float(sys.argv[i+1])
                print('T_start={}'.format(T_start))
            elif arg[1:] == 'te':
                T_end = float(sys.argv[i+1])
                print('T_end={}'.format(T_end))

filedir = os.path.dirname(os.path.abspath(__file__))
datadir = filedir+'/../coreflows/data/'
if not os.path.exists(filedir+'/'+outdir):
    os.makedirs(filedir+'/'+outdir)

# Import wave fits
data = dill.load(open(datadir+'wavefits012.p','rb'))
c012 = data['c012']
f012 = data['f012']

# Import Steady Flow Fit
th_sf,ph_sf,vth_sf,vph_sf = sf.import_fortran_flow_DH(datadir+'steady_flow_fortran_fit')
vth_sfSH = sf.v2vSH(vth_sf)
vph_sfSH = sf.v2vSH(vph_sf)

# Import magnetic model
magmod = cm.models.Chaos6()


th, ph = magmod.get_thvec_phvec_DH(l_max=l_max)
Nt = (T_end-T_start)*3
T = np.linspace(T_start, T_end, Nt)


# compute magnetic field data over timeframe
B_lmax = 14

th, ph = magmod.get_thvec_phvec_DH(Nth)

Bsh = magmod.get_sht_allT(T, l_max = B_lmax)
B = magmod.B_sht_allT(Bsh, Nth=Nth, l_max=B_lmax)
_, dthB, dphB = magmod.gradB_sht_allT(Bsh, Nth=Nth, l_max=B_lmax)

SVsh = magmod.get_SVsht_allT(T, l_max=B_lmax)
SV = magmod.B_sht_allT(SVsh, Nth=Nth, l_max=B_lmax)
_, dthSV, dphSV = magmod.gradB_sht_allT(SVsh, Nth=Nth, l_max=B_lmax)

SAsh = magmod.get_SAsht_allT(T, l_max=B_lmax)
SA = magmod.B_sht_allT(SAsh, Nth=Nth, l_max=B_lmax)
_, dthSA, dphSA = magmod.gradB_sht_allT(SAsh, Nth=Nth, l_max=B_lmax)


# Compute Residual SV 
steadyflow_lmax = 14
SV_steadyflow = sf.SV_steadyflow_allT(vth_sfSH, vph_sfSH, Bsh, magmodel=magmod, Nth=Nth, B_lmax=B_lmax, v_lmax=steadyflow_lmax)
SV_resid = SV-SV_steadyflow

SA_steadyflow = sf.SA_steadyflow_allT(vth_sfSH, vph_sfSH, SVsh, magmodel=magmod, Nth=Nth, B_lmax=B_lmax, v_lmax=steadyflow_lmax)
SA_resid = SA- SA_steadyflow

## Compute Many Correlations

Nphase = 18
period_min = 3
period_max = 15
Nperiod = (period_max-period_min)+1

if len(sys.argv) > 1:
    for i,arg in enumerate(sys.argv[1:]):
        if arg[0] == '-':
            if arg[1:] == 'test':
                Nphase = 2
                Nperiod = 2

phases = np.linspace(0, 180, Nphase, endpoint=False)
periods = np.linspace(period_min, period_max, Nperiod, endpoint=False)
if wt_type == 'const_sym':
    corr_wt = cf.functions.hermite((th-90)/wt_dth, 0)
elif wt_type == 'sq':
    corr_wt = cf.functions.square((th-90), wt_dth)
for delta_th in delta_ths:
    print('delta_th={}'.format(delta_th))
    if wt_type == 'sym':
        corr_wt = cf.functions.hermite((th-90)/delta_th,0)
    for m in range(-12,12):
        print('m={}'.format(m))
        for l in ls:
            if wt_type == 'emp':
                corr_wt = cf.functions.empirical_wavepower((th-90), delta_th, l)
            filename = filedir+'/'+outdir+'/l{}m{}dth{}.m'.format(l,m,delta_th)
            if not os.path.isfile(filename):
                wave_params = (l,m,np.nan,1.,np.nan,delta_th)
                SASV_from_phaseperiod = wv.make_SASV_from_phaseperiod_wave_function(wave_params, T, c012, Nth,
                                                                         B=B, dthB=dthB, dphB=dphB, 
                                                                          SV=SV, dthSV=dthSV, dphSV=dphSV)

                SAcorr, SVcorr = cf.analyze.sweep_SASVcrosscorr(phases, periods, T, SA_resid, SV_resid, SASV_from_phaseperiod, 
                                                                weights=corr_wt)
                dill.dump((SAcorr,SVcorr),open(filename, 'wb'))
                