import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from . import analyze as _anl

def period_wavenumber(m, freq, yf, Tmin=2.5, Tmax=24, m_max=12, Nylabels=10, title='period-wavenumber',
                           savefig=False, logfft=False, vmin=None, vmax=None, newfig=False, colorbar=True, cblbl=None, cbfmt=None):
    if newfig:
        plt.figure(figsize=(7, 5))
    if logfft:
        yplt = np.log(np.abs(yf[:yf.shape[0] // 2, :]))
    else:
        yplt = np.abs(yf[:yf.shape[0] // 2, :])
    f = plt.pcolormesh(m, freq, yplt, cmap=mpl.cm.afmhot_r, vmin=vmin, vmax=vmax)
    plt.ylim(1 / Tmax, 1 / Tmin)
    ylabel_loc = np.linspace(1 / Tmax, 1 / Tmin, Nylabels)
    plt.yticks(ylabel_loc, ['{0:.1f}'.format(1 / x) for x in ylabel_loc])
    plt.xlim(-m_max, m_max)
    xticks = np.linspace(-m_max, m_max, m_max + 1)
    plt.xticks(xticks)
    plt.grid()
    plt.title(title)
    if cblbl is None:
        if logfft:
            cblbl = 'log |FFT|'
        else:
            cblbl = '|FFT|'
    if cbfmt is None:
        cbfmt = '%.1e'
    if colorbar:
        plt.colorbar(label=cblbl, format=cbfmt)
    plt.ylabel('period (yrs)')
    plt.xlabel('m (longitudinal wavenumber)')
    if savefig:
        plt.savefig(title + '.png')
    return f

def period_wavenumber_contourf(m, freq, yf, Tmin=2.5, Tmax=24, m_max=12, Nylabels=10, title='period-wavenumber',
                           savefig=False, logfft=False, vmin=None, vmax=None, newfig=False, colorbar=True, cblbl=None, cbfmt=None,
                                    over_color='black', under_color='white', extend='both'):

    if newfig:
        plt.figure(figsize=(7, 5))
    if logfft:
        yplt = np.log(np.abs(yf[:yf.shape[0] // 2, :]))
    else:
        yplt = np.abs(yf[:yf.shape[0] // 2, :])
    if vmin is None:
        vmin = 0.
    if vmax is None:
        vmax = np.max(yplt)
    contours = np.linspace(vmin, vmax, 16)

    f = plt.contourf(m, freq, yplt, contours, cmap=mpl.cm.afmhot_r, extend=extend)

    f.cmap.set_over(over_color)
    f.cmap.set_under(under_color)
    plt.ylim(1 / Tmax, 1 / Tmin)
    ylabel_loc = np.linspace(1 / Tmax, 1 / Tmin, Nylabels)
    plt.yticks(ylabel_loc, ['{0:.1f}'.format(1 / x) for x in ylabel_loc])
    plt.xlim(-m_max, m_max)
    xticks = np.linspace(-m_max, m_max, m_max + 1)
    plt.xticks(xticks)
    plt.grid()
    plt.title(title)
    if cblbl is None:
        if logfft:
            cblbl = 'log |FFT|'
        else:
            cblbl = '|FFT|'
    if cbfmt is None:
        cbfmt = '%.1e'
    if colorbar:
        plt.colorbar(f, label=cblbl, format=cbfmt)
    plt.ylabel('period (yrs)')
    plt.xlabel('m (longitudinal wavenumber)')
    if savefig:
        plt.savefig(title + '.png')
    return f


def two_pwn_contourf(m, freq, yf1, yf2, title1='title 1', title2='title 2', newfig=True, Tmin=2.5, Tmax=24, m_max=12, Nylabels=10,
                        logfft=False,over_color='black', under_color='white', extend='both', vmin=0.1, vmax=10.,
                       cblbl=None, cbar=True, savename=None, cbfmt=None):
    if newfig:
        fig = plt.figure(figsize=(10, 4))
    plt.subplot(121)
    period_wavenumber_contourf(m, freq, yf1, Tmin=Tmin, Tmax=Tmax, m_max=m_max, Nylabels=Nylabels, title=title1,
                           savefig=False, logfft=logfft, vmin=vmin, vmax=vmax, newfig=False, colorbar=False, cblbl=None, cbfmt=None,
                                    over_color=over_color, under_color=under_color, extend=extend)
    plt.subplot(122)
    f = period_wavenumber_contourf(m, freq, yf2, Tmin=Tmin, Tmax=Tmax, m_max=m_max, Nylabels=Nylabels, title=title2,
                           savefig=False, logfft=logfft, vmin=vmin, vmax=vmax, newfig=False, colorbar=False, cblbl=None, cbfmt=None,
                                    over_color=over_color, under_color=under_color, extend=extend)
    if cblbl is None:
        if logfft:
            cblbl = 'log |FFT|'
        else:
            cblbl = '|FFT|'
    if cbfmt is None:
        cbfmt = '%.1e'
    plt.tight_layout()
    if cbar:
        plt.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.75])
        fig.colorbar(f, cax=cbar_ax, label=cblbl, format=cbfmt)
    if savename:
        plt.savefig(savename)
    return fig

def vs_latitude(th, z, title=None, savename=None, newfig=True):
    lat = th-90
    if newfig:
        plt.figure(figsize=(8,4))
    plt.plot(lat, z)
    plt.xlim(-90,90)
    if title is not None:
        plt.title(title)
    plt.grid()
    plt.xlabel('degrees latitude')
    if savename:
        plt.savefig(savename)

def longitudetime(z, T, title='Longitude vs Time', newfig=False, vmin=None, vmax=None):
    T_plt = T
    ph_plt = np.linspace(0, 360, z.shape[1])
    xx, yy = np.meshgrid(ph_plt, T_plt)
    if vmax is None:
        vmax = np.max(np.abs(z))
    if vmin is None:
        vmin = -vmax
    if newfig:
        plt.figure()
    plt.pcolormesh(xx, yy, z, cmap=mpl.cm.PuOr, vmin=vmin, vmax=vmax)
    plt.xlim(0, 360)
    plt.ylim(T[0], T[-1])
    plt.xlabel('longitude (degrees)')
    plt.ylabel('Time (yr)')
    plt.title(title)
    return plt.gcf()

def correlation_contourf(phases, periods, corr, title='Correlation', newfig=False, savename=None,
                              colorbar=True, cblbl=None, cbfmt=None, vmin=None, vmax=None,
                              over_color='white', under_color='black', extend='neither', cmap='RdBu_r', real_period=None, real_phase=None, real_markersize=10):
    """ plot the reuslts of sweeping the correlation across phase and period

    :param periods:
    :param corr:
    :param title:
    :param newfig:
    :param savename:
    :param colorbar:
    :param cblbl:
    :param cbfmt:
    :param vmin:
    :param vmax:
    :param over_color:
    :param under_color:
    :param extend:
    :param cmap:
    :return:
    """
    z = np.array(corr.T)
    z = np.concatenate((z,-z), axis=1)
    phase_plt = np.linspace(0,360,len(phases)*2, endpoint=False)
    zpeak = np.max(z)
    peri,phsi = np.where(z==zpeak)
    phase_val = phase_plt[phsi[0]]
    per_val = periods[peri[0]]
    print("Peak Correlation phase={0:.1f} degrees, period={1:.2f} yrs".format(phase_val, per_val))
    if vmax is None:
        vmax = np.max(np.abs(z))
    if vmin is None:
        vmin = -vmax
    contours = np.linspace(vmin, vmax, 16)
    if newfig:
        plt.figure()
    f = plt.contourf(phase_plt, periods, z, contours, cmap=cmap, extend=extend)
    if real_period is not None and real_phase is not None:
        plt.plot(real_phase, real_period, '*', color='white', markeredgecolor='black', markersize=10)
    f.cmap.set_over(over_color)
    f.cmap.set_under(under_color)
    plt.title(title)
    plt.grid()
    if cblbl is None:
        cblbl = 'correlation'
    if cbfmt is None:
        cbfmt = '%.1f'
    if colorbar:
        plt.colorbar(f, label=cblbl, format=cbfmt)
    plt.xlabel('phase (degrees)')
    plt.ylabel('period (years)')
    if savename:
        plt.savefig(savename)
    return f

def amplitude_fit_2waves(error, amp_min, amp_max, Namps, newfig=False, title='Amplitude Fit for Two Waves',
                              savename=None, clbl=None, cfmt=None, over_color='Black', under_color='white',
                              extend='both',
                              vmin=None, vmax=None, real_amp1=None, real_amp2=None, xlbl=None, ylbl=None):
    if newfig:
        plt.figure(figsize=(6, 4))
    amp = np.linspace(amp_min, amp_max, Namps)
    if vmin is None:
        vmin = 0.
    if vmax is None:
        vmax = np.max(error)
    contours = np.linspace(vmin, vmax, 16)
    f = plt.contourf(amp, amp, error.T, contours, cmap=mpl.cm.afmhot_r, extend=extend)
    f.cmap.set_over(over_color)
    f.cmap.set_under(under_color)
    plt.ylim(amp_min, amp_max)
    plt.xlim(amp_min, amp_max)
    ticks = np.linspace(amp_min, amp_max, (amp_max - amp_min) * 2 + 1)
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.gca().set_aspect('equal', adjustable='box')
    if real_amp1 is not None and real_amp2 is not None:
        plt.plot(real_amp1, real_amp2, '*', color='white', markeredgecolor='black', markersize=20)
    if xlbl is not None:
        plt.xlabel(xlbl)
    if ylbl is not None:
        plt.ylabel(ylbl)
    if clbl is None:
        clbl = 'misfit'
    if cfmt is None:
        cfmt = '%.1f'
    plt.colorbar(f, label=clbl, format=cfmt)
    plt.title(title)
    if savename:
        plt.savefig(savename)

### Correlation post-analysis plotting

def corr_phase(SA, SV, params_wv, params_all):
    """plot the correlation with phase

    :param SA:
    :param SV:
    :param params_wv:
    :param params_all:
    :return:
    """
    l, m, per, vthm, phase, dth = params_wv
    saph = SA[l, _anl.get_mi(m), _anl.get_peri(per), 0, :, _anl.get_dthi(dth)]
    svph = SV[l, _anl.get_mi(m), _anl.get_peri(per), 0, :, _anl.get_dthi(dth)]
    phaseplt = params_all[4]
    plt.plot(phaseplt, saph, label='SA')
    plt.plot(phaseplt, svph, label='SV')
    plt.grid()
    plt.title('phase for l={}, m={}, per={}, dth={}'.format(l, m, per, dth))
    plt.legend(loc=0)
    plt.xlabel('phase (deg)')
    plt.ylabel('correlation')

def corr_dth(SA, SV, params_wv, params_all):
    l, m, per, vthm, phase, dth = params_wv
    saph = SA[l, _anl.get_mi(m), _anl.get_peri(per), 0, _anl.get_phsei(phase), :]
    svph = SV[l, _anl.get_mi(m), _anl.get_peri(per), 0, _anl.get_phsei(phase), :]
    delth_plt = params_all[5]
    plt.plot(delth_plt, saph, label='SA')
    plt.plot(delth_plt, svph, label='SV')
    plt.grid()
    plt.title('delth for l={}, m={}, per={}, phase={}'.format(l, m, per, phase))
    plt.legend(loc=0)
    plt.xlabel('delta_th (deg)')
    plt.ylabel('correlation')

def corr_mvper(S, params_all, l=0, phase=None, dth=5, title='', vlims=(0, 0.12), nozero=True):
    if phase == 'max' or phase is None:
        Splt = np.max(S[l, :, :, 0, :, _anl.get_dthi(dth)], axis=-1)
        phase = 'max'
    else:
        Splt = S[l, :, :, 0, _anl.get_phsei(phase), _anl.get_dthi(dth)]
    ms = params_all[1]
    pers = params_all[2]
    mplt = np.linspace(ms[0] - .5, ms[-1] + .5, len(ms) + 1)
    perplt = np.linspace(pers[0] - .5, pers[-1] + .5, len(pers) + 1)
    if nozero:
        mplt = np.linspace(-0.5, len(ms) - 1.5, len(ms))
        Splt = np.vstack((Splt[:_anl.get_mi(0), :], Splt[_anl.get_mi(0) + 1:, :]))
    plt.pcolormesh(mplt, perplt, Splt.T, vmin=vlims[0], vmax=vlims[1], cmap=plt.cm.jet)
    if nozero:
        mlbls = list(range(-11, 0, 2)) + list(range(1, 12, 2))
        mpltlocs = list(range(1, 12, 2)) + list(range(12, 23, 2))
        plt.xticks(mpltlocs, mlbls)
        plt.xlim(.5, )
    plt.grid()
    plt.colorbar()
    plt.xlabel('m')
    plt.ylabel('period (yr)')
    plt.title(title + ' l={}, dth={}, phase={}'.format(l, dth, phase))

def corr_pervm_all(SA, SV, params_all, l, title='', save=False):
    Ni = len(params_all[-1])
    plt.figure(figsize=(14, 4 * Ni))
    delta_ths = params_all[5]
    for i in range(Ni):
        dth = delta_ths[i]
        plt.subplot(Ni, 2, 2 * i + 1)
        corr_mvper(SA, params_all, l=l, dth=dth, title='SAr ' + title)
        plt.subplot(Ni, 2, 2 * i + 2)
        corr_mvper(SV, params_all, l=l, dth=dth, title='SVr ' + title)
    plt.tight_layout()
    if save:
        plt.savefig('corr_Tvm_' + title + '_l{}.png'.format(l))

def corr_phase_dth(SA, SV, params_wv, params_all):
    plt.figure(figsize=(14, 4))
    plt.subplot(121)
    corr_phase(SA, SV, params_wv, params_all=params_all)
    plt.ylim(-.12, 0.12)
    plt.subplot(122)
    corr_dth(SA, SV, params_wv, params_all=params_all)
    plt.ylim(0., 0.12)
