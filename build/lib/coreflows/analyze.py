import pyshtools as _sht
import numpy as _np
import coremagmodels as _cm
import itertools as _it
import scipy.optimize as _op
import dill as _dill


### Helper functions

def _convert_weights(f, weights):
    '''converts weights to the correct dimensions for f on a sphere'''
    Nth = f.shape[1]
    dtt = _np.pi / Nth
    tt = _np.linspace(dtt / 2, _np.pi - dtt / 2, Nth)

    if len(weights.shape)==1:
        if weights.shape[0] == 1:
            wt = weights[:, None, None]
        elif weights.shape[0] == f.shape[1]:
            wt = weights[None, :, None]
        elif weights.shape[0] == f.shape[0]:
            wt = weights[:,None, None]
        else:
            raise ValueError('weights wrong shape')
    elif len(weights.shape)==2:
            if weights.shape == f.shape[:2]:
                wt = weights[:,:,None]
            else:
                raise ValueError('weights wrong shape')
    else:
        raise ValueError('weights wrong shape')
    return _np.sin(tt)[None,:,None]*wt

def standard_deviation(f, weights=_np.ones((1)), fwm=None):
    ''' computes the standard deviation on a sphere of f

    :param f:
    :param weights:
    :return:
    '''
    wt = _convert_weights(f,weights)
    fw = f*wt
    if fwm is None:
        fwm = _np.mean(fw)
    return _np.sqrt(_np.sum((fw - fwm) ** 2))

def cross_correlation(f, g, weights=_np.ones((1)), gw=None, fw=None, fwm=None, gwm=None, fwsd=None, gwsd=None, swt=None):
    ''' compute cross-correlation of f and g on a sphere with weight function across theta

    :param f:
    :param g:
    :param weights:
    :return:
    '''
    if swt is None:
        swt = _convert_weights(f,weights)
    if fw is None:
        fw = f * swt
    if gw is None:
        gw = g * swt
    if fwm is None:
        fwm = _np.mean(fw)
    if gwm is None:
        gwm = _np.mean(gw)
    if fwsd is None:
        fwsd = standard_deviation(fw, fwm=fwm)
    if gwsd is None:
        gwsd = standard_deviation(gw, fwm=gwm)
    cross_cov = _np.sum((fw - fwm) * (gw - gwm))
    cross_corr = cross_cov / (fwsd * gwsd)
    return cross_corr

def convolve(f, g, T, th=None, ph=None, R=3480., thmax=None, weights=1.):
    '''
    Convolve functions f and g with shapes (i_time, i_th, i_ph) where i_ph = 2*i_th

    Parameters
    ----------
    f: dim (N time, N th, N ph)
    g: dim (N time, N th, N ph)
    T: array of time coords
    th: array of theta coords
    dph: spacing of ph coords
    R: radius [m]
    thmax: maximum latitude off the equator in degrees
    Returns
    -------

    '''
    raise DeprecationWarning
    dt = T[1] - T[0]
    n = f.shape[1]
    if th is None:
        th = _np.linspace(0, _np.pi, n, endpoint=False)
        dth = _np.pi / n
    else:
        dth = th[1] - th[0]

    if thmax is not None:
        min_ind = _np.where((90 - thmax) * _np.pi / 180 < th)[0][0]
        max_ind = _np.where((90 + thmax) * _np.pi / 180 > th)[0][-1]
        f = f[:,min_ind:max_ind+1,:]
        g = g[:,min_ind:max_ind+1,:]
        th = th[min_ind:max_ind+1]

    if ph is None:
        dph = 2 * _np.pi / (2 * n)
    else:
        dph = ph[1]-ph[0]
    fg = (f * g)
    diff = weights*_np.sin(th) * dth * dph * R ** 2 * dt
    conv = _np.sum(_np.tensordot(fg, diff, axes=(1, 0)))/(4*_np.pi*R**2*(T[-1]-T[0]))
    return conv

def sweep_convolution(data, fit_fun, phases, periods, T, th, ph, thmax=None):
    '''Sweeps convolution across provided array of phases and periods

    :param data:
    :param fit_fun:
    :param phases:
    :param periods:
    :param T:
    :param th:
    :param ph:
    :param thmax:
    :return:
    '''
    raise DeprecationWarning
    power = _np.zeros((len(phases), len(periods)))
    for i, p in enumerate(phases):
        for j, t in enumerate(periods):
            power[i, j] = convolve(data, fit_fun(p,t), T, th, ph=ph, thmax=thmax)
    return power

def rms_region(z, lat=None, lon=None, weights=1., R=3480e3, axis=None):
    '''
    find the rms of z in region of thmin - thmax, pmin-pmax measured in degrees latitude and longitude
    z: data of size len(th), len(ph)
    th: location in degrees latitude (evenly spaced)
    ph: location in degrees longitude (evenly spaced)
    '''
    if lat is None:
        Nth = z.shape[0]
        dth = 180/Nth
        lat = _np.linspace(-90+dth/2,90-dth/2,Nth)
    else:
        dth = (lat[1] - lat[0]) * _np.pi / 180
    if lon is None:
        nph = z.shape[1]
        dph = 360/nph
        lon = _np.linspace(-180+dph/2,180-dph/2,nph)
    else:
        dph = (lon[1] - lon[0]) * _np.pi / 180
    colat = lat + 90
    pp, tt = _np.meshgrid(lon, colat)
    area = _np.sum(_np.abs(weights) * _np.sin(tt * _np.pi / 180) * dth * dph * R ** 2, axis=axis)
    return _np.sqrt(_np.sum(_np.abs(z)**2 * weights * _np.sin(tt * _np.pi / 180) * dth * dph * R ** 2, axis=axis) / area)

def rms_region_allT(z, lat=None, lon=None, weights=1., R=3480e3, axis=None):
    '''find the rms of z in region of thmin - thmax, pmin-pmax measured in degrees latitude and longitude

    z: data of size len(th), len(ph)
    th: location in degrees latitude (evenly spaced)
    ph: location in degrees longitude (evenly spaced)
    '''
    if lat is None:
        Nth = z.shape[1]
        dth = 180/Nth
        lat = _np.linspace(-90+dth/2,90-dth/2,Nth)
    else:
        dth = (lat[1] - lat[0]) * _np.pi / 180
    if lon is None:
        nph = z.shape[2]
        dph = 360/nph
        lon = _np.linspace(-180+dph/2,180-dph/2,nph)
    else:
        dph = (lon[1] - lon[0]) * _np.pi / 180
    colat = lat + 90
    pp, tt = _np.meshgrid(lon, colat)
    if type(weights) is _np.ndarray:
        if len(weights.shape) == 2:
            if weights.shape == z.shape[1:]:
                wt = weights
            elif weights.shape[0] == z.shape[1] and weights.shape[1] == 1:
                wt = weights.repeat(z.shape[2], axis=1)
            else:
                raise TypeError('weight wrong shape')
        elif len(weights.shape) == 1:
            if weights.shape[0] == z.shape[1]:
                wt = weights[:,_np.newaxis].repeat(z.shape[2], axis=1)
            else:
                raise TypeError('weight wrong shape')
        else:
            raise TypeError('weight wrong shape')
    elif type(weights) is float:
        wt = weights

    area = _np.sum(_np.abs(wt) * _np.sin(tt * _np.pi / 180) * dth * dph * R ** 2, axis=axis)
    z_rms = 0
    for i in range(z.shape[0]):
        z_rms += _np.sqrt(_np.sum(_np.abs(z[i,:,:])**2 * wt * _np.sin(tt * _np.pi / 180) * dth * dph * R ** 2, axis=axis) / area)
    return z_rms/z.shape[0]

def weighted_mean_region(z, lat=None, lon=None, weights=1., R=3480e3, axis=None):
    '''
    find the mean of z in region of thmin - thmax, pmin-pmax measured in degrees latitude and longitude

    z: data of size len(th), len(ph)
    th: location in degrees latitude (evenly spaced)
    ph: location in degrees longitude (evenly spaced)
    '''
    if lat is None:
        Nth = z.shape[0]
        dth = 180/Nth
        lat = _np.linspace(-90+dth/2,90-dth/2,Nth)
    else:
        dth = (lat[1] - lat[0]) * _np.pi / 180
    if lon is None:
        nph = z.shape[1]
        dph = 360/nph
        lon = _np.linspace(-180+dph/2,180-dph/2,nph)
    else:
        dph = (lon[1] - lon[0]) * _np.pi / 180
    colat = lat + 90
    pp, tt = _np.meshgrid(lon, colat)
    area = _np.sum(_np.abs(weights) * _np.sin(tt * _np.pi / 180) * dth * dph * R ** 2, axis=axis)
    return _np.sum(z * weights * _np.sin(tt * _np.pi / 180) * dth * dph * R ** 2, axis=axis) / area

def weighted_mean_region_allT(z, th=None, ph=None, weights=1., R=3480e3, axis=1):
    '''
    computes weighted mean in a region

    :param z: data in _nparray with dimension [len(T), len(th), len(ph)]
    :param th: _nparray of colatitudes
    :param ph: _nparray of longitudes
    :param weights: _nparray with dimension [len(th)]
    :param R: radius of spherical surface
    :param axis: axis upon which to perform mean 0:time, 1: colatitude, 3: longitude. (default: 1)
    :return:
    '''
    if th is None:
        Nth = z.shape[1]
        dth = 180/Nth
        th = _np.linspace(dth/2,180-dth/2,Nth)
    else:
        dth = (th[1] - th[0]) * _np.pi / 180
    if ph is None:
        nph = z.shape[2]
        dph = 360/nph
        ph = _np.linspace(dph/2,360-dph/2,nph)
    else:
        dph = (ph[1] - ph[0]) * _np.pi / 180
    pp, tt = _np.meshgrid(ph, th)
    if len(weights.shape) == 2:
        if weights.shape == z.shape[1:]:
            wt = weights
        elif weights.shape[0] == z.shape[1] and weights.shape[1] == 1:
            wt = weights.repeat(z.shape[2], axis=1)
        else:
            raise TypeError('weight wrong shape')
    elif len(weights.shape) == 1:
        if weights.shape[0] == z.shape[1]:
            wt = weights[:,_np.newaxis].repeat(z.shape[2], axis=1)
        else:
            raise TypeError('weight wrong shape')
    else:
        raise TypeError('weight wrong shape')

    if axis>0:
        area_axis = axis - 1
        area = _np.sum(_np.abs(wt) * _np.sin(tt * _np.pi / 180) * dth * dph * R ** 2, axis=area_axis)
    else:
        area = _np.abs(wt) * _np.sin(tt * _np.pi / 180) * dth * dph * R ** 2
    return _np.sum(z*wt*_np.sin(tt*_np.pi/180) * dth * dph * R ** 2, axis=axis) / area

def normal(x, mu, sigma):
    return (_np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (2 * sigma ** 2 * _np.pi) ** 0.5)[:, None]

### PWN Analysis

def compute_frequency_wavenumber(y, T, fourier_mult=20):
    '''
    computes freq / wavenumber 2D fft for given data (y), with axes of time [0] and longitude [1].
    fourier_mult determines the sampling rate for the fourier transform
    '''
    Tend = T[-1]
    Tstart = T[0]
    nph = y.shape[1]
    Nt = y.shape[0]

    nphf = int(nph * fourier_mult)
    Tphf = 1 * fourier_mult / nphf
    Ntf = int(Nt * fourier_mult)
    Ttf = (Tend - Tstart) * fourier_mult / Ntf

    yf = _np.fft.fft2(y, s=[Ntf, nphf])
    yf = _np.fft.fftshift(yf, axes=1)

    m = _np.linspace(-1 / (2 * Tphf), 1 / (2 * Tphf), nphf)
    freq = _np.linspace(0, 1 / (2 * Ttf), Ntf // 2)
    return m, freq, yf

def compute_frequency_wavenumber_region(z, T, fourier_mult=10, m_min=-10, m_max=10,
                                        period_min=2.5, period_max=32, return_axis_labels=True):
    '''
    take 2dfft of particular m and period region
    '''
    m, freq, zf = compute_frequency_wavenumber(z, T, fourier_mult=fourier_mult)
    # find proper data to plot
    ximin = _np.where(m > m_min)[0][0]
    ximax = _np.where(m < m_max)[0][-1]
    yimax = _np.where(freq < 1 / period_min)[0][-1]
    yimin = _np.where(freq > 1 / period_max)[0][0]

    # set up axes
    m_plt = m[ximin:ximax]
    freq_plt = freq[yimin:yimax]
    z_plt = _np.abs(zf[yimin:yimax, ximin:ximax])

    if return_axis_labels:
        freq_label_loc = _np.linspace(freq_plt[0], freq_plt[-1], 10)  # 10 is simply the number of labels on chart
        period_labels = ['{:.1f}'.format(1 / x) for x in freq_label_loc]
        return m_plt, freq_plt, z_plt, freq_label_loc, period_labels
    else:
        return m_plt, freq_plt, z_plt

## Correlation Analysis
def sweep_SASVconv(phases, periods, T, SA_t, SV_t, SASV_from_phaseperiod_function, weights=1., normalize=False):
    ''' computes the cross-correlation between observed SA/SV and SA/SV produced by a wave of a series of periods and phases

    :param phases:
    :param periods:
    :param T:
    :param SA_t:
    :param SV_t:
    :param SASV_from_periodphase_function:
    :param latmax:
    :return:
    '''
    SAcorr = _np.empty((len(phases), len(periods)))
    SVcorr = _np.empty((len(phases), len(periods)))
    if normalize:
        SAautocorr0 = _np.sqrt(convolve(SA_t, SA_t, T, weights=weights))
        SVautocorr0 = _np.sqrt(convolve(SV_t, SV_t, T, weights=weights))
    for i, phase in enumerate(phases):
        for j, period in enumerate(periods):
            SAwave_t, SVwave_t = SASV_from_phaseperiod_function(phase, period)
            if normalize:
                SAautocorr = _np.sqrt(convolve(SAwave_t, SAwave_t, T, weights=weights)) * SAautocorr0
                SVautocorr = _np.sqrt(convolve(SVwave_t, SVwave_t, T, weights=weights)) * SVautocorr0
                SAcorr[i, j] = convolve(SAwave_t, SA_t, T, weights=weights)/SAautocorr
                SVcorr[i, j] = convolve(SVwave_t, SV_t, T, weights=weights)/SVautocorr
            else:
                SAcorr[i, j] = convolve(SAwave_t, SA_t, T, weights=weights)
                SVcorr[i, j] = convolve(SVwave_t, SV_t, T, weights=weights)
        print('finished phase {}/{}'.format(i + 1, len(phases)))
    return SAcorr, SVcorr

def sweep_SASVcrosscorr(phases, periods, T, SA_t, SV_t, SASV_from_phaseperiod_function, weights=_np.ones((1)), print_update=True):
    ''' computes the cross-correlation between observed SA/SV and SA/SV produced by a wave of a series of periods and phases

    :param phases:
    :param periods:
    :param T:
    :param SA_t:
    :param SV_t:
    :param SASV_from_periodphase_function:
    :param latmax:
    :return:
    '''
    SAcrosscorr = _np.empty((len(phases), len(periods)))
    SVcrosscorr = _np.empty((len(phases), len(periods)))

    swt = _convert_weights(SA_t, weights)
    SAswt = SA_t * swt
    SAswtm = _np.mean(SAswt)
    SAswtsd = standard_deviation(SAswt, fwm=SAswtm)

    SVswt = SV_t * swt
    SVswtm = _np.mean(SVswt)
    SVswtsd = standard_deviation(SVswt, fwm=SVswtm)

    for i, phase in enumerate(phases):
        for j, period in enumerate(periods):
            SAwave_t, SVwave_t = SASV_from_phaseperiod_function(phase, period)
            SAcrosscorr[i, j] = cross_correlation(SA_t, SAwave_t, swt=swt, fw=SAswt, fwm=SAswtm, fwsd=SAswtsd)
            SVcrosscorr[i, j] = cross_correlation(SV_t, SVwave_t, swt=swt, fw=SVswt, fwm=SVswtm, fwsd=SVswtsd)
        if print_update:
            print('\r\t\tfinished phase {}/{}'.format(i + 1, len(phases)), end='')
    return SAcrosscorr, SVcrosscorr

def sweep_SVcrosscorr(phases, periods, T, SV_t, SASV_from_phaseperiod_function, weights=_np.ones((1)), print_update=True):
    ''' computes the cross-correlation between observed SV and SV produced by a wave of a series of periods and phases

    :param phases:
    :param periods:
    :param T:
    :param SA_t:
    :param SV_t:
    :param SASV_from_periodphase_function:
    :param latmax:
    :return:
    '''
    SVcrosscorr = _np.empty((len(phases), len(periods)))

    swt = _convert_weights(SV_t, weights)

    SVswt = SV_t * swt
    SVswtm = _np.mean(SVswt)
    SVswtsd = standard_deviation(SVswt, fwm=SVswtm)

    for i, phase in enumerate(phases):
        for j, period in enumerate(periods):
            _, SVwave_t = SASV_from_phaseperiod_function(phase, period)
            SVcrosscorr[i, j] = cross_correlation(SV_t, SVwave_t, swt=swt, fw=SVswt, fwm=SVswtm, fwsd=SVswtsd)
        if print_update:
            print('\r\t\tfinished phase {}/{}'.format(i + 1, len(phases)), end='')
    return SVcrosscorr


### Correlation Post-Analysis Functions
def stackphase(S):
    """code computes only 0-180 deg, stack output to get full 0-360 deg"""
    return _np.hstack((S, -S))

def unroll_phase(phases):
    """plot phases continuously on -inf < th < inf instead of 0 < th < 360 with wrap-around. """
    dist = 90
    phases = _np.array(phases)
    for i in range(len(phases) - 1):
        p1 = phases[i + 1]
        p0 = phases[i]
        dp = p1 - p0
        if (_np.abs(p1 % 360 - 360) < dist) or (_np.abs(p1 % 360) < dist):
            if (_np.abs(p0 % 360 - 360) < dist) or (_np.abs(p0 % 360) < dist):
                if p0 % 360 > 180 and p1 % 360 < 180:
                    phases[i + 1:] += 360
                elif p0 % 360 < 180 and p1 % 360 > 180:
                    phases[i + 1:] -= 360
    return phases

def load_corr(dirname, params, dthfloat=True):
    Nparams = [len(x) for x in params]
    SAcorr = _np.zeros(Nparams)
    SVcorr = _np.zeros(Nparams)
    Nl, Nm, Nperiod, Nvm, Nphaseplt, Ndelth = Nparams
    Nphase = int(Nphaseplt/2)
    ls, ms, periods, vmaxs, phaseplt, delta_ths = params
    for il in range(Nl):
        for im in range(Nm):
            for idth in range(Ndelth):
                for ivm in range(Nvm):
                    if dthfloat:
                        filename = dirname + 'l{}m{}dth{:.1f}.m'.format(ls[il], ms[im], delta_ths[idth])
                    else:
                        filename = dirname + 'l{}m{}dth{:.0f}.m'.format(ls[il], ms[im], delta_ths[idth])
                    try:
                        SAc, SVc = _dill.load(open(filename, 'rb'))
                    except:
                        print('no file ' + filename + ' filling with nans')
                        SAc = _np.ones((Nperiod, Nphase)).T * _np.nan
                        SVc = _np.ones((Nperiod, Nphase)).T * _np.nan
                    SAcorr[il, im, :, ivm, :, idth] = stackphase(SAc.T)
                    SVcorr[il, im, :, ivm, :, idth] = stackphase(SVc.T)
    return SAcorr, SVcorr

def get_mi(m, minm=-12, dm=1):
    return (m - minm) // dm

def get_peri(per, minper=3, dper=1):
    return int((per - minper) // dper)

def get_phsei(phase, minphase=0, dphase=10):
    return (phase - minphase) // dphase

def get_dthi(dth, mindth=5, ddth=5):
    return (dth - mindth) // ddth



def get_peak_phase_period_slice(phases, periods, corr, return_peak_location=False):
    z = _np.array(corr.T)
    z = _np.concatenate((z,-z), axis=1)
    phase_plt = _np.linspace(0,360,len(phases)*2, endpoint=False)
    zpeak = _np.max(z)
    peri,phsi = _np.where(z==zpeak)
    phase_val = phase_plt[phsi[0]]
    per_val = periods[peri[0]]
    # print("Peak Correlation phase={0:.1f} degrees, period={1:.2f} yrs".format(phase_val, per_val))
    m_slice = z[:,phsi[0]]
    if return_peak_location:
        return m_slice, phase_val, per_val
    else:
        return m_slice

def get_peak_phase_period_eachperiod(phases, periods, corr):
    z = _np.array(corr.T)
    z = _np.concatenate((z, -z), axis=1)
    return _np.max(z,axis=1)

def get_corr_phase(S, wave_params, params):
    l,m,per,vmax,phase,dth = wave_params
    ls,ms,pers,vmaxs,phases,dths = params
    sph = S[l, get_mi(m), get_peri(per), 0, :, get_dthi(dth)]
    return sph

def find_phasemax(S, wave_params, params):
    sph = get_corr_phase(S, wave_params, params)
    sph /= _np.max(sph)
    phaseplt = params[4]
    def phfit(pmax):
        y = _np.cos((phaseplt-pmax)*_np.pi/180)
        return _np.sum((y-sph)**2)
    out = _op.minimize(phfit,0)
    pmax = out.x[0] % 360  
    return pmax

def set_phasemax_list(S, wave_params_list, params):
    out_list = []
    for wp in wave_params_list:
        nwp = list(wp)
        nwp[4] = find_phasemax(S, wp, params)
        out_list.append(tuple(nwp))
    return out_list

#### Amplitude Routines
def sweep_amplitude_misfit(SA_obs, SA_waves_list, amp_min=0.1, amp_max=5, Namps=20, weights=1.):
    ''' Computes the goodness-of-fit across an array of amplitudes for each wave,

    given the observed SA and a list of SA resulting from each wave (with vmax of 1 km/yr)

    :param SA_obs:
    :param SA_waves_list:
    :param vmin:
    :param vmax:
    :param Namps:
    :return:
    '''
    amp = _np.linspace(amp_min, amp_max, Namps)
    args = tuple([amp] * len(SA_waves_list))
    out = _it.product(*args)
    misfit = _np.empty(Namps ** len(SA_waves_list))
    for i, amps in enumerate(out):
        SA_waves = _np.zeros_like(SA_waves_list[0])
        for sa, a in zip(SA_waves_list, amps):
            SA_waves += sa * a
        misfit[i] = rms_region_allT(SA_obs - SA_waves, weights=weights)
    return _np.reshape(misfit, [Namps] * len(SA_waves_list))

def find_best_from_swept_misfit(amp_swept, amp_min=0.1, amp_max=5, Namps=20, return_inds=False):
    ''' finds the best-fit set of amplitudes given an array of fit values

    :param amp_fits:
    :param vmin:
    :param vmax:
    :param Namps:
    :param return_inds:
    :return:
    '''
    ind_max = _np.unravel_index(_np.argmin(amp_swept), amp_swept.shape)
    amp = _np.linspace(amp_min, amp_max, Namps)
    v_fits = []
    for i in ind_max:
        v_fits.append(amp[i])
    if return_inds:
        return v_fits, ind_max
    else:
        return v_fits

def fit_amplitudes(SA, SAw_normalized, amp0=None, bounds=None, weights=1., opfun=None):
    if opfun is None:
        opfun = _op.fmin_slsqp
    if amp0 is None:
        amp0 = [1.]*len(SAw_normalized)
    if bounds is None:
        bounds = [(0, 4)] * len(SAw_normalized)

    def misfit(amps):
        mis = _np.array(SA)
        for i in range(len(SAw_normalized)):
            mis -= amps[i] * SAw_normalized[i]
        return rms_region_allT(mis, weights=weights)

    res = opfun(misfit, amp0, bounds=bounds)
    return res