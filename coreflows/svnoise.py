from . import analyze as _anl
from . import advect as _advect
from . import functions as _fn
import numpy as _np
import dill as _dill

_adv = _advect.Advect()

def fit_lm_sd(lm_data, deg=1, return_real_sd=False):
    ''' fit the mean and standard deviation of a fft coefficient of a set of spherical harmonic coefficients up to l_max

    :param lm_data:
    :param deg:
    :param return_real_sd:
    :return:
    '''
    l_max = lm_data.shape[0] - 1
    l_weights = _np.linspace(1, l_max + 1, l_max + 1)
    l_values = _np.linspace(0, l_max, l_max + 1)
    mean_by_l = _np.empty(l_max + 1)
    sd_by_l = _np.empty(l_max + 1)
    for l in range(l_max + 1):
        mean_by_l[l] = _np.mean(lm_data[l, :l + 1])
        sd_by_l[l] = _np.std(lm_data[l, :l + 1])
    pf = _np.polyfit(l_values, mean_by_l, deg, w=l_weights)
    fit = _np.polyval(pf, l_values)
    pf_sd = _np.polyfit(l_values, sd_by_l, deg=1)
    sd_fit = _np.polyval(pf_sd, l_values)
    sd_fit[sd_fit<0.] =0.
    if return_real_sd:
        return fit, sd_fit, sd_by_l
    else:
        return fit, sd_fit

def fit_lm_sd_in_log(lm_data, deg=1):
    ''' fit mean, high 1 std, low 1 std, in log space

    :param lm_data:
    :param deg:
    :return:
    '''
    mean, sd = fit_lm_sd(_np.log10(lm_data), deg=deg)
    return 10 ** mean, 10 ** (mean - sd), 10 ** (mean + sd)

def fit_lm_sd_in_linear(lm_data, deg=1):
    ''' fit mean, high 1 std, low 1 std, in linear space

    :param lm_data:
    :param deg:
    :return:
    '''
    mean, sd = fit_lm_sd(lm_data, deg=deg)
    return mean, mean - sd, mean + sd

def fit_all_lm_sd(lm_fft, Nfft=None, deg_fits=(1,2,2,2,2), log=False):
    ''' fits mean, sdl, sdh for each Fourier coefficient

    :param Nfft:
    :param deg_fits:
    :param log:
    :return:
    '''
    if Nfft is None:
        Nfft = min(lm_fft.shape[-1], len(deg_fits))
    fits = _np.empty((Nfft, lm_fft.shape[0]))
    sdls = _np.empty((Nfft, lm_fft.shape[0]))
    sdhs = _np.empty((Nfft, lm_fft.shape[0]))
    if log:
        for i in range(Nfft):
            fits[i, :], sdls[i, :], sdhs[i, :] = fit_lm_sd_in_log(_np.abs(lm_fft[:, :, i]), deg=deg_fits[i])
    else:
        for i in range(Nfft):
            fits[i, :], sdls[i, :], sdhs[i, :] = fit_lm_sd_in_linear(_np.abs(lm_fft[:, :, i]), deg=deg_fits[i])
    return fits, sdls, sdhs

def generate_rand_lm_mags(mean, sd, l_max=None, m0_modifier=0.9):
    ''' generates random values for the magnitudes of a particular fft coefficient of a set of spherical harmonics

    Normal Distribution

    :param mean:
    :param sd:
    :param l_max:
    :param m0_modifier:
    :return:
    '''
    if l_max is None:
        l_max = mean.shape[0] - 1
    rand_vals = _np.zeros((l_max + 1, l_max + 1))
    l = 0
    rand_vals[l:l_max + 1, l] = _np.abs(_np.random.normal(loc=mean[l:l_max + 1], scale=sd[l:l_max + 1])) * m0_modifier
    for l in range(1, l_max + 1):
        rand_vals[l:l_max + 1, l] = _np.abs(_np.random.normal(loc=mean[l:l_max + 1], scale=sd[l:l_max + 1]))
    return rand_vals

def generate_rand_lm_phases(l_max):
    ''' generates random values for the phases of a particular fft coefficient of a set of spherical harmonics.

    Uniform Distribution

    :param l_max:
    :return:
    '''
    rand_phases = _np.zeros((l_max + 1, l_max + 1))
    for l in range(l_max + 1):
        rand_phases[l:l_max + 1, l] = _np.random.uniform(low=-_np.pi, high=_np.pi, size=l_max + 1 - l)
    return rand_phases

def generate_all_rand_lm_magphase(lm_fft, degfit_by_fft=(1,2,2,2,2), log=False):
    ''' generates random magnitudes and phases for each Fourier coefficient for each spherical harmonic lm

    :param lm_fft:
    :param degfit_by_fft:
    :param log:
    :return:
    '''
    l_max = lm_fft.shape[0] - 1
    Nfft = lm_fft.shape[-1]
    if log:
        lm_mag = _np.log10(_np.abs(lm_fft))
        m0_modifier = 0.9
    else:
        lm_mag = _np.abs(lm_fft)
        m0_modifier = 0.5
    rand_mags = _np.zeros((l_max + 1, l_max + 1, Nfft))
    rand_phases = _np.zeros((l_max + 1, l_max + 1, Nfft))
    for n in range(Nfft):
        lm_data = lm_mag[:, :, n]
        mean, sd = fit_lm_sd(lm_data, deg=degfit_by_fft[n])
        if not log:
            mean[mean<0.] = 0.
        rand_mags[:, :, n] = generate_rand_lm_mags(mean, sd, m0_modifier=m0_modifier)
        rand_phases[:, :, n] = generate_rand_lm_phases(l_max)
    if log:
        rand_mags = 10 ** rand_mags
    return rand_mags, rand_phases

def generate_rand_SV(T, lm_fft, degfit_by_fft=(1,2,2,2,2), log=False, norm='4pi', Nth=None, normalize_to_rms=None, norm_weights=1., return_norm_ratio=False):
    ''' generates a new realization of the SV resdiual spherical harmonics across time

    :param T:
    :param lm_fft:
    :param degfit_by_fft:
    :param log:
    :param norm:
    :return:
    '''
    if Nth is None:
        Nth = lm_fft.shape[0]*2
    rand_mags, rand_phases = generate_all_rand_lm_magphase(lm_fft, degfit_by_fft=degfit_by_fft, log=log)
    rand_fft = rand_mags * _np.exp(1j * rand_phases)
    SV_rand_sh = get_lm_ifft(T, rand_fft, norm=norm)
    if normalize_to_rms is not None:
        SV_rand, SV_rand_sh, norm_ratio = normalize_SV(SVsh_to_normalize=SV_rand_sh, Nth=Nth, rms_norm=normalize_to_rms, weights=norm_weights, return_norm_ratio=True)
    else:
        SV_rand = _adv.vSH2v_allT(SV_rand_sh, Nth=Nth)
    if return_norm_ratio and normalize_to_rms is not None:
        return SV_rand, SV_rand_sh, norm_ratio
    else:
        return SV_rand, SV_rand_sh

def generate_rand_SA(T, lm_fft, degfit_by_fft=(1,2,2,2,2), log=False, norm='4pi', Nth=None, normalize_to_rms=None, norm_weights=1., return_norm_ratio=False):
    ''' generates a new realization of the SV resdiual spherical harmonics across time

    :param T:
    :param lm_fft:
    :param degfit_by_fft:
    :param log:
    :param norm:
    :return:
    '''
    if Nth is None:
        Nth = lm_fft.shape[0]*2
    rand_mags, rand_phases = generate_all_rand_lm_magphase(lm_fft, degfit_by_fft=degfit_by_fft, log=log)
    rand_fft = 1j*_np.fft.fftfreq(len(rand_mags), d=T[-1]-T[0])*rand_mags * _np.exp(1j * rand_phases)
    SV_rand_sh = get_lm_ifft(T, rand_fft, norm=norm)
    if normalize_to_rms is not None:
        SV_rand, SV_rand_sh, norm_ratio = normalize_SV(SVsh_to_normalize=SV_rand_sh, Nth=Nth, rms_norm=normalize_to_rms, weights=norm_weights, return_norm_ratio=True)
    else:
        SV_rand = _adv.vSH2v_allT(SV_rand_sh, Nth=Nth)
    if return_norm_ratio and normalize_to_rms is not None:
        return SV_rand, SV_rand_sh, norm_ratio
    else:
        return SV_rand, SV_rand_sh

def get_lm_ifft(T, lm_fft, norm='4pi'):
    ''' computes the inverse Fourier transform across time for a set of spherical harmonics

    :param T:
    :param lm_fft:
    :param norm:
    :return:
    '''
    l_max = lm_fft.shape[0] - 1
    lm_sh = _np.zeros((len(T), 2, l_max + 1, l_max + 1))
    for l in range(l_max + 1):
        for m in range(l + 1):
            f = _np.zeros((len(T)), dtype='complex')
            Nfreq = (lm_fft.shape[2] - 1) // 2
            f[:Nfreq + 1] = lm_fft[l, m, :Nfreq + 1]
            f[-Nfreq:] = lm_fft[l, m, -Nfreq:]
            ifft = _np.fft.ifft(f, n=len(T))
            lm_sh[:, 0, l, m] = ifft.real
            lm_sh[:, 1, l, m] = ifft.imag
    return lm_sh

def get_lm_fft(T, shcoeffs_t, Nfft=5, l_max=14, norm='4pi', return_l_values=False):
    ''' computes the Fourier transform across time for a set of spherical harmonics

    :param T:
    :param shcoeffs_t:
    :param Nfft:
    :param l_max:
    :param norm:
    :param return_l_values:
    :return:
    '''
    l_arr = min(shcoeffs_t.shape[2], l_max + 1)
    lm_fft = _np.zeros((l_max + 1, l_max + 1, Nfft), dtype='complex')
    Nfreq = (Nfft - 1) // 2
    for l in range(l_arr):
        for m in range(l + 1):
            fft = _np.fft.fft(shcoeffs_t[:, 0, l, m] + shcoeffs_t[:, 1, l, m] * 1j)
            ni = (len(fft) - 1) // 2 + 1
            lm_fft[l, m, 0] = fft[0]
            lm_fft[l, m, 1:Nfreq + 1] = fft[1:Nfreq + 1]
            lm_fft[l, m, -Nfreq:] = fft[-Nfreq:]
    return lm_fft

def unroll_phase(phase):
    ''' takes a list of phases across time in the range (-pi, pi) and unrolls it into a continuous function of unlimited range

    :param phase:
    :return:
    '''
    p = phase
    for i in range(len(p) - 1):
        dp = p[i + 1] - p[i]
        if dp > _np.pi:
            phase[i + 1:] += -2 * _np.pi
        elif dp < -_np.pi:
            p[i + 1:] += 2 * _np.pi
    return p

def crop_pwn(m, freq, pwn, m_max, T_min, T_max, return_indexes=False):
    ''' crops a period-wavenumber transformation into only the desired range for smaller storage.

    :param pwn:
    :param m_max:
    :param T_min:
    :param T_max:
    :return:
    '''
    im_max = _np.where(m > m_max)[0][0]
    im_min = _np.where(m > -m_max)[0][0]
    it_min = _np.where(freq > 1 / T_max)[0][0]
    it_max = _np.where(freq > 1 / T_min)[0][0]
    pwn_ind = ((it_min, it_max+(it_max-it_min)), (im_min, im_max))
    m_out = m[im_min:im_max]
    freq_out = freq[it_min:it_max]
    pwn_out = pwn[pwn_ind[0][0]:pwn_ind[0][1], pwn_ind[1][0]:pwn_ind[1][1]]
    if return_indexes:
        return m_out, freq_out, pwn_out, (im_min, im_max), (it_min, it_max), pwn_ind
    else:
        return m_out, freq_out

def get_magphase(data_in):
    '''

    :param data_in:
    :return:
    '''
    mag = _np.abs(data_in)
    if isinstance(data_in, (list, _np.ndarray)):
        phase = unroll_phase(_np.arctan2(data_in.imag,data_in.real))
    else:
        phase = _np.arctan2(data_in.imag,data_in.real)
    return mag, phase

def get_lm_magphase(lm_fft):
    '''

    :param lm_fft:
    :return:
    '''
    mag = _np.zeros_like(lm_fft, dtype=float)
    phase = _np.zeros_like(lm_fft, dtype=float)
    for l in range(lm_fft.shape[0]):
        for m in range(l+1):
            for i in range(lm_fft.shape[-1]):
                mag[l,m,i], phase[l,m,i] = get_magphase(lm_fft[l,m,i])
    return mag, phase

def normalize_SV(SVsh_to_normalize=None, SV_to_normalize=None, rms_norm=None, SV_real=None, weights=1., Nth=None, l_max=None, return_norm_ratio=False):
    ''' normalizes the rms power of one dataset to match another

    :param SV_real:
    :param SV_to_normalize:
    :param weights:
    :param SVsh_to_normalize:
    :return:
    '''
    if rms_norm is None:
        if SV_real is None:
            raise ValueError('Must specify either rms_norm or a SV dataset to normalize to')
        else:
            rms_norm = _anl.rms_region_allT(SV_real, weights=weights)
    if SVsh_to_normalize is None:
        if SV_to_normalize is None:
            raise ValueError('must specify either SV_to_normalize or SVsh_to_normalize or both')
        else:
            SVsh_to_normalize = _adv.v2vSH_allT(SV_to_normalize, l_max=l_max)
    if SV_to_normalize is None:
        if SVsh_to_normalize is None:
            raise ValueError('must specify either SV_to_normalize or SVsh_to_normalize or both')
        else:
            SV_to_normalize = _adv.vSH2v_allT(SVsh_to_normalize, Nth=Nth, l_max=l_max)
    SV_2norm_rms = _anl.rms_region_allT(SV_to_normalize, weights=weights)
    norm_ratio = rms_norm/SV_2norm_rms
    if return_norm_ratio:
        return SV_to_normalize*norm_ratio, SVsh_to_normalize*norm_ratio, norm_ratio
    else:
        return SV_to_normalize*norm_ratio, SVsh_to_normalize*norm_ratio

def compute_many_SVsr_pwn(T, SVr, N, pwn_weights=None, Nth=None, Nfft=5, degfit=(1,2,2,2,2), logfit=False, l_max=14, m_max=14, T_min=2.5, T_max=24, norm_weights=1.):
    th, ph = _adv.get_thvec_phvec_DH(Nth=SVr.shape[1], l_max=l_max)
    if pwn_weights is None:
        lat = th-90
        sigmath = 16
        pwn_weights = _fn.hermite(lat/sigmath, 0)
    if Nth is None:
        Nth = SVr.shape[1]
    SVr_rms = _anl.rms_region_allT(SVr, weights=norm_weights)
    SVrsh = _adv.v2vSH_allT(SVr)
    SVr_fft = get_lm_fft(T, SVrsh, Nfft=Nfft, l_max=l_max)
    SVsr, _ = generate_rand_SV(T, SVr_fft, degfit_by_fft=degfit, log=logfit, Nth=Nth, normalize_to_rms=SVr_rms, norm_weights=norm_weights, return_norm_ratio=False)
    SVsr_eq = _anl.weighted_mean_region_allT(SVsr, th=th, weights=pwn_weights)
    m, freq, SVsr_pwn  = _anl.compute_frequency_wavenumber(SVsr_eq, T)
    m_save, freq_save, pwn, m_ind, freq_ind, pwn_ind = crop_pwn(m,freq, SVsr_pwn, m_max, T_min, T_max, return_indexes=True)
    Nm = len(m_save)
    Nt = len(freq_save)
    pwn_all = _np.empty((N, Nt*2, Nm), dtype=_np.float)
    N10 = max(N//10,1)
    for i in range(N):
        if i%N10 == 0:
            print('on step {}/{}'.format(i,N))
        SVsr, _ = generate_rand_SV(T, SVr_fft, degfit_by_fft=degfit, log=logfit, Nth=Nth, normalize_to_rms=SVr_rms, norm_weights=norm_weights, return_norm_ratio=False)
        SVsr_eq = _anl.weighted_mean_region_allT(SVsr, th=th, weights=pwn_weights)
        _, _, SVsr_pwn  = _anl.compute_frequency_wavenumber(SVsr_eq, T)
        pwn_all[i,:,:] = _np.abs(SVsr_pwn[pwn_ind[0][0]:pwn_ind[0][1], pwn_ind[1][0]:pwn_ind[1][1],])
    return m_save, freq_save, pwn_all

def save_computed_noise(m,freq,pwn_all, filename='computed_noise.m'):
    _dill.dump((m,freq,pwn_all),open(filename,'wb'))

def load_computed_noise(filename='computed_noise.m'):
    m,freq,pwn_all = _dill.load(open(filename,'rb'))
    return m,freq,pwn_all

def compute_pwn_percentile(pwn_all, p):
    return _np.percentile(pwn_all, p, axis=0)

# def fit_l_over_t_using_fft(self, T, data, Nfft=10, scale_mult=1):
#     dfft = _np.fft.fft(data)
#     fft_mag = _np.random.normal(loc=_np.zeros(Nfft), scale=_np.abs(dfft[:Nfft])*scale_mult)
#     fft_phase = _np.random.uniform(size=Nfft)
#     ffts = fft_mag * _np.exp(1j * 2 * _np.pi * fft_phase)
#     ys = _np.fft.ifft(ffts, n=len(T))
#     return ys.real
#
# def simulate_coeffs_fft(self, T, shcoeffs_t, Nfft=10, scale_mult=1, scale_l=0, l_max=14):
#     l_arr = min(shcoeffs_t.shape[2], l_max)
#     shcoeffs_sim = _np.zeros((shcoeffs_t.shape[0], shcoeffs_t.shape[1], l_arr, l_arr))
#     for k in range(shcoeffs_t.shape[1]):
#         for l in range(l_arr):
#             for m in range(l_arr):
#                 shcoeffs_sim[:, k, l, m] = self.fit_l_over_t_using_fft(T, shcoeffs_t[:, k, l, m], Nfft=Nfft, scale_mult=scale_mult*(l**scale_l))
#     return shcoeffs_sim
