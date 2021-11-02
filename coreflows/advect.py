import pyshtools as _sht
import numpy as _np
import coremagmodels as _cm
import scipy.optimize as _op
from . import functions as _fn

class Advect(_cm.models.SphereHarmBase):
    def __init__(self):
        super(Advect, self).__init__()

    def v2vSH(self, v, l_max=None):
        """convert v at DH grid points to v in SH

        :param v:
        :return:
        """
        lm = (v.shape[0]-2)//2
        if l_max is not None:
            lm = min(l_max, lm)
        return _sht.shtools.SHExpandDH(v, norm=1, sampling=2, csphase=1)[:,:lm+1,:lm+1]

    def v2vSH_allT(self, v_t, l_max=None):
        """convert v at DH grid points to v in SH at all times T in v_t

        :param v_t:
        :return:
        """
        lm = (v_t.shape[1]-2)//2
        if l_max is not None:
            lm = min(l_max, lm)
        vSH_t = _np.empty((v_t.shape[0],2,lm+1,lm+1))
        for i in range(v_t.shape[0]):
            vSH_t[i,:,:,:] = self.v2vSH(v_t[i,:,:], l_max=lm)
        return vSH_t

    def vSH2v_allT(self, vSH_t, l_max=None, Nth=None):
        """convert v in SH to v at DH grid points at all times T in vSH_t

        :param vSH_t:
        :param l_max:
        :param Nth:
        :return:
        """
        if Nth is None and l_max is None:
            l_max = self._get_lmax(vSH_t[0])
            Nth = l_max * 2 + 2
            lm = l_max
        elif Nth is None:
            lm = l_max
            Nth = lm*2+2
        else:
            lm = (Nth-2)/2
        v_t = _np.empty((len(vSH_t), Nth, Nth*2))
        for i,vSH in enumerate(vSH_t):
            SH = self._convert_SHin(vSH, l_max=l_max)
            v_t[i,:,:] = _sht.shtools.MakeGridDH(SH, norm=1, sampling=2, csphase=1, lmax_calc=l_max, lmax=lm)
        return v_t

    def vSH2v(self, vSH, l_max=None, Nth=None):
        """convert v in spherical harmonics to points on DH grid

        :param vSH:
        :param l_max:
        :param Nth:
        :return:
        """
        if Nth is None and l_max is None:
            l_max = self._get_lmax(vSH)
            Nth = l_max * 2 + 2
            lm = l_max
        elif Nth is None:
            lm = l_max
        else:
            lm = (Nth-2)/2
        SH = self._convert_SHin(vSH, l_max=l_max)
        return _sht.shtools.MakeGridDH(SH, norm=1, sampling=2, csphase=1, lmax_calc=l_max, lmax=lm)

    def gradient_vSH(self, vSH, l_max=None, Nth=None):
        """
        compute th,ph gradients of a v in SH at DH grid points

        :param self:
        :param vSH: velocity field spherical harmonics [km/yr]
        :param l_max:
        :param Nth:
        :return: gradients of the field in [ km / yr-km ]
        """
        if Nth is None and l_max is None:
            l_max = self._get_lmax(vSH)
            Nth = l_max*2+2
            lm = l_max
        if Nth is None:
            lm = l_max
        else:
            lm = (Nth - 2) / 2
        SH = self._convert_SHin(vSH, l_max=l_max)
        out = _sht.shtools.MakeGravGridDH(SH, 3480, 3480, sampling=2, normal_gravity=0, lmax_calc=l_max, lmax=lm)
        dth_v = out[1]
        dph_v = out[2]
        return dth_v, dph_v

    def gradients_v_allT(self, v_t, Nth=None, l_max=None):
        """compute th,ph gradients of v across all times in v_t

        :param v_t:
        :param Nth:
        :param l_max:
        :return:
        """
        dth_t = _np.empty_like(v_t)
        dph_t = _np.empty_like(v_t)
        for i, t in enumerate(range(v_t.shape[0])):
            dSH = self.v2vSH(v_t[i, :, :])
            dth_t[i, :, :], dph_t[i, :, :] = self.gradient_vSH(dSH, l_max=l_max, Nth=Nth)
        return dth_t, dph_t

    def dth_v(self, vSH, l_max=14, Nth=None):
        """compute latitudinal (th) gradient of v from SH

        :param self:
        :param vSH: spherical harmonics of a velocity field with units [km/yr]
        :param l_max:
        :param Nth:
        :return: gradient of velocity field in units of [ km / (yr-km) ]
        """

        if Nth is None:
            lm = l_max
        else:
            lm = (Nth-2)/2
        SH = self._convert_SHin(vSH, l_max=l_max)
        out = _sht.shtools.MakeGravGridDH(SH, 3480, 3480, sampling=2, normal_gravity=0, lmax_calc=l_max, lmax=lm)
        dth_v = out[1]
        return dth_v

    def dph_v(self, vSH, l_max=14, Nth=None):
        """compute the longitudinal (ph) gradient of v in SH

        :param self:
        :param vSH: spherical harmonics of a velocity field with units [km/yr]
        :param l_max:
        :param Nth:
        :return: gradient of velocity field in units of [ km / (yr-km) ]
        """
        if Nth is None:
            lm = l_max
        else:
            lm = (Nth-2)/2
        SH = self._convert_SHin(vSH, l_max=l_max)
        out = _sht.shtools.MakeGravGridDH(SH, 3480, 3480, sampling=2, normal_gravity=0, lmax_calc=l_max, lmax=lm)
        dph_v = out[2]
        return dph_v

    def advSV(self, vthSH, vphSH, BSH, magmodel=None, l_max=14, Nth=None, B_lmax = None, v_lmax=None, B_in_vSHform=False):
        """compute advection of SV

        :param vthSH: latitudinal velocity in SH [km/yr]
        :param vphSH: longitudinal velocity in SH [km/yr]
        :param BSH: Br at CMB in SH [nT]
        :param magmodel: coremagmodel class used
        :param l_max: maximum degree of SH to use in computation [-]
        :param Nth: number of latitudinal grid points to output [-]
        :param B_lmax: maximum degree of SH to use for Br [-]
        :param v_lmax: maximum degree of SH to use for velocities [-]
        :param B_in_vSHform: optional parameter to pass in Br computed not from full field,
                             but from points transformed from DH grid into SH using v2vSH [nT]
        :return: SV_adv = _np.array((Nth,2*Nth)) [nT/yr]
        """
        if v_lmax is None:
            v_lmax = l_max
        if B_lmax is None:
            B_lmax = l_max
        vth = self.vSH2v(vthSH, l_max=v_lmax, Nth=Nth)
        vph = self.vSH2v(vphSH, l_max=v_lmax, Nth=Nth)
        if B_in_vSHform:
            dthB, dphB = self.gradient_vSH(BSH, l_max=B_lmax, Nth=Nth)
        else:
            drB, dthB, dphB = magmodel.gradB_sht(BSH, l_max=B_lmax, Nth=Nth)
        return - vth*dthB - vph*dphB

    def divSV(self, vthSH, vphSH, BSH, magmodel=None, l_max=14, Nth=None, B_lmax = None, v_lmax=None, B_in_vSHform=False):
        """compute divergence of SV

        :param vthSH: latitudinal velocity in SH [km/yr]
        :param vphSH: longitudinal velocity in SH [km/yr]
        :param BSH: Br at CMB in SH [nT]
        :param magmodel: coremagmodel class used
        :param l_max: maximum degree of SH to use in computation [-]
        :param Nth: number of latitudinal grid points to output [-]
        :param B_lmax: maximum degree of SH to use for Br [-]
        :param v_lmax: maximum degree of SH to use for velocities [-]
        :param B_in_vSHform: optional parameter to pass in Br computed not from full field,
                             but from points transformed from DH grid into SH using v2vSH [nT]
        :return: SV_div = _np.array((Nth,2*Nth)) [nT/yr]
        """
        if v_lmax is None:
            v_lmax = l_max
        if B_lmax is None:
            B_lmax = l_max
        dthvth = self.dth_v(vthSH, l_max=v_lmax, Nth=Nth)
        dphvph = self.dph_v(vphSH, l_max=v_lmax, Nth=Nth)
        if B_in_vSHform:
            B = self.vSH2v(BSH, l_max=B_lmax, Nth=Nth)
        else:
            B = magmodel.B_sht(BSH, l_max=B_lmax, Nth=Nth)
        return -B*(dthvth + dphvph)

    def SV_from_flow(self, vthSH, vphSH, BSH, magmodel=None, l_max=14, Nth=None, B_lmax = None, v_lmax=None, B_in_vSHform=False):
        """compute SV from flow advection of background field

        :param vthSH: latitudinal flow [km/yr]
        :param vphSH: longitudinal flow [km/yr]
        :param BSH: field [nT]
        :param magmodel: coremagmodel class used
        :param l_max:
        :param Nth:
        :param B_lmax:
        :param v_lmax:
        :return: secular variation [nT/yr]
        """
        return self.advSV(vthSH,vphSH,BSH, magmodel=magmodel, l_max=l_max, Nth=Nth, B_lmax = B_lmax, v_lmax=v_lmax, B_in_vSHform=B_in_vSHform) \
                + self.divSV(vthSH,vphSH,BSH, magmodel=magmodel, l_max=l_max, Nth=Nth, B_lmax = B_lmax, v_lmax=v_lmax, B_in_vSHform=B_in_vSHform)

    def SA_from_flow_accel(self, athSH, aphSH, BSH, magmodel=None, l_max=14, Nth=None, B_lmax = None, v_lmax=None, B_in_vSHform=False):
        """compute secular acceleration from acceleration of advection of background field

        :param athSH: latitudinal fl accel [km/yr]
        :param aphSH: longitudinal fl accel [km/yr]
        :param BSH: field [nT]
        :param magmodel: coremagmodel class used
        :param l_max:
        :param Nth:
        :param B_lmax:
        :param v_lmax:
        :param B_in_vSHform: B in SH form from using v2vSH function

        :return: secular variation [nT/yr]
        """
        return self.SV_from_flow(athSH, aphSH, BSH, magmodel=magmodel, l_max=l_max, Nth=Nth, B_lmax=B_lmax, v_lmax=v_lmax, B_in_vSHform=B_in_vSHform)

    def SA_from_flow_SV(self, vthSH, vphSH, svSH, magmodel=None, l_max=14, Nth=None, B_lmax = None, v_lmax=None, B_in_vSHform=False):
        """compute secular acceleration from flow advection of secular variation

        :param vthSH: latitudinal flow [km/yr]
        :param vphSH: longitudinal flow [km/yr]
        :param svSH: SV [nT/yr]
        :param magmodel: coremagmodel class used
        :param l_max:
        :param Nth:
        :param B_lmax:
        :param v_lmax:
        :param B_in_vSHform: B in SH form from using v2vSH function

        :return: secular variation [nT/yr]
        """
        return self.SV_from_flow(vthSH, vphSH, svSH, magmodel=magmodel, l_max=l_max, Nth=Nth, B_lmax=B_lmax, v_lmax=v_lmax, B_in_vSHform=B_in_vSHform)

    def tp2v(self, torSH, polSH, l_max=14):
        """ convert toroidal and poloidal spherical harmonics to vectors on grid

        :param torSH:
        :param polSH:
        :param l_max:
        :return:
        """
        vtht, vpht = self.t2v(torSH, l_max=l_max)
        vthp, vphp = self.p2v(polSH, l_max=l_max)
        vth = vtht + vthp
        vph = vpht + vphp
        return vth, vph

    def t2v(self, torSH, l_max=14):
        """ convert toroidal spherical harmonics to physical vectors on grid"""
        z = self._convert_SHin(torSH, l_max=l_max)
        _, dth, dph, _ = _sht.shtools.MakeGravGridDH(z, 3480, 3480, lmax=l_max, sampling=2)
        vtht = -dph
        vpht = dth
        return vtht, vpht

    def p2v(self, polSH, l_max=14):
        """ convert poloidal spherical harmonics to physical vectors on grid"""
        z = self._convert_SHin(polSH, l_max=l_max)
        _,_,_,_,dtr,dpr = _sht.shtools.MakeGravGradGridDH(z, 3480,3480, sampling=2)
        vthp = dtr
        vphp = dpr
        return vthp, vphp

class SteadyFlow(Advect):
    def __init__(self):
        super(SteadyFlow, self).__init__()

    def import_fortran_flow_DH(self, filename):
        """ import flow from fortran fitting code for steady flow over a time period

        :param filename:
        :return:
        """
        th = []
        ph = []
        ang = []
        mag = []
        with open(filename, 'rb') as velfile:
            for line in velfile:
                ln_cln = line.strip().decode('UTF-8').split()
                ph.append(float(ln_cln[0]))
                th.append(float(ln_cln[1]))
                ang.append(float(ln_cln[2]))
                mag.append(float(ln_cln[3]))
        th = _np.array(th)
        ph = _np.array(ph)
        ph = (ph + 180.) % 360. - 180.
        ang = _np.array(ang)
        mag = _np.array(mag)
        vph = mag * _np.sin(ang * _np.pi / 180)
        vth = mag * _np.cos(ang * _np.pi / 180)
        keys = _np.lexsort((ph, th))
        n = int((len(th) / 2) ** 0.5)
        vth_trans = _np.reshape(vth[keys], (n, 2 * n))[::-1, ::]
        vph_trans = _np.reshape(vph[keys], (n, 2 * n))[::-1, ::]
        th_trans = _np.reshape(th[keys], (n, 2 * n))[::-1, ::]
        ph_trans = _np.reshape(ph[keys], (n, 2 * n))[::-1, ::]
        return th_trans, ph_trans, -vth_trans, vph_trans

    def import_fortran_flow_tpSH(self, filename):
        """BROKEN -- Import Fortran flow in toroidal and poloidal spherical harmonics

        Returns
        -------

        """
        # raw = []
        # with open(filename, 'rb') as velfile:
        #     velfile.readline()
        #     velfile.readline()
        #     for line in velfile:
        #         ln_cln = line.strip().decode('UTF-8').split()
        #         for num in ln_cln:
        #             raw.append(float(num))
        # n = len(raw) // 2
        # l_max = int((n + 1) ** 0.5) - 1
        # toroidal = raw[:n]
        # poloidal = raw[n:]
        # tcoeffs = self._convert_g_raw_to_shtarray(toroidal, l_max=l_max)
        # tSH = _sht.SHCoeffs.from_array(tcoeffs, normalization='schmidt', csphase=-1)
        # pcoeffs = self._convert_g_raw_to_shtarray(poloidal, l_max=l_max)
        # pSH = _sht.SHCoeffs.from_array(pcoeffs, normalization='schmidt', csphase=-1)
        # return tSH, pSH
        raise NotImplementedError("this function doesn't work yet")

    def SV_steadyflow_allT(self, vthSH, vphSH, BSH_t, magmodel=None, B_lmax=14, v_lmax=14, Nth=None):
        """compute SV from steady flow at all time points within BSH_t

        :param vthSH:
        :param vphSH:
        :param BSH_t:
        :param magmodel: coremagmodel class used
        :param Nth:
        :param B_lmax:
        :param v_lmax:
        :return:
        """
        SVsteadyflow_t = _np.empty((len(BSH_t), Nth, Nth * 2))
        for i, bSH in enumerate(BSH_t):
            SVsteadyflow_t[i, :, :] = self.SV_from_flow(vthSH, vphSH, bSH, magmodel=magmodel, B_lmax=B_lmax, v_lmax=v_lmax, Nth=Nth)
        return SVsteadyflow_t

    def SA_steadyflow_allT(self, vthSH, vphSH, SVsh_t, magmodel=None, B_lmax=14, v_lmax=14, Nth=None):
        """ compute SA from steady flow for all T in SVsh_t

        :param vthSH:
        :param vphSH:
        :param SVsh_t:
        :param magmodel: coremagmodel class used
        :param Nth:
        :param B_lmax:
        :param v_lmax:
        :return:
        """
        SAsteadyflow_t = _np.empty((len(SVsh_t), Nth, Nth * 2))
        for i, SVsh in enumerate(SVsh_t):
            SAsteadyflow_t[i, :, :] = self.SA_from_flow_SV(vthSH, vphSH, SVsh, magmodel=magmodel, B_lmax=B_lmax, v_lmax=v_lmax, Nth=Nth)
        return SAsteadyflow_t

class Waves(Advect):
    def __init__(self):
        super(Waves, self).__init__()

    def fit_with_hermite(self, lat, data, deg, return_coeffs=False):
        """fit 1D function to a set of hermite basis functions

        :param lat:
        :param data:
        :param deg:
        :param return_coeffs:
        :return:
        """
        fitfun_data = lambda c: _np.sum((_fn.hermite_fit(lat, c) - data) ** 2)
        c0 = _np.ones((deg + 1))
        c0[-1] = 10.
        res = _op.fmin_bfgs(fitfun_data, c0)
        outfun = lambda x: _fn.hermite_fit(x, res)
        if return_coeffs:
            return outfun, res
        else:
            return outfun

    def fit_horiz_flows_hermite(self, vec, Nk, Nl, deg, FVF, return_coefficients=False):
        """fit complex vth,vph to hermite basis functions across co-latitude

        :param vec:
        :param Nk:
        :param Nl:
        :param deg:
        :param FVF:
        :param return_coefficients:
        :return:
        """
        dth = 180 / Nl
        lat = _np.linspace(-90 + dth / 2, 90 - dth / 2, Nl)
        vec = FVF.anlyze.shift_vec_real_nomodel(vec, Nk, Nl, var='vph')
        vph = FVF.utilities.get_variable_from_vec(vec, 'vph', Nk, Nl)
        vec /= _np.max(_np.abs(vph.real))
        vph = FVF.utilities.get_variable_from_vec(vec, 'vph', Nk, Nl)
        vphr = vph[-1, :].real
        vphi = vph[-1, :].imag
        vth = FVF.utilities.get_variable_from_vec(vec, 'vth', Nk, Nl)
        vthr = vth[-1, :].real
        vthi = vth[-1, :].imag
        ffphr, cphr = self.fit_with_hermite(lat, vphr, deg, return_coeffs=True)
        ffphi, cphi = self.fit_with_hermite(lat, vphi, deg, return_coeffs=True)
        ffthr, cthr = self.fit_with_hermite(lat, vthr, deg, return_coeffs=True)
        ffthi, cthi = self.fit_with_hermite(lat, vthi, deg, return_coeffs=True)
        if return_coefficients:
            return (ffthr, ffthi, ffphr, ffphi), (cthr, cthi, cphr, cphi)
        else:
            return ffthr, ffthi, ffphr, ffphi

    def horiz_flows_given_coeffs(self, lat, coeffs, delta_th_override=None):
        """compute horizontal velocity given coefficients

        :param lat:
        :param coeffs:
        :param delta_th_override:
        :return:
        """
        if delta_th_override is None:
            vthr = _fn.hermite_fit(lat, coeffs[0])
            vthi = _fn.hermite_fit(lat, coeffs[1])
            vphr = _fn.hermite_fit(lat, coeffs[2])
            vphi = _fn.hermite_fit(lat, coeffs[3])
        else:
            vthr =  _fn.hermite_sum(lat, coeffs[0][:-1], delta_th_override)
            vthi =  _fn.hermite_sum(lat, coeffs[1][:-1], delta_th_override)
            vphr =  _fn.hermite_sum(lat, coeffs[2][:-1], delta_th_override)
            vphi =  _fn.hermite_sum(lat, coeffs[3][:-1], delta_th_override)
        return vthr + vthi * 1j, vphr + vphi * 1j

    def u_v_divv(self, lat, ph, wave_params, t, c012):
        """ compute the longitudinal (u) and latitudinal (v) flows and divergence at a grid of points specifed by lat and ph

        :param lat: latitude vector (deg)
        :param ph: longitude vector (deg)
        :param wave_params: tuple or list of wave parameters (l,m,period,vmax, phase, delta_th)
        :param t: time in [years]
        :param c012: hermite fit constants

        :return:
        """
        l, m, period, vmax, phase, delta_th = wave_params
        lon_grid, lat_grid = _np.meshgrid(ph, lat)
        vth, vph = self.horiz_flows_given_coeffs(lat, c012[l], delta_th_override=delta_th)
        v_magnitude = vmax / _np.max(_np.abs((vth ** 2 + vph ** 2) ** 0.5))
        w = 2 * _np.pi / period
        u = _np.real(vph[:, None] * _np.exp(1j * (m * lon_grid * _np.pi / 180 + w * t + phase))).T * v_magnitude
        v = _np.real(vth[:, None] * _np.exp(1j * (m * lon_grid * _np.pi / 180 + w * t + phase))).T * v_magnitude
        absv = (u ** 2 + v ** 2) ** .05
        dudph = _np.gradient(u, axis=0)
        dvdth = _np.gradient(v, axis=1)
        divv = dudph + dvdth
        return u, v, divv

    def SV_from_hermite_flows(self, l, B_lmax, c012, t=2010, period=7.5, delta_th=17, peak_flow=2, Nth=200, v_lmax=14, m=6, magmodel=None):
        """compute the SV produce from waves parameterized with hermite basis functions"""
        dth = 180 / Nth
        lat = _np.linspace(-90 + dth / 2, 90 - dth / 2, Nth)
        lon = _np.linspace(-180 + dth / 2, 180 - dth / 2, Nth * 2)
        sh = magmodel.get_shtcoeffs_at_t(t, l_max=14)
        u, v, divv = self.u_v_divv(l, m, delta_th, c012, lat, lon, t=t, period=period, peak_flow=peak_flow)
        uSH = self.v2vSH(u.T)
        vSH = self.v2vSH(v.T)
        SV = self.SV_from_flow(vSH, uSH, sh, B_lmax=B_lmax, v_lmax=v_lmax, Nth=Nth)
        return SV

    def vel_accel(self, lat, ph, wave_params, t, c012):
        """compute the horizontal velocity and acceleration of a wave, given its hermite fit constants

        :param c012: hermite fit constants
        :param lat: latitude in [degrees]
        :param ph: longitude in [degrees]
        :param wave_params: tuple or list of wave parameters (l,m,period,vmax, phase, delta_th)
        :param t: time in [years]
        :return: vth, vph, ath, aph
        """
        l, m, period, vmax, phase, delta_th = wave_params
        lon_grid, lat_grid = _np.meshgrid(ph, lat)
        vth1, vph1 = self.horiz_flows_given_coeffs(lat, c012[l], delta_th_override=delta_th)
        v_magnitude = vmax / _np.max(_np.abs((vth1 ** 2 + vph1 ** 2) ** 0.5))
        w = 2 * _np.pi / period * _np.sign(m)
        m = _np.abs(m)
        deg2rad = _np.pi / 180
        phase_offset = _np.mod(phase * deg2rad - w * 2000, 2 * _np.pi)
        vphi = (vph1[None, :] * _np.exp(1j * (m * lon_grid.T * deg2rad + w * t + phase_offset))).T * v_magnitude
        vthi = (vth1[None, :] * _np.exp(1j * (m * lon_grid.T * deg2rad + w * t + phase_offset))).T * v_magnitude
        vph = _np.real(vphi)
        vth = _np.real(vthi)
        aph = _np.real(vphi * 1j * w)
        ath = _np.real(vthi * 1j * w)
        return vth, vph, ath, aph

    def vel_accel_allT(self, wave_params, T, c012, Nth, vmax=1):
        """ computed velocity and acceleration of the wave in units of km/yr and km/yr^2

        :param wave_params: tuple or list of wave parameters (l,m,period,vmax, phase, delta_th)
        :param T:
        :param c012:
        :param Nth:
        :return:
        """

        dlat = 180 / Nth
        lat = _np.linspace(-90 + dlat / 2, 90 - dlat / 2, Nth)
        ph = _np.linspace(dlat / 2, 360 - dlat / 2, Nth * 2)
        vth_t = _np.empty((len(T), len(lat), len(ph)))
        vph_t = _np.empty((len(T), len(lat), len(ph)))
        ath_t = _np.empty((len(T), len(lat), len(ph)))
        aph_t = _np.empty((len(T), len(lat), len(ph)))
        for i, t in enumerate(T):
            vth_t[i, :, :], vph_t[i, :, :], ath_t[i, :, :], aph_t[i, :, :] = self.vel_accel(lat=lat, ph=ph, wave_params=wave_params, t=t, c012=c012)
        vmag = _np.max(_np.sqrt(vth_t ** 2 + vph_t ** 2))
        vth_t = vth_t * vmax / vmag
        vph_t = vph_t * vmax / vmag
        ath_t = ath_t * vmax / vmag
        aph_t = aph_t * vmax / vmag
        return vth_t, vph_t, ath_t, aph_t

    def SV_wave_allT(self, Br_t, dthB_t, dphB_t, vth_t, vph_t, divv_t):
        """ computes the secular variaion from wave motion given the data

        :param Br_t:
        :param dthB_t:
        :param dphB_t:
        :param vth_t:
        :param vph_t:
        :param divv_t:
        :return:
        """
        return -Br_t * divv_t - dthB_t * vth_t - dphB_t * vph_t

    def SA_wave_fluidaccel_allT(self, Br_t, dthB_t, dphB_t, ath_t, aph_t, diva_t):
        """ computes secular acceleration from fluid acceleration given data

        :param Br_t:
        :param dthB_t:
        :param dphB_t:
        :param ath_t:
        :param aph_t:
        :param diva_t:
        :return:
        """
        return -Br_t * diva_t - dthB_t * ath_t - dphB_t * aph_t

    def SA_wave_magSV_allT(self, SV_t, dthSV_t, dphSV_t, vth_t, vph_t, divv_t):
        """ computes seccular acceleration from SV and fluid velocity given data

        :param SV_t:
        :param dthSV_t:
        :param dphSV_t:
        :param vth_t:
        :param vph_t:
        :param divv_t:
        :return:
        """
        return -SV_t * divv_t - dthSV_t * vth_t - dphSV_t * vph_t

    def div_allT(self, v_th_t, v_ph_t, Nth=None, l_max=None):
        """computes the divergence given the velocity (or acceleration) of a vector field at a given set of points

        :param v_th_t:
        :param v_ph_t:
        :param Nth:
        :param l_max:
        :return:
        """
        dth_vth_t, _ = self.gradients_v_allT(v_th_t, Nth=Nth, l_max=l_max)
        _, dph_vph_t = self.gradients_v_allT(v_ph_t, Nth=Nth, l_max=l_max)
        return dth_vth_t + dph_vph_t

    def compute_SASVwave_allT(self, wave_params, T, c012, Nth=None, Bdata=None, B=None, dthB=None, dphB=None,
                              SV=None, dthSV=None, dphSV=None, magmodel=None):
        """ computes the secular acceleration and secular variation over a series of times given all the data required

        :param wave_params: tuple or list of wave parameters (l,m,period,vmax, phase, delta_th)
        :param T:
        :param c012:
        :param Nth:
        :param B:
        :param dthB:
        :param dphB:
        :param SV:
        :param dthSV:
        :param dphSV:
        :param magmodel:
        :return:
        """
        if Bdata is None:
            if ((B is None) or (dthB is None) or (dthB is None)) and (magmodel is not None):
                Bsh = magmodel.get_sht_allT(T)
                if B is None:
                    B = magmodel.B_sht_allT(Bsh)
                if dthB is None or dphB is None:
                    _, dthB, dphB = magmodel.gradB_sht_allT(Bsh)
            if ((SV is None) or (dthSV is None) or (dphSV is None)) and (magmodel is not None):
                SVsh = magmodel.get_SVsht_allT(T)
                if SV is None:
                    SV = magmodel.B_sht_allT(SVsh)
                if dthSV is None or dphSV is None:
                    _, dthSV, dphSV = magmodel.gradB_sht_allT(SVsh)
        else:
            B, dthB, dphB, Bsh, SV, dthSV, dphSV, SVsh, SA, dthSA, dphSA, SAsh = Bdata
        if Nth is None:
            Nth = B.shape[1]
        vth_t, vph_t, ath_t, aph_t = self.vel_accel_allT(wave_params, T, c012, Nth)
        divv_t = self.div_allT(vth_t, vph_t, Nth)
        diva_t = self.div_allT(ath_t, aph_t, Nth)

        SAwave_t = self.SA_wave_fluidaccel_allT(B, dthB, dphB, ath_t, aph_t, diva_t)
        SAwave_t += self.SA_wave_magSV_allT(SV, dthSV, dphSV, vth_t, vph_t, divv_t)
        SVwave_t = self.SV_wave_allT(B, dthB, dphB, vth_t, vph_t, divv_t)
        return SAwave_t, SVwave_t

    def make_SASV_from_phaseperiod_wave_function(self, wave_params, T, c012, Nth, Bdata=None,
                                                 B=None, dthB=None, dphB=None, SV=None, dthSV=None, dphSV=None):
        def SASV_from_phaseperiod(phase, period):
            wp = list(wave_params)
            wp[2] = period
            wp[4] = phase
            return self.compute_SASVwave_allT(wp, T, c012, Nth=Nth, Bdata=Bdata,
                                                           B=B, dthB=dthB, dphB=dphB, SV=SV, dthSV=dthSV, dphSV=dphSV)
        return SASV_from_phaseperiod