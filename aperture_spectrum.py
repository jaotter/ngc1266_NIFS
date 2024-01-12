from astropy.io import fits
from scipy.optimize import curve_fit
from scipy import odr
from astropy.table import Table
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from regions import CirclePixelRegion, CircleSkyRegion, PixCoord, CircleAnnulusPixelRegion
from spectral_cube import SpectralCube
from matplotlib.gridspec import GridSpec
from matplotlib.patches import ConnectionPatch

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const
import pandas as pd
import os

from astropy.cosmology import FlatLambdaCDM

cosmo = FlatLambdaCDM(H0=70., Om0=0.3)


### Script to make full spectrum figure within a given aperture size
### Steps:
### 1. Extract spectrum from aperture
### 2. Fit each line in the spectrum
### 3. Make nice figures

def fit_continuum(spectrum, wave_vel, line_name, degree=1, mask_ind=None):
	## function to do continuum fit for a given line, with hand-selected continuum bands for each line

	line_vel = wave_vel - galv.value

	#chosen by hand for H2_212
	lower_band_kms = [-1700,-1000]
	upper_band_kms = [1000,1700]

	if line_name == 'H2(1-0)S(2)':
		lower_band_kms = [-1900, -1100]
		upper_band_kms = [700, 1100]

	if line_name == 'H2(2-1)S(2)':
		lower_band_kms = [-2000, -1400]
		upper_band_kms = [800, 1250]

	if line_name == 'Brgamma':
		lower_band_kms = [-1000, -500]
		upper_band_kms = [700,1500]

	if line_name == 'H2(1-0)S(0)':
		lower_band_kms = [-1850, -1200]
		upper_band_kms = [1100,1600]

	if line_name == 'H2(2-1)S(3)':
		lower_band_kms = [-1800, -800]
		upper_band_kms = [800,1800]

	if line_name == 'H2(2-1)S(1)':
		lower_band_kms = [-2000, -1000]
		upper_band_kms = [700,1400]


	lower_ind1 = np.where(line_vel > lower_band_kms[0])
	lower_ind2 = np.where(line_vel < lower_band_kms[1])
	lower_ind = np.intersect1d(lower_ind1, lower_ind2)

	upper_ind1 = np.where(line_vel > upper_band_kms[0])
	upper_ind2 = np.where(line_vel < upper_band_kms[1])
	upper_ind = np.intersect1d(upper_ind1, upper_ind2)

	if mask_ind is not None:
		lower_ind = np.setdiff1d(lower_ind, mask_ind)
		upper_ind = np.setdiff1d(upper_ind, mask_ind)

	fit_spec = np.concatenate((spectrum[lower_ind], spectrum[upper_ind]))
	fit_wave = np.concatenate((wave_vel[lower_ind], wave_vel[upper_ind]))

	cont_model = odr.polynomial(degree)
	data = odr.Data(fit_wave, fit_spec)
	odr_obj = odr.ODR(data, cont_model)

	output = odr_obj.run()

	cont_params = output.beta
	#cont_params is [intercept, slope] with velocity as unit

	if degree == 1:
		cont_fit = wave_vel * cont_params[1] + cont_params[0]

	spectrum_contsub = spectrum - cont_fit

	fit_residuals = fit_spec - (fit_wave * cont_params[1] + cont_params[0])
	residuals_std = np.nanstd(fit_residuals)

	sclip_keep_ind = np.where(fit_residuals < 3*residuals_std)[0]
	residuals_clip_std = np.nanstd(fit_residuals[sclip_keep_ind])
	cont_serr = residuals_clip_std / np.sqrt(len(sclip_keep_ind))

	outlier_ind = np.where(fit_residuals > 5*residuals_std)[0]

	if len(outlier_ind) > 0 and mask_ind is None:
		lower_outlier_ind = outlier_ind[np.where(outlier_ind < len(lower_ind))[0]]
		upper_outlier_ind = outlier_ind[np.where(outlier_ind >= len(lower_ind))[0]] - len(lower_ind)

		lower_full_ind = lower_ind[lower_outlier_ind]
		upper_full_ind = upper_ind[upper_outlier_ind]
		full_ind = np.concatenate((lower_full_ind, upper_full_ind))

		cont_params, residuals_clip_std, cont_serr, mask_ind = fit_continuum(spectrum, wave_vel, line_name, degree=1, mask_ind=full_ind)


	return cont_params, residuals_clip_std, cont_serr, mask_ind


def fit_line(spectrum_fit, wave_vel_fit, spectrum_err_fit, cont_params, line_name, start, bounds, xy_loc, runID, ncomp=2, plot=False, mask_ind=None):
	## function to fit line to continuum-subtracted spectrum, with start values, bounds, and varying number of gaussian components

	cont_fit = wave_vel_fit * cont_params[1] + cont_params[0]
	contsub_spec = spectrum_fit - (cont_fit)
	#spectrum_err_fit = None

	print(f'fitting {line_name} for bin {xy_loc}.')

	popt, pcov = curve_fit(gauss_sum, wave_vel_fit, contsub_spec, sigma=spectrum_err_fit, p0=start, bounds=bounds, absolute_sigma=True, maxfev=5000)


	if plot == True:
		fig = plt.figure(figsize=(6,8))
		
		gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=(3,1), hspace=0)
		ax0 = fig.add_subplot(gs[0,0])
		ax1 = fig.add_subplot(gs[1,0], sharex=ax0)

		amp1 = np.round(popt[1])

		sig1 = int(np.round(popt[2]))


		if ncomp == 2:
			amp2 = np.round(popt[4])
			sig2 = int(np.round(popt[5]))
			ax0.plot(wave_vel_fit-galv.value, gauss_sum(wave_vel_fit, *popt[3:6]) + cont_fit, linestyle='-', color='tab:purple', label=fr'Comp 2 (A={amp2}e-18,$\sigma$={sig2})')


		ax0.plot(wave_vel_fit-galv.value, spectrum_fit, color='tab:blue', linestyle='-', marker='.', label='Data')
		ax0.plot(wave_vel_fit-galv.value, gauss_sum(wave_vel_fit, *popt[0:3]) + cont_fit, linestyle='--', color='tab:purple', label=fr'Comp 1 (A={amp1}e-18,$\sigma$={sig1})')
		ax0.plot(wave_vel_fit-galv.value, cont_fit, linestyle='-', color='tab:orange', label='Continuum')
		ax0.plot(wave_vel_fit-galv.value, cont_fit + gauss_sum(wave_vel_fit, *popt), linestyle='-', color='k', label='Full Fit')

		if mask_ind is not None:
			ax0.plot(wave_vel_fit[mask_ind]-galv.value, spectrum_fit[mask_ind], color='tab:red', linestyle='-', marker='x')

		#ax0.text(0, 1.1, start, transform=ax0.transAxes)

		ax0.axvspan(-1700, -1000, color='tab:red', alpha=0.1)
		ax0.axvspan(1000, 1700, color='tab:red', alpha=0.1)

		ax0.legend()

		#ax0.text()

		residuals = spectrum_fit - gauss_sum(wave_vel_fit, *popt) - cont_fit

		ax1.plot(wave_vel_fit-galv.value, residuals, color='tab:red', marker='p', linestyle='-')
		ax1.axhline(0)

		ax0.set_ylabel('Flux (erg/s/cm^2/A)')
		ax1.set_ylabel('Residuals (erg/s/cm^2/A)')

		ax1.set_xlabel('Velocity (km/s)')

		ax0.set_title(f'{line_name} fit for bin {xy_loc}' )
		plt.subplots_adjust(hspace=0)

		savename = f'{line_name}_x{xy_loc[0]}y{xy_loc[1]}_fit.png'
		savepath = f'/Users/jotter/highres_PSBs/ngc1266_NIFS/plots/gaussfit/{runID}/'
		if os.path.exists(savepath) == False:
			os.mkdir(savepath)
		plt.savefig(f'{savepath}{savename}')
		print(f'Figure saved as {savename}')

		plt.close()

	return popt, pcov


def compute_flux(popt, pcov, line_wave, ncomp=2):
	#take fit parameter output and compute flux and error

	width1 = popt[2] * u.km/u.s
	amp1 = popt[1] #* u.erg/u.s/u.cm**2/u.AA
	wave_width1 = ((width1 / const.c) * line_wave).to(u.angstrom).value

	amp1_err = np.sqrt(pcov[1,1]) #wrong with unit conversion
	width1_err = np.sqrt(pcov[2,2]) * u.km/u.s
	wave_width1_err = ((width1_err / const.c) * line_wave).to(u.angstrom).value

	if ncomp == 2:
		width2 = popt[5] * u.km/u.s
		amp2 = popt[4] #* u.erg/u.s/u.cm**2/u.AA
		wave_width2 = ((width2 / const.c) * line_wave).to(u.angstrom).value

		amp2_err = np.sqrt(pcov[4,4])
		width2_err = np.sqrt(pcov[5,5]) * u.km/u.s
		wave_width2_err = ((width2_err / const.c) * line_wave).to(u.angstrom).value

		total_flux = np.sqrt(2 * np.pi) * ((wave_width1 * amp1) + (wave_width2 * amp2))
		total_flux_err = np.sqrt(2*np.pi) * np.sqrt((wave_width1*amp1_err)**2 + (amp1*wave_width1_err)**2 + (wave_width2*amp2_err)**2 + (amp2*wave_width2_err)**2)

	else:
		total_flux = np.sqrt(2 * np.pi) * ((wave_width1 * amp1))
		total_flux_err = np.sqrt(2*np.pi) * np.sqrt((wave_width1*amp1_err)**2 + (amp1*wave_width1_err)**2)


	return total_flux, total_flux_err

def gauss_sum(velocity, *params):
	#velocity is velocity array for given line
	#params must be divisible by 3, should be center (velocity), amplitude, width

	y = np.zeros_like(velocity)

	for i in range(0, len(params)-1, 3):
		ctr = params[i]
		amp = params[i+1]
		wid = params[i+2]
		y = y + amp * np.exp( -(velocity - ctr)**2/(2*wid)**2)
	return y


def extract_spectrum(ap_shape, ap_dimensions=None, pix_center=ConnectionPatch):
	#ap_shape - either "circle" or "annulus", or "fov" for entire fov minus edges
	#ap_dimensions - [radius] for circle, [inner radius, outer radius] for annulus in length, angular, or dimensionless (pixel) units
	#pix_center - center of aperture in pixel coordinates, if None use galaxy center

	cube_path = '/Users/jotter/highres_PSBs/ngc1266_data/NIFS_data/NGC1266_NIFS_final_trim_wcs2.fits'
	#cube_path = '/Users/jotter/highres_PSBs/ngc1266_data/NIFS_data/NGC1266_NIFS_dec6_wcs.fits'

	cube_fl = fits.open(cube_path)	
	cube_data = cube_fl[1].data
	nifs_wcs = WCS(cube_fl[1].header)
	cube_fl.close()

	cube = SpectralCube(data=cube_data, wcs=nifs_wcs)

	cube_shape = cube_data.shape

	if ap_shape == 'fov':
		trunc_cube = cube[1:cube_shape[0]-1, 1:cube_shape[1]]
		fov_spectrum = trunc_cube.sum(axis=(1,2))

		return fov_spectrum

	else:
		#convert dimensions from kpc to arcsec
		z = 0.007214
		D_L = cosmo.luminosity_distance(z).to(u.Mpc)
		as_per_kpc = cosmo.arcsec_per_kpc_comoving(z)

		if u.get_physical_type(ap_dimensions.unit) == 'length':
			ap_dimensions_as = (ap_dimensions * as_per_kpc).decompose()
			ap_dimensions_pix = (ap_dimensions_as / (0.043*u.arcsecond)).decompose().value
		elif u.get_physical_type(ap_dimensions.unit) == 'angle':
			ap_dimensions_as = ap_dimensions.to(u.arcsecond)
			ap_dimensions_pix = (ap_dimensions_as / (0.043*u.arcsecond)).decompose().value
		elif u.get_physical_type(ap_dimensions.unit) == 'dimensionless': #assume pixel units
			ap_dimensions_as = (ap_dimensions * as_per_kpc).decompose()
			ap_dimensions_pix = ap_dimensions.value

		print(f'Aperture radius in pixel: {ap_dimensions_pix}')

		if pix_center == None:
			gal_cen = SkyCoord(ra='3:16:00.74576', dec='-2:25:38.70151', unit=(u.hourangle, u.degree),
			                  frame='icrs') #from HST H band image, by eye

			center_pix = nifs_wcs.celestial.all_world2pix(gal_cen.ra, gal_cen.dec, 0)

			center_pixcoord = PixCoord(center_pix[0], center_pix[1])

		else:
			center_pixcoord = PixCoord(pix_center[0], pix_center[1])

		if ap_shape == 'circle':
			ap = CirclePixelRegion(center=center_pixcoord, radius=ap_dimensions_pix[0])

		elif ap_shape == 'annulus':
			ap = CircleAnnulusPixelRegion(center=center_pixcoord, inner_radius=ap_dimensions_pix[0], outer_radius=ap_dimensions_pix[1])

		mask_cube = cube.subcube_from_regions([ap])

		spectrum = mask_cube.sum(axis=(1,2))

		return spectrum


def fit_spectrum(spectrum, line_dict, savename):
	#given aperture spectrum and list of lines, fit each one, return dictionary with fit parameters for each

	return_dict = {}

	for i, line_name in enumerate(line_dict.keys()):
		line_wave = line_dict[line_name]

		spec_kms = spectrum.with_spectral_unit(u.km/u.s, velocity_convention='optical', rest_value=line_wave*u.micron)

		fit_ind1 = np.where(spec_kms.spectral_axis > galv-2000* u.km/u.s)
		fit_ind2 = np.where(spec_kms.spectral_axis < galv+2000* u.km/u.s)
		fit_ind = np.intersect1d(fit_ind1, fit_ind2)

		if line_name == 'Brgamma':
			tell_ind_range = [795,805]
			mask_start = tell_ind_range[0] - fit_ind[0]

			fit_ind = np.concatenate((fit_ind[0:mask_start], fit_ind[mask_start+10:-1]))


		fit_spec = spec_kms[fit_ind]
		fit_spec_vel = spec_kms.spectral_axis[fit_ind].value

		cont_params, cont_std, cont_serr, mask_ind = fit_continuum(fit_spec, fit_spec_vel, line_name, degree=1, mask_ind=None)
		#cont_fit = fit_spec_vel * cont_params[1] + cont_params[0]
		cont_std_err = np.repeat(cont_std, len(fit_spec))

		fit_ind3 = np.where(spec_kms.spectral_axis > galv-1000* u.km/u.s)
		fit_ind4 = np.where(spec_kms.spectral_axis < galv+1000* u.km/u.s)
		fit_ind_line = np.intersect1d(fit_ind3, fit_ind4)
		fit_spec_line = spec_kms[fit_ind_line]

		peak_flux = np.nanmax(fit_spec_line)
		peak_ind = np.where(fit_spec == peak_flux)[0][0]
		peak_vel = fit_spec_vel[peak_ind]

		if np.abs(peak_vel - galv.value) > 500:
			peak_vel = galv.value

		start = [peak_vel, peak_flux, 200, peak_vel, peak_flux*0.25, 400]
		bounds = ([galv.value - 500, peak_flux * 0.01, 20, galv.value - 500, 0, 50], [galv.value + 500, peak_flux*3, 600, galv.value + 500, peak_flux*10, 600])
		ncomp = 2

		if line_name == 'Brgamma':
			start = start[0:3]
			bounds = (bounds[0][0:3], bounds[1][0:3])
			ncomp = 1
			

		popt, pcov = fit_line(fit_spec, fit_spec_vel, cont_std_err, cont_params, line_name, start, bounds, (0,0), savename, ncomp=ncomp, plot=True, mask_ind=mask_ind)

		fit_spec_wave = spectrum.spectral_axis[fit_ind].to(u.Angstrom)

		return_dict[line_name] = [popt, pcov, cont_params]

	return return_dict


def spectrum_figure(spectrum, fit_dict, savename):
	fig = plt.figure(figsize=(10,10))
	gs = GridSpec(4,4, wspace=0.25, hspace=0.6)

	ax0 = fig.add_subplot(gs[1:3, 0:4])

	ax0.plot(spectrum.spectral_axis.to(u.micron), spectrum*1e16)

	ax0.set_xlim(2.005, 2.43)
	ax0.set_xlabel(r'Observed Wavelength ($\mu$m)', size=12)
	ax0.tick_params(axis='both', labelsize=12)

	ax0.set_ylabel(r'Flux (10$^{-16}$ erg/s/cm$^2$/$\AA$)', size=14)

	ax1 = fig.add_subplot(gs[0,0])
	ax2 = fig.add_subplot(gs[0,1])
	ax3 = fig.add_subplot(gs[0,2])
	ax4 = fig.add_subplot(gs[0,3])

	ax5 = fig.add_subplot(gs[3,0])
	ax6 = fig.add_subplot(gs[3,1])
	ax7 = fig.add_subplot(gs[3,2])
	ax8 = fig.add_subplot(gs[3,3])

	line_ax = [ax1, ax5, ax2, ax6, ax3, ax7, ax4, ax8]
	line_name_list_full = [r'H$_2$(1-0)S(2)', r'H$_2$(2-1)S(3)', r'H$_2$(1-0)S(1)', r'H$_2$(2-1)S(2)', r'Br$\gamma$', r'H$_2$(1-0)S(0)', r'H$_2$(2-1)S(1)', r'H$_2$(1-0)Q(1)']
	color_list = ['tab:blue', 'tab:red', 'tab:orange', 'tab:green', 'tab:pink', 'tab:olive', 'tab:cyan', 'tab:brown']

	for i, line_name in enumerate(line_dict):
		ax = line_ax[i]
		line_wave = line_dict[line_name]
		fit_returns = fit_dict[line_name]
		fit_params = fit_returns[0]
		cont_params = fit_returns[2]

		ax.set_title(line_name_list_full[i])

		spec_kms = spectrum.with_spectral_unit(u.km/u.s, velocity_convention='optical', rest_value=line_wave*u.micron)

		fit_ind1 = np.where(spec_kms.spectral_axis > galv-2000* u.km/u.s)
		fit_ind2 = np.where(spec_kms.spectral_axis < galv+2000* u.km/u.s)
		fit_ind = np.intersect1d(fit_ind1, fit_ind2)
		fit_spec = spec_kms[fit_ind]
		fit_spec_vel = spec_kms.spectral_axis[fit_ind].value

		if line_name == 'Brgamma':
			tell_ind_range = [795,805]
			mask_start = tell_ind_range[0] - fit_ind[0]

			ax.axvspan(fit_spec_vel[mask_start]-galv.value, fit_spec_vel[mask_start+10]-galv.value, color='k', alpha=0.2)

		ax.plot(fit_spec_vel - galv.value, fit_spec*1e16)

		cont_fit = fit_spec_vel * cont_params[1] + cont_params[0]
		contsub_spec = fit_spec - cont_fit

		ax.plot(fit_spec_vel-galv.value, (gauss_sum(fit_spec_vel, *fit_params[0:3]) + cont_fit)*1e16, linestyle='--', color='tab:purple')
		if line_name != 'Brgamma':
			ax.plot(fit_spec_vel-galv.value, (gauss_sum(fit_spec_vel, *fit_params[3:6]) + cont_fit)*1e16, linestyle='-', color='tab:purple')
		ax.plot(fit_spec_vel-galv.value, cont_fit*1e16, linestyle='-', color='tab:orange')
		ax.plot(fit_spec_vel-galv.value, (cont_fit + gauss_sum(fit_spec_vel, *fit_params))*1e16, linestyle='-', color='k')

		ax.set_xlabel('Velocity (km/s)', fontsize=12)

		ax.tick_params(axis='both', labelsize=12)

		if i == 5 and savename == 'circ_100pc':
			ax.set_yticks([8,10,12])

		if i != 7:
			ax.set_xlim(-1250,1250)
			ax.set_xticks([-1000, 0, 1000])
			ax.axvspan(-1500,1500, color=color_list[i], alpha=0.2)

		else:
			ax.set_xlim(-1500, 500)
			ax.set_xticks([-1000, 0, 500])
			ax.axvspan(-1500,500, color=color_list[i], alpha=0.2)

		obs_wave = line_wave * (1+z)
		ax0.axvspan(obs_wave-5e-3, obs_wave+5e-3, color=color_list[i], alpha=0.2)

	#plt.savefig(f'plots/NIFS_{savename}_spec_noarr.png', dpi=300, bbox_inches='tight')

	#xy_1 = [2.048, 18]
	#xy_1 = [2.048, 28]
	xy_1 = [0.102, 0.9]
	xy_ax1 = [0.5, -0.4]
	arrow1 = ConnectionPatch(xyA=xy_1, coordsA=ax0.transAxes, xyB=xy_ax1, coordsB=ax1.transAxes, arrowstyle='-|>', fc=color_list[0])
	fig.add_artist(arrow1)

	#xy_2 = [2.14, 30]
	#xy_2 = [2.14, 28]
	xy_2 = [0.315, 0.97]
	xy_ax2 = [0.5, -0.4]
	arrow2 = ConnectionPatch(xyA=xy_2, coordsA=ax0.transAxes, xyB=xy_ax2, coordsB=ax2.transAxes, arrowstyle='-|>', fc=color_list[2])
	fig.add_artist(arrow2)

	#xy_3 = [2.182, 10]
	#xy_3 = [2.182, 28]
	xy_3 = [0.415, 0.9]
	xy_ax3 = [0.5, -0.4]
	arrow3 = ConnectionPatch(xyA=xy_3, coordsA=ax0.transAxes, xyB=xy_ax3, coordsB=ax3.transAxes, arrowstyle='-|>', fc=color_list[4])
	fig.add_artist(arrow3)

	#xy_4 = [2.265, 11]
	#xy_4 = [2.265, 28]
	xy_4 = [0.61, 0.9]
	xy_ax4 = [0.5, -0.4]
	arrow4 = ConnectionPatch(xyA=xy_4, coordsA=ax0.transAxes, xyB=xy_ax4, coordsB=ax4.transAxes, arrowstyle='-|>', fc=color_list[6])
	fig.add_artist(arrow4)

	#xy_5 = [2.087, 8]
	#xy_5 = [2.087, 6]
	xy_5 = [0.2, 0.1]
	xy_ax5 = [0.5, 1.2]
	arrow5 = ConnectionPatch(xyA=xy_5, coordsA=ax0.transAxes, xyB=xy_ax5, coordsB=ax5.transAxes, arrowstyle='-|>', fc=color_list[1])
	fig.add_artist(arrow5)

	#xy_6 = [2.169, 8]
	#xy_6 = [2.169, 6]
	xy_6 = [0.39, 0.1]
	xy_ax6 = [0.3, 1.2]
	arrow6 = ConnectionPatch(xyA=xy_6, coordsA=ax0.transAxes, xyB=xy_ax6, coordsB=ax6.transAxes, arrowstyle='-|>', fc=color_list[3])
	fig.add_artist(arrow6)

	#xy_7 = [2.24, 8]
	#xy_7 = [2.24, 6]
	xy_7 = [0.55, 0.1]
	xy_ax7 = [0.7, 1.2]
	arrow7= ConnectionPatch(xyA=xy_7, coordsA=ax0.transAxes, xyB=xy_ax7, coordsB=ax7.transAxes, arrowstyle='-|>', fc=color_list[5])
	fig.add_artist(arrow7)

	#xy_8 = [2.423, 6]
	xy_8 = [0.98, 0.095]
	xy_ax8 = [0.5, 1.2]
	arrow8 = ConnectionPatch(xyA=xy_8, coordsA=ax0.transAxes, xyB=xy_ax8, coordsB=ax8.transAxes, arrowstyle='-|>', fc=color_list[7])
	fig.add_artist(arrow8)

	plt.savefig(f'plots/NIFS_{savename}_spec.pdf', dpi=300, bbox_inches='tight')


def spectrum_figure_multispec(spectrum_list, fit_dict_list, savename):
	fig = plt.figure(figsize=(10,10))
	gs = GridSpec(4,4, wspace=0.25, hspace=0.6)

	ax0 = fig.add_subplot(gs[1:3, 0:4])

	ax0.plot(spectrum.spectral_axis.to(u.micron), spectrum*1e16)

	ax0.set_xlim(2.005, 2.43)
	ax0.set_xlabel(r'Observed Wavelength ($\mu$m)', size=12)
	ax0.tick_params(axis='both', labelsize=12)

	ax0.set_ylabel(r'Flux (10$^{-16}$ erg/s/cmf$^2$/$\AA$)', size=14)

	ax1 = fig.add_subplot(gs[0,0])
	ax2 = fig.add_subplot(gs[0,1])
	ax3 = fig.add_subplot(gs[0,2])
	ax4 = fig.add_subplot(gs[0,3])

	ax5 = fig.add_subplot(gs[3,0])
	ax6 = fig.add_subplot(gs[3,1])
	ax7 = fig.add_subplot(gs[3,2])
	ax8 = fig.add_subplot(gs[3,3])

	line_ax = [ax1, ax5, ax2, ax6, ax3, ax7, ax4, ax8]
	line_name_list_full = [r'H$_2$(1-0)S(2)', r'H$_2$(2-1)S(3)', r'H$_2$(1-0)S(1)', r'H$_2$(2-1)S(2)', r'Br$\gamma$', r'H$_2$(1-0)S(0)', r'H$_2$(2-1)S(1)', r'H$_2$(1-0)Q(1)']
	color_list = ['tab:blue', 'tab:red', 'tab:orange', 'tab:green', 'tab:pink', 'tab:olive', 'tab:cyan', 'tab:brown']

	for i, line_name in enumerate(line_dict):
		ax = line_ax[i]
		line_wave = line_dict[line_name]
		fit_returns = fit_dict[line_name]
		fit_params = fit_returns[0]
		cont_fit = fit_returns[2]

		ax.set_title(line_name_list_full[i])

		spec_kms = spectrum.with_spectral_unit(u.km/u.s, velocity_convention='optical', rest_value=line_wave*u.micron)

		fit_ind1 = np.where(spec_kms.spectral_axis > galv-2000* u.km/u.s)
		fit_ind2 = np.where(spec_kms.spectral_axis < galv+2000* u.km/u.s)
		fit_ind = np.intersect1d(fit_ind1, fit_ind2)
		fit_spec = spec_kms[fit_ind]
		fit_spec_vel = spec_kms.spectral_axis[fit_ind].value

		ax.plot(fit_spec_vel - galv.value, fit_spec*1e16)

		contsub_spec = fit_spec - cont_fit

		ax.plot(fit_spec_vel-galv.value, (gauss_sum(fit_spec_vel, *fit_params[0:3]) + cont_fit)*1e16, linestyle='--', color='tab:purple')
		ax.plot(fit_spec_vel-galv.value, (gauss_sum(fit_spec_vel, *fit_params[3:6]) + cont_fit)*1e16, linestyle='-', color='tab:purple')
		ax.plot(fit_spec_vel-galv.value, cont_fit*1e16, linestyle='-', color='tab:orange')
		ax.plot(fit_spec_vel-galv.value, (cont_fit + gauss_sum(fit_spec_vel, *fit_params))*1e16, linestyle='-', color='k')

		ax.set_xlabel('Velocity (km/s)', fontsize=12)

		ax.tick_params(axis='both', labelsize=12)

		if i == 5 and savename == 'circ_100pc':
			ax.set_yticks([8,10,12])

		if i != 7:
			ax.set_xlim(-1250,1250)
			ax.set_xticks([-1000, 0, 1000])
			ax.axvspan(-1500,1500, color=color_list[i], alpha=0.2)

		else:
			ax.set_xlim(-1500, 500)
			ax.set_xticks([-1000, 0, 500])
			ax.axvspan(-1500,500, color=color_list[i], alpha=0.2)

		obs_wave = line_wave * (1+z)
		ax0.axvspan(obs_wave-5e-3, obs_wave+5e-3, color=color_list[i], alpha=0.2)

	#xy_1 = [2.048, 18]
	#xy_1 = [2.048, 28]
	xy_1 = [0.102, 0.9]
	xy_ax1 = [0.5, -0.4]
	arrow1 = ConnectionPatch(xyA=xy_1, coordsA=ax0.transAxes, xyB=xy_ax1, coordsB=ax1.transAxes, arrowstyle='-|>', fc=color_list[0])
	fig.add_artist(arrow1)

	#xy_2 = [2.14, 30]
	#xy_2 = [2.14, 28]
	xy_2 = [0.315, 0.97]
	xy_ax2 = [0.5, -0.4]
	arrow2 = ConnectionPatch(xyA=xy_2, coordsA=ax0.transAxes, xyB=xy_ax2, coordsB=ax2.transAxes, arrowstyle='-|>', fc=color_list[2])
	fig.add_artist(arrow2)

	#xy_3 = [2.182, 10]
	#xy_3 = [2.182, 28]
	xy_3 = [0.415, 0.9]
	xy_ax3 = [0.5, -0.4]
	arrow3 = ConnectionPatch(xyA=xy_3, coordsA=ax0.transAxes, xyB=xy_ax3, coordsB=ax3.transAxes, arrowstyle='-|>', fc=color_list[4])
	fig.add_artist(arrow3)

	#xy_4 = [2.265, 11]
	#xy_4 = [2.265, 28]
	xy_4 = [0.61, 0.9]
	xy_ax4 = [0.5, -0.4]
	arrow4 = ConnectionPatch(xyA=xy_4, coordsA=ax0.transAxes, xyB=xy_ax4, coordsB=ax4.transAxes, arrowstyle='-|>', fc=color_list[6])
	fig.add_artist(arrow4)

	#xy_5 = [2.087, 8]
	#xy_5 = [2.087, 6]
	xy_5 = [0.2, 0.1]
	xy_ax5 = [0.5, 1.2]
	arrow5 = ConnectionPatch(xyA=xy_5, coordsA=ax0.transAxes, xyB=xy_ax5, coordsB=ax5.transAxes, arrowstyle='-|>', fc=color_list[1])
	fig.add_artist(arrow5)

	#xy_6 = [2.169, 8]
	#xy_6 = [2.169, 6]
	xy_6 = [0.39, 0.1]
	xy_ax6 = [0.3, 1.2]
	arrow6 = ConnectionPatch(xyA=xy_6, coordsA=ax0.transAxes, xyB=xy_ax6, coordsB=ax6.transAxes, arrowstyle='-|>', fc=color_list[3])
	fig.add_artist(arrow6)

	#xy_7 = [2.24, 8]
	#xy_7 = [2.24, 6]
	xy_7 = [0.55, 0.1]
	xy_ax7 = [0.7, 1.2]
	arrow7= ConnectionPatch(xyA=xy_7, coordsA=ax0.transAxes, xyB=xy_ax7, coordsB=ax7.transAxes, arrowstyle='-|>', fc=color_list[5])
	fig.add_artist(arrow7)

	#xy_8 = [2.423, 6]
	xy_8 = [0.98, 0.095]
	xy_ax8 = [0.5, 1.2]
	arrow8 = ConnectionPatch(xyA=xy_8, coordsA=ax0.transAxes, xyB=xy_ax8, coordsB=ax8.transAxes, arrowstyle='-|>', fc=color_list[7])
	fig.add_artist(arrow8)

	plt.savefig(f'plots/NIFS_{savename}_spec.pdf', dpi=300, bbox_inches='tight')



def save_fluxes(fit_dict, savename):

	save_tab = Table()

	save_tab['Flux_type'] = ['Total', 'Comp1', 'Comp2']

	for i, line_name in enumerate(fit_dict.keys()):
		line_wave = line_dict[line_name]

		fit_returns = fit_dict[line_name]
		fit_params = fit_returns[0]
		fit_cov = fit_returns[1]
		cont_params = fit_returns[2]

		ncomp = 2
		if line_name == 'Brgamma':
			ncomp = 1

		obs_wave = line_wave * (1+z) * u.micron
		total_flux, total_flux_err = compute_flux(fit_params, fit_cov, obs_wave, ncomp=ncomp)

		comp1_flux, comp1_flux_err = compute_flux(fit_params[0:3], fit_cov[0:3], obs_wave, ncomp=1)

		if ncomp == 2:
			comp2_flux, comp2_flux_err = compute_flux(fit_params[3:6], fit_cov[3:6], obs_wave, ncomp=1)

			save_tab[f'{line_name}_flux'] = [total_flux, comp1_flux, comp2_flux]
			save_tab[f'{line_name}_flux_err'] = [total_flux_err, comp1_flux_err, comp2_flux_err]

		else: 
			save_tab[f'{line_name}_flux'] = [total_flux, comp1_flux, np.nan]
			save_tab[f'{line_name}_flux_err'] = [total_flux_err, comp1_flux_err, np.nan]

	save_tab.write(f'/Users/jotter/highres_PSBs/ngc1266_NIFS/fit_output/{savename}.csv', format='csv', overwrite=True)


'''def save_fluxes(fit_dict, savename):

	save_tab = Table()

	save_tab['Flux_type'] = ['Total', 'Comp1', 'Comp2']

	for i, line_name in enumerate(fit_dict.keys()):

		line_wave = line_dict[line_name]

		fit_returns = fit_dict[line_name]
		fit_params = fit_returns[0]
		fit_cov = fit_returns[1]
		cont_fit = fit_returns[2]

		obs_wave = line_wave * (1+z) * u.micron
		total_flux, total_flux_err = compute_flux(fit_params, fit_cov, obs_wave)

		comp1_flux, comp1_flux_err = compute_flux(fit_params[0:3], fit_cov[0:3], obs_wave)
		comp2_flux, comp2_flux_err = compute_flux(fit_params[3:6], fit_cov[3:6], obs_wave)

		save_tab[f'{line_name}_flux'] = [total_flux, comp1_flux, comp2_flux]
		save_tab[f'{line_name}_flux_err'] = [total_flux_err, comp1_flux_err, comp2_flux_err]

	save_tab.write(f'/Users/jotter/highres_PSBs/ngc1266_NIFS/fit_output/{savename}.csv', format='csv', overwrite=True)
'''



z = 0.007214
galv = np.log(z+1)*const.c.to(u.km/u.s)
line_dict = {'H2(1-0)S(2)':2.0332, 'H2(2-1)S(3)':2.0735, 'H2(1-0)S(1)':2.1213, 'H2(2-1)S(2)':2.1542, 'Brgamma':2.1654, 'H2(1-0)S(0)':2.2230, 'H2(2-1)S(1)':2.2470, 'H2(1-0)Q(1)':2.4066}

#spec = extract_spectrum('annulus', [150, 300]*u.pc)
#fit_dict = fit_spectrum(spec, line_dict, 'annulus_300pc')
#spectrum_figure(spec, fit_dict, 'annulus_300pc')
#save_fluxes(fit_dict, 'NIFS_ann_300pc_fluxes')

#pixcent=None
#spec = extract_spectrum('circle', [50]*u.pc, pixcent)
#fit_dict = fit_spectrum(spec, line_dict, 'circ_50pc_cent_mask')
#spectrum_figure(spec, fit_dict, 'circ_50pc_cent_mask')
#save_fluxes(fit_dict, 'NIFS_50pc_fluxes_cent_mask')

fov_spec = extract_spectrum('fov', None, None)
fit_dict = fit_spectrum(fov_spec, line_dict, 'fov_spectrum')
spectrum_figure(fov_spec, fit_dict, 'fov_spectrum')
save_fluxes(fit_dict, 'NIFS_fov_spectrum_fluxes')

#pixcent=[22,42]
#spec = extract_spectrum('circle', [50]*u.pc, pixcent)
#fit_dict = fit_spectrum(spec, line_dict, 'circ_50pc_x22y42_mask')
#spectrum_figure(spec, fit_dict, 'circ_50pc_x22y42_mask')
#save_fluxes(fit_dict, 'NIFS_50pc_fluxes_x22y42_mask')

#pixcent=[62,22]
#spec = extract_spectrum('circle', [50]*u.pc, pixcent)
#fit_dict = fit_spectrum(spec, line_dict, 'circ_50pc_x62y22_mask')
#spectrum_figure(spec, fit_dict, 'circ_50pc_x62y22_mask')
#save_fluxes(fit_dict, 'NIFS_50pc_fluxes_x62y22_mask')

#pixcent=[52,66]
#spec = extract_spectrum('circle', [50]*u.pc, pixcent)
#fit_dict = fit_spectrum(spec, line_dict, 'circ_50pc_x52y66_bg')
#spectrum_figure(spec, fit_dict, 'circ_50pc_x52y66_bg')



