from astropy.io import fits
from scipy.optimize import curve_fit
from scipy import odr
from astropy.table import Table
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from regions import CirclePixelRegion, CircleSkyRegion, PixCoord, CircleAnnulusPixelRegion
from spectral_cube import SpectralCube
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import ConnectionPatch, Circle, Rectangle
from matplotlib.lines import Line2D
from matplotlib import ticker

import glob
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const
import pandas as pd
import os
import datetime
from time import perf_counter as clock
import gc

from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util
import ppxf.miles_util as lib
import ppxf_custom_util_nifs as custom_util

from astropy.stats import mad_std, bayesian_info_criterion_lsq
from scipy.interpolate import interp1d
from astropy.cosmology import FlatLambdaCDM

cosmo = FlatLambdaCDM(H0=70., Om0=0.3)

z = 0.007214
galv = np.log(z+1)*const.c.to(u.km/u.s)
line_dict = {'H2(1-0)S(2)':2.0332, 'He I':2.0587, 'H2(2-1)S(3)':2.0735, 'H2(1-0)S(1)':2.1213, 'H2(2-1)S(2)':2.1542, 'Brgamma':2.1654, 'H2(1-0)S(0)':2.2230, 'H2(2-1)S(1)':2.2470, 'H2(1-0)Q(1)':2.4066}
line_dict_noHe = {'H2(1-0)S(2)':2.0332, 'H2(2-1)S(3)':2.0735, 'H2(1-0)S(1)':2.1213, 'H2(2-1)S(2)':2.1542, 'Brgamma':2.1654, 'H2(1-0)S(0)':2.2230, 'H2(2-1)S(1)':2.2470, 'H2(1-0)Q(1)':2.4066}
line_names_save = ['H2_10_S2','He I','H2_21_S3','H2_10_S1', 'H2_21_S2', 'Brgamma', 'H2_10_S0', 'H2_21_S1', 'H2_10_Q1']

### Script to fit extracted spectrum from aperture with ppxf

def load_xsl_temps(data_wave, temp_folder="/Users/jotter/ppxf_files/XSL_DR3_release/"):
	
	temp_paths_1 = glob.glob(temp_folder+"*_merged.fits")
	temp_paths_2 = glob.glob(temp_folder+"*_merged_scl.fits")
	temp_paths = np.concatenate((temp_paths_1, temp_paths_2))

	temp_starts = []
	temp_delts = []
	temp_ends = []

	temp_data_nobin = []
	temp_wave_nobin = []

	for tp in temp_paths:
		temp_fl = fits.open(tp)
		temp_data = temp_fl[1].data

		if tp[-8::] == "scl.fits":
			temp_spec = temp_data['FLUX_SC'].copy()
		else:
			temp_spec = temp_data['FLUX_DR'].copy()

		temp_wave = ((temp_data['WAVE'].copy() * u.nm).to(u.Angstrom)).value

		del temp_fl[1].data
		del temp_fl[0].data
		del temp_data

		temp_fl.close()

		temp_start = temp_wave[0]
		temp_end = temp_wave[-1]

		temp_starts.append(temp_start)
		temp_ends.append(temp_end)

		temp_data_nobin.append(temp_spec)
		temp_wave_nobin.append(temp_wave)

	max_start = np.max(temp_starts)
	min_end = np.min(temp_ends)

	#print(max_start, min_end)

	wave_low_ind = np.searchsorted(data_wave, max_start)
	wave_high_ind = np.searchsorted(data_wave, min_end)

	#print(data_wave)
	#print(wave_low_ind, wave_high_ind)

	wave_trunc = data_wave[wave_low_ind:wave_high_ind]

	temp_rebin = []

	for i in range(len(temp_paths)):
		interp_func = interp1d(temp_wave_nobin[i], temp_data_nobin[i])
		new_temp = interp_func(wave_trunc)
		temp_rebin.append(new_temp)

	print(len(temp_paths))

	return np.array(temp_rebin).transpose(), wave_trunc #to be consistent with ppxf MILES loading


def output_ppxf_fit_plot(plot_name, pp, good_pix_total, logLam, vel_comp0, z, ncomp, mad_std_residuals, SN_wMadStandardDev, fit_gas):

	print(f'Saving ppxf plot')
	plot_dir = 'ppxf_output/plots/'

	fig, ax = plt.subplots(figsize = (9,6), sharey = True)
	ax1 = plt.subplot(212)
	ax1.margins(0.08) 
	ax2 = plt.subplot(221)
	ax2.margins(0.05)
	ax3 = plt.subplot(222)
	ax3.margins(0.05)

	wave_plot_rest = np.exp(logLam)/(z+1)

	wave_lam_rest_ind = np.arange(len(wave_plot_rest))
	masked_ind = np.setdiff1d(wave_lam_rest_ind, good_pix_total)
	mask_reg_upper = []
	mask_reg_lower = []
	for ind in masked_ind:
		if ind+1 not in masked_ind:
			mask_reg_lower.append(wave_plot_rest[ind])
		elif ind-1 not in masked_ind:
			mask_reg_upper.append(wave_plot_rest[ind])

	fig.text(0.05, 0.93, f'Mean abs. dev. of residuals: {np.round(mad_std_residuals,1)}, S/N: {int(np.round(SN_wMadStandardDev,0))}')
	fig.text(0.45, 0.93, f'Chi-Squared/DOF: {np.round(pp.chi2,3)}')

	bin_stell_vel = vel_comp0 - z*(const.c.to(u.km/u.s)).value
	fig.text(0.7, 0.93, f'Stellar velocity: {int(np.round(bin_stell_vel,0))} km/s')

	fig.text(0.03, 0.3, r'Flux (10$^{-20}$ erg/s/cm$^2$/Å)', fontsize = 12, rotation=90)

	for bound_ind in range(len(mask_reg_upper)):
		ax1.axvspan(mask_reg_lower[bound_ind], mask_reg_upper[bound_ind], alpha=0.25, color='gray')
		ax2.axvspan(mask_reg_lower[bound_ind], mask_reg_upper[bound_ind], alpha=0.25, color='gray')
		ax3.axvspan(mask_reg_lower[bound_ind], mask_reg_upper[bound_ind], alpha=0.25, color='gray')

	plt.sca(ax1)
	pp.plot()
	xticks = ax1.get_xticks()
	#ax1.set_xticks(xticks, labels=np.array(xticks*1e4, dtype='int'))
	ax1.set_xlabel(r'Restframe Wavelength ($\mu$m)',fontsize = 12)
	ax1.set_ylabel('')
	legend_elements = [Line2D([0], [0], color='k', label='Data', lw=2),
						Line2D([0], [0], color='r', label='Stellar continuum', lw=2),
						Line2D([0], [0], marker='d', color='g', label='Residuals', markersize=5, lw=0),
						Line2D([0], [0], color='b', label='Masked regions', lw=2),]
	ax1.legend(handles=legend_elements, loc='upper right', fontsize='small', ncol=2)


	plt.sca(ax2)
	pp.plot()
	ax2.set_xlim(2.1,2.2)
	xticks = ax2.get_xticks()
	#ax2.set_xticks(xticks, labels=np.array(np.round(xticks*1e4, -1), dtype='int'))
	ax2.set_ylabel('')
	ax2.set_xlabel('')
	ax2.set_title(r'Zoom-in on H2 lines', fontsize = 12)
	
	plt.sca(ax3)
	pp.plot()
	ax3.set_xlim(2.3,2.4)
	xticks = ax3.get_xticks()
	#ax3.set_xticks(xticks, labels=np.array(np.round(xticks*1e4, -1), dtype='int'))
	ax3.set_title(r'Zoom-in on CO bandheads', fontsize = 12)
	ax3.set_yticklabels([])
	ax3.set_xlabel('')
	ax3.set_ylabel('')

	full_plot_dir = f'{plot_dir}/ppxf_aperture/'
	plot_fl = f'{full_plot_dir}/ppxf_stellarfit_{plot_name}.png'


	
	if os.path.exists(full_plot_dir) == False:
		os.mkdir(full_plot_dir)
	plt.savefig(plot_fl, dpi = 300) 
	plt.close()

	print(f'Saved ppxf plot to {plot_name}')


def fit_line(contsub_spec, wave_vel_fit, spectrum_err_fit, line_name, start, bounds, xy_loc, runID, ncomp=2, plot=False):
	## function to fit line to continuum-subtracted spectrum, with start values, bounds, and varying number of gaussian components

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
			ax0.plot(wave_vel_fit-galv.value, gauss_sum(wave_vel_fit, *popt[3:6]), linestyle='-', color='tab:purple', label=fr'Comp 2 (A={amp2}e-18,$\sigma$={sig2})')


		ax0.plot(wave_vel_fit-galv.value, contsub_spec, color='tab:blue', linestyle='-', marker='.', label='Contsub spectrum')
		ax0.plot(wave_vel_fit-galv.value, gauss_sum(wave_vel_fit, *popt[0:3]), linestyle='--', color='tab:purple', label=fr'Comp 1 (A={amp1}e-18,$\sigma$={sig1})')

		#ax0.text(0, 1.1, start, transform=ax0.transAxes)

		#ax0.axvspan(-1700, -1000, color='tab:red', alpha=0.1)
		#ax0.axvspan(1000, 1700, color='tab:red', alpha=0.1)

		ax0.legend()

		ax0.grid()
		#ax0.text()

		residuals = contsub_spec - gauss_sum(wave_vel_fit, *popt) 

		ax1.plot(wave_vel_fit-galv.value, residuals, color='tab:red', marker='p', linestyle='-')
		ax1.axhline(0)

		ax0.set_ylabel('Flux (erg/s/cm^2/A)')
		ax1.set_ylabel('Residuals (erg/s/cm^2/A)')

		ax1.set_xlabel('Velocity (km/s)')

		ax0.set_title(f'{line_name} fit for bin {xy_loc}' )
		plt.subplots_adjust(hspace=0)

		savename = f'{line_name}_x{xy_loc[0]}y{xy_loc[1]}_{ncomp}ncomp.png'
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
	wave_spec = cube.spectral_axis.to(u.angstrom).value

	cube_shape = cube_data.shape

	if ap_shape == 'fov':
		trunc_cube = cube[1:cube_shape[0]-1, 1:cube_shape[1]]
		spectrum = trunc_cube.sum(axis=(1,2))

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
			print('center in pix', center_pix)

			center_pixcoord = PixCoord(center_pix[0], center_pix[1])

		else:
			center_pixcoord = PixCoord(pix_center[0], pix_center[1])

		if ap_shape == 'circle':
			ap = CirclePixelRegion(center=center_pixcoord, radius=ap_dimensions_pix[0])

		elif ap_shape == 'annulus':
			ap = CircleAnnulusPixelRegion(center=center_pixcoord, inner_radius=ap_dimensions_pix[0], outer_radius=ap_dimensions_pix[1])

		mask_cube = cube.subcube_from_regions([ap])

		spectrum = mask_cube.sum(axis=(1,2))

	rough_err_bounds = [20900, 21100]
	err_ind1 = np.where(wave_spec > rough_err_bounds[0])
	err_ind2 = np.where(wave_spec < rough_err_bounds[1])
	err_ind = np.intersect1d(err_ind1, err_ind2)
	err = np.nanstd(spectrum[err_ind])

	return spectrum, err, wave_spec


def measure_spectrum_err(contsub_spec, wave_arr):
	#take standard deviation of a line-free region of the spectrum
	#2.26 - 2.35
	spec_region = [22600, 23500]

	ind1 = np.where(wave_arr > spec_region[0])[0]
	ind2 = np.where(wave_arr < spec_region[1])[0]
	ind = np.intersect1d(ind1, ind2)

	line_free_region = contsub_spec[ind]
	spec_std = np.nanstd(line_free_region)

	sclip_keep_ind = np.where(line_free_region < 3*spec_std)[0]


	residuals_clip_std = np.nanstd(line_free_region[sclip_keep_ind])
	cont_serr = residuals_clip_std / np.sqrt(len(sclip_keep_ind))

	return cont_serr, residuals_clip_std


def fit_spectrum(spectrum, cont_spec, wave_arr, line_dict, savename):
	#given aperture spectrum and list of lines, fit each one, return dictionary with fit parameters for each

	contsub_spec = spectrum - cont_spec

	return_dict = {}

	for i, line_name in enumerate(line_dict.keys()):
		line_wave = line_dict[line_name]

		wave_to_vel = u.doppler_optical(line_wave*u.micron)
		wave_vel = (wave_arr*u.Angstrom).to(u.km/u.s, equivalencies=wave_to_vel) #- galv
		#spec_kms = contsub_spec.with_spectral_unit(u.km/u.s, velocity_convention='optical', rest_value=line_wave*u.micron)

		fit_ind1 = np.where(wave_vel > galv-2500* u.km/u.s)
		fit_ind2 = np.where(wave_vel < galv+2500* u.km/u.s)
		fit_ind = np.intersect1d(fit_ind1, fit_ind2)

		if line_name == 'Brgamma':

			'''fig = plt.figure()
			plt.plot(np.arange(len(fit_ind)), spectrum[fit_ind], marker='o')
			plt.grid()
			plt.xlim(72,90)
			plt.savefig('brgamma_test.png')'''

			mask_ind = [81,83] #exclude these from fit_ind
			fit_ind = np.concatenate((fit_ind[0:mask_ind[0]], fit_ind[mask_ind[1]:-1]))

		if line_name == 'He I':

			'''fig = plt.figure()
			plt.plot(np.arange(len(fit_ind)), spectrum[fit_ind], marker='o')
			plt.grid()
			plt.xlim(72,90)
			plt.savefig('hei_test.png')'''

			mask_ind = [81,84] #exclude these from fit_ind
			fit_ind = np.concatenate((fit_ind[0:mask_ind[0]], fit_ind[mask_ind[1]:-1]))


		fit_spec = contsub_spec[fit_ind]
		fit_vel = wave_vel[fit_ind].value

		#changed this to get the spectrum uncertainty from a line-free region
		cont_serr, cont_std =  measure_spectrum_err(contsub_spec, wave_arr)
		cont_std_err = np.repeat(cont_std, len(fit_spec))

		fit_ind3 = np.where(wave_vel > galv-1000* u.km/u.s)
		fit_ind4 = np.where(wave_vel < galv+1000* u.km/u.s)
		fit_ind_line = np.intersect1d(fit_ind3, fit_ind4)
		fit_spec_line = contsub_spec[fit_ind_line]

		peak_flux = np.nanmax(fit_spec_line)
		peak_ind = np.where(fit_spec == peak_flux)[0][0]
		peak_vel = fit_vel[peak_ind]

		if np.abs(peak_vel - galv.value) > 500:
			peak_vel = galv.value

		start_2comp = [peak_vel, peak_flux, 100, peak_vel, peak_flux*0.25, 200]
		bounds_2comp = ([galv.value - 500, peak_flux * 0.01, 20, galv.value - 500, 0, 50], [galv.value + 500, peak_flux*3, 300, galv.value + 500, peak_flux*10, 300])

		popt_2comp, pcov_2comp = fit_line(fit_spec, fit_vel, cont_std_err, line_name, start_2comp, bounds_2comp, (0,0), savename, ncomp=2, plot=True)

		if np.abs(peak_vel - galv.value) > 200:
			peak_vel = galv.value
		start_1comp = [peak_vel, peak_flux, 100]
		bounds_1comp = ([galv.value - 200, peak_flux * 0.01, 20], [galv.value + 200, peak_flux*3, 300])

		popt_1comp, pcov_1comp = fit_line(fit_spec, fit_vel, cont_std_err, line_name, start_1comp, bounds_1comp, (0,0), savename, ncomp=1, plot=True)

		residuals_2comp = fit_spec - gauss_sum(fit_vel, *popt_2comp)
		ssr_2comp = np.nansum(np.power(residuals_2comp,2))

		bic_2comp = bayesian_info_criterion_lsq(ssr_2comp, 6, fit_spec.shape[0])

		residuals_1comp = fit_spec - gauss_sum(fit_vel, *popt_1comp)
		ssr_1comp = np.nansum(np.power(residuals_1comp,2))

		bic_1comp = bayesian_info_criterion_lsq(ssr_1comp, 3, fit_spec.shape[0])

		delta_bic = bic_1comp - bic_2comp

		sum_flux =  np.nansum(fit_spec_line)

		if delta_bic > 10:
			return_dict[line_name] = [popt_2comp, pcov_2comp, delta_bic, sum_flux, cont_std]
		else:
			return_dict[line_name] = [popt_1comp, pcov_1comp, delta_bic, sum_flux, cont_std]

	return return_dict


def ppxf_fit_stellar(spectrum, err, moments, start, adegree, mdegree, wave_lam, plot_name=None, correct_cont=True):
	#cube - unbinned data cube
	#err
	#start
	#moments - ppxf moments for fitting
	#adegree - degree of additive poly
	#mdegree - degree of multiplicative poly
	#wave_lam - wavelength array
	#plot_name - name for output plots

	save_dict = {'star_vel':[], 'star_vel_error':[], 'star_sig':[], 'star_sig_error':[],
	'SN_mad_STD':[], 'chisq/dof':[], 'contsub_spec':[]}

	# NIFS spectral resolution, in Angstroms, R=5280
	FWHM_gal = 4.15
	
	wave_lam_rest = wave_lam/(1+z)

	#rebinning the wavelength to get the velscale
	spec_rebin, log_wave_rest, velscale = util.log_rebin(wave_lam_rest[[0,-1]], spectrum)

	wave_rest_rebin = np.exp(log_wave_rest)

	#preparing stellar templates
	stars_templates, wave_temp_rest = load_xsl_temps(wave_rest_rebin)
	
	wave_rebin = wave_rest_rebin * (1+z)
	log_wave_rebin = np.log(wave_rebin)
	err_spec = np.full(spectrum.shape, err)

	#create cont sub empty cube 
	contsub_spec = np.full(spectrum.shape, fill_value=np.nan)

	#for saving optimal templates for each bin
	#optimal_templates_save = np.empty((stars_templates.shape[0], n_bins))

	lam_temp = wave_temp_rest
	lamRange_temp = [lam_temp[0], lam_temp[-1]]

	templates = stars_templates
	gas_component = None
	gas_names = None
	component = 0

	#velocity difference between templates and data
	dv = (const.c.to(u.km/u.s)).value*(np.log(lam_temp[0]/wave_rebin[0])) # eq.(8) of Cappellari (2017)

	print('\n============================================================================')
	#also use previous optimal stellar template if fitting gas
	#choose starting values - just use galaxy velocity if no previous fit included
	
	bounds = None
	templates /= np.median(templates)

	good_pix_total = custom_util.determine_goodpixels(log_wave_rebin, lamRange_temp, z) #uses z to get redshifted line wavelengths, so logLam should be observer frame
	#good_pix_total = np.intersect1d(goodpix_util, good_pixels)

	#find nans, set them to median values and then remove from good pixels
	nan_ind = np.where(np.isnan(spec_rebin))

	spec_rebin[nan_ind] = np.nanmedian(spec_rebin)
	#noise_lin[nan_ind] = np.nanmedian(noise_lin)

	goodpix_ind = np.where(good_pix_total == nan_ind)
	good_pix_total = np.delete(good_pix_total, goodpix_ind)

	#CALLING PPXF HERE!
	t = clock()

	#old lam=np.exp(log_wave_trunc)/(1+z)

	pp = ppxf(templates, spec_rebin, err_spec, velscale, start,
				plot=False, moments=moments, degree=adegree, mdegree=mdegree, vsyst=dv,
				clean=False, lam=wave_rest_rebin, #regul= 1/0.1 , reg_dim=reg_dim,
				component=component, gas_component=gas_component, bounds=bounds,
				goodpixels = good_pix_total, global_search=False)

	bestfit = pp.bestﬁt
	contsub_spec = spec_rebin - bestfit

	param0 = pp.sol
	error = pp.error

	mad_std_residuals = mad_std(contsub_spec, ignore_nan=True)    
	med_spec = np.nanmedian(spec_rebin) #this will stand in for signal probably will be ~1
	SN_wMadStandardDev = med_spec/mad_std_residuals 
	save_dict['SN_mad_STD'].append(SN_wMadStandardDev)
	print('S/N w/ mad_std: '+str(SN_wMadStandardDev))

	vel_comp0 = param0[0]
	vel_error_comp0 = error[0]*np.sqrt(pp.chi2)
	sig_comp0 = param0[1]
	sig_error_comp0 = error[1]*np.sqrt(pp.chi2)

	optimal_template = templates @ pp.weights
	#optimal_templates_save = optimal_template

	save_dict['star_vel'].append(vel_comp0) 
	save_dict['star_vel_error'].append(vel_error_comp0)
	save_dict['star_sig'].append(sig_comp0)
	save_dict['star_sig_error'].append(sig_error_comp0)
	save_dict['chisq/dof'].append(pp.chi2)
	save_dict['contsub_spec'].append(bestfit)

	fit_chisq = (pp.chi2 - 1)*spec_rebin.size
	print('============================================================================')
	print('Desired Delta Chi^2: %.4g' % np.sqrt(2*spec_rebin.size))
	print('Current Delta Chi^2: %.4g' % (fit_chisq))
	print('Elapsed time in PPXF: %.2f s' % (clock() - t))
	print('============================================================================')

	if plot_name is not None:
		output_ppxf_fit_plot(plot_name, pp, good_pix_total, log_wave_rebin, vel_comp0, z, 0, 
							mad_std_residuals, SN_wMadStandardDev, fit_gas=False)


	if correct_cont == True:
		##account for weird upturn past 2.4 um
		end_ind = np.where(wave_rebin > 24070)
		H2Q_ind = np.where(wave_rebin > 24190)

		line_free_ind = np.setdiff1d(end_ind, H2Q_ind)
		new_cont = np.nanmedian(spec_rebin[line_free_ind])
		cont_spec = bestfit
		cont_spec[end_ind] = np.repeat(new_cont, len(end_ind[0]))

		start_ind = np.where(wave_rebin < 20250)
		beg_ind = np.where(wave_rebin < 20430)

		line_free_ind_start = np.setdiff1d(beg_ind, start_ind)
		new_cont_start = np.nanmedian(spec_rebin[line_free_ind_start])

		cont_spec[start_ind] = np.repeat(new_cont_start, len(start_ind[0]))

	else:
		cont_spec = bestfit



	return save_dict, optimal_template, spec_rebin, cont_spec, wave_rebin #bestfit is continuum spectrum



def run_stellar_fit(runID, spectrum, err, wave_lam):
	#run stellar fit, optionally also fit gas or mask it out
	#runID - name identifier for the run (e.g. Nov25_1comp)
	#cube_path - path to data cube
	#vorbin_path - path to voronoi bin text file
	#prev_cmap_path - if supplied, uses initial fitting conditions

	#ppxf parameters
	moments = 2
	adegree = 10     		# Additive polynomial used to correct the template continuum shape during the fit.
							# Degree = 10 is the suggested order for stellar populations.
	mdegree = 10			# Multiplicative polynomial used to correct the template continuum shape during the fit.
							# Mdegree = -1 tells fit to not include the multiplicative polynomial.

	start = [galv.value, 25]

	csv_save_name = f'ppxf_output/plots/ppxf_aperture/stellarfit_{runID}.csv'

	run_dict, opt_temps_save, spec_rebin, cont_spec, wave_rebin = ppxf_fit_stellar(spectrum, err, moments, start, adegree, mdegree, wave_lam, plot_name=runID)

	run_dict_df = pd.DataFrame.from_dict(run_dict)
	run_dict_df.to_csv(csv_save_name, index=False, header=True)
	print(f'Saved ppxf stellar fit csv to {csv_save_name}')

	#opt_temps_save_path = csv_save_name[:-4]+'_templates.npy'
	#np.save(opt_temps_save_path, opt_temps_save)
	#print(f'Saved optimal templates to {opt_temps_save_path}')

	return run_dict, opt_temps_save, cont_spec, spec_rebin, wave_rebin


def spectrum_figure(spectrum, cont_spec, wave_arr, fit_dict, savename, ap_name='center'):
	fig = plt.figure(figsize=(12,10))
	gs = GridSpec(5,5, wspace=0.25, hspace=0.75)

	gs2 = GridSpecFromSubplotSpec(1,4, subplot_spec=gs[4,:], wspace=0.25)

	ax0 = fig.add_subplot(gs[1:4, 0:5])

	wave_arr = (wave_arr * u.Angstrom).to(u.micron)

	ax0.plot(wave_arr, spectrum*1e16, color='tab:blue', label='NIFS spectrum')
	ax0.plot(wave_arr, cont_spec*1e16, color='tab:orange', label='Stellar continuum')

	ax0.set_xlim(2.005, 2.43)
	ax0.set_xlabel(r'Observed Wavelength ($\mu$m)', size=12)
	ax0.tick_params(axis='both', labelsize=12)

	ax0.set_ylabel(r'Flux (10$^{-16}$ erg/s/cm$^2$/$\AA$)', size=14)

	handles = [Line2D([0], [0], label='NIFS spectrum', color='tab:blue'),
				Line2D([0], [0], label='Stellar continuum', color='tab:orange'),
				Line2D([0], [0], label='Gaussian fit', color='tab:purple'),
				Line2D([0], [0], label='Full model', color='k')]
	ax0.legend(bbox_to_anchor=(0.9,0.95), fontsize=12)

	ax1 = fig.add_subplot(gs[0,0])
	ax2 = fig.add_subplot(gs[0,1])
	ax3 = fig.add_subplot(gs[0,2])
	ax4 = fig.add_subplot(gs[0,3])
	ax5 = fig.add_subplot(gs[0,4])

	ax6 = fig.add_subplot(gs2[0,0])
	ax7 = fig.add_subplot(gs2[0,1])
	ax8 = fig.add_subplot(gs2[0,2])
	ax9 = fig.add_subplot(gs2[0,3])

	line_ax = [ax1, ax6, ax2, ax7, ax3, ax8, ax4, ax5]
	line_name_list_full = [r'H$_2$(1-0)S(2)', r'H$_2$(2-1)S(3)', r'H$_2$(1-0)S(1)', r'H$_2$(2-1)S(2)', r'Br$\gamma$', r'H$_2$(1-0)S(0)', r'H$_2$(2-1)S(1)', r'H$_2$(1-0)Q(1)']
	color_list = ['tab:blue', 'tab:purple', 'tab:red', 'tab:orange', 'tab:green', 'tab:pink', 'tab:olive', 'tab:cyan']

	for i, line_name in enumerate(line_dict_noHe):

		ax = line_ax[i]
		line_wave = line_dict[line_name]
		fit_returns = fit_dict[line_name]
		fit_params = fit_returns[0]
		delta_bic = fit_returns[2]

		ax.set_title(line_name_list_full[i])


		wave_to_vel = u.doppler_optical(line_wave*u.micron)
		wave_vel = wave_arr.to(u.km/u.s, equivalencies=wave_to_vel)
		#spec_kms = spectrum.with_spectral_unit(u.km/u.s, velocity_convention='optical', rest_value=line_wave*u.micron)

		fit_ind1 = np.where(wave_vel > galv-2500* u.km/u.s)
		fit_ind2 = np.where(wave_vel < galv+2500* u.km/u.s)
		fit_ind = np.intersect1d(fit_ind1, fit_ind2)
		fit_spec = spectrum[fit_ind]
		fit_vel = wave_vel[fit_ind].value
		fit_cont = cont_spec[fit_ind]

		if line_name == 'Brgamma':
			mask_ind = [81,83] #exclude these from fit
			ax.axvspan(fit_vel[mask_ind[0]]-galv.value, fit_vel[mask_ind[1]]-galv.value, color='k', alpha=0.2)

		ax.plot(fit_vel - galv.value, fit_spec*1e16, color='tab:blue')

		ax.plot(fit_vel-galv.value, (gauss_sum(fit_vel, *fit_params[0:3]) + fit_cont)*1e16, linestyle='--', color='tab:purple')

		if line_name != len(fit_params) > 3:
			ax.plot(fit_vel-galv.value, (gauss_sum(fit_vel, *fit_params[3:6]) + fit_cont)*1e16, linestyle='-', color='tab:purple')

		ax.plot(fit_vel-galv.value, fit_cont*1e16, linestyle='-', color='tab:orange')
		ax.plot(fit_vel-galv.value, (fit_cont + gauss_sum(fit_vel, *fit_params))*1e16, linestyle='-', color='k')

		bic_print = int(np.round(delta_bic, 0))
		ax.text(0.02, 0.87, rf'$\Delta$BIC = {bic_print}', fontsize=10, transform=ax.transAxes)

		ax.set_xlabel('Velocity (km/s)', fontsize=12)

		ax.tick_params(axis='both', labelsize=12)

		if i != 8:
			ax.set_xlim(-900,900)
			ax.set_xticks([-500, 0, 500])
			ax.axvspan(-900,900, color=color_list[i], alpha=0.2)

		else:
			ax.set_xlim(-900, 500)
			ax.set_xticks([-500, 0, 500])
			ax.axvspan(-900,500, color=color_list[i], alpha=0.2)

		obs_wave = line_wave * (1+z)
		ax0.axvspan(obs_wave-5e-3, obs_wave+5e-3, color=color_list[i], alpha=0.2)

		ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))

	#plt.savefig(f'plots/NIFS_{savename}_spec_noarr.png', dpi=300, bbox_inches='tight')

	xy_1 = [0.102, 0.9]
	xy_ax1 = [0.5, -0.5]
	arrow1 = ConnectionPatch(xyA=xy_1, coordsA=ax0.transAxes, xyB=xy_ax1, coordsB=ax1.transAxes, arrowstyle='-|>', fc=color_list[0])
	fig.add_artist(arrow1)

	#xy_2 = [0.2, 0.97]
	xy_2 = [0.315, 0.97]
	xy_ax2 = [0.5, -0.5]
	arrow2 = ConnectionPatch(xyA=xy_2, coordsA=ax0.transAxes, xyB=xy_ax2, coordsB=ax2.transAxes, arrowstyle='-|>', fc=color_list[2])
	fig.add_artist(arrow2)

	#xy_3 = [0.415, 0.9]
	xy_3 = [0.415, 0.9]
	xy_ax3 = [0.5, -0.5]
	arrow3 = ConnectionPatch(xyA=xy_3, coordsA=ax0.transAxes, xyB=xy_ax3, coordsB=ax3.transAxes, arrowstyle='-|>', fc=color_list[4])
	fig.add_artist(arrow3)

	#xy_4 = [0.61, 0.9]
	xy_4 = [0.61, 0.9]
	xy_ax4 = [0.5, -0.5]
	arrow4 = ConnectionPatch(xyA=xy_4, coordsA=ax0.transAxes, xyB=xy_ax4, coordsB=ax4.transAxes, arrowstyle='-|>', fc=color_list[6])
	fig.add_artist(arrow4)

	#xy_5 = [0.2, 0.1]
	xy_5 = [0.98, 0.9]
	xy_ax5 = [0.5, -0.5]
	arrow5 = ConnectionPatch(xyA=xy_5, coordsA=ax0.transAxes, xyB=xy_ax5, coordsB=ax5.transAxes, arrowstyle='-|>', fc=color_list[7])
	fig.add_artist(arrow5)

	#xy_6 = [0.39, 0.1]
	xy_6 = [0.2, 0.1]
	xy_ax6 = [0.5, 1.3]
	arrow6 = ConnectionPatch(xyA=xy_6, coordsA=ax0.transAxes, xyB=xy_ax6, coordsB=ax6.transAxes, arrowstyle='-|>', fc=color_list[1])
	fig.add_artist(arrow6)

	#xy_7 = [0.55, 0.1]
	xy_7 = [0.39, 0.1]
	xy_ax7 = [0.5, 1.3]
	arrow7= ConnectionPatch(xyA=xy_7, coordsA=ax0.transAxes, xyB=xy_ax7, coordsB=ax7.transAxes, arrowstyle='-|>', fc=color_list[3])
	fig.add_artist(arrow7)

	#xy_8 = [0.98, 0.095]
	xy_8 = [0.55, 0.1]
	xy_ax8 = [0.65, 1.3]
	arrow8 = ConnectionPatch(xyA=xy_8, coordsA=ax0.transAxes, xyB=xy_ax8, coordsB=ax8.transAxes, arrowstyle='-|>', fc=color_list[5])
	fig.add_artist(arrow8)

	#xy_9 = [0.61, 0.1]
	#xy_ax9 = [0.5, 1.3]
	#arrow9 = ConnectionPatch(xyA=xy_9, coordsA=ax0.transAxes, xyB=xy_ax9, coordsB=ax9.transAxes, arrowstyle='-|>', fc=color_list[7])
	#fig.add_artist(arrow9)


	maps_fl = fits.open('/Users/jotter/highres_PSBs/ngc1266_NIFS/fit_output/c3_run4_gaussfit_maps.fits')
	maps = maps_fl[0].data
	maps_fl.close()

	H2_flux = maps[11,:,:]
	H2_mask = maps[13,:,:]
	H2_bool = np.where(H2_mask == 1, True, False)
	H2_flux_det = H2_flux.copy()
	H2_flux_det[H2_bool == False] = np.nan

	ax9.imshow(np.log10(H2_flux_det), cmap='gray', vmin=-17.1, vmax=-16)

	if ap_name == 'center':
		ap_center = (42,34)
	if ap_name == 'east':
		ap_center = (22,42)
	if ap_name == 'west':
		ap_center = (64,20)

	ap_radius = 7.78
	ap_patch = Circle(ap_center, radius=ap_radius, fill=None, edgecolor='red', linewidth=1)

	ax9.add_patch(ap_patch)
	ax9.set_xlim(5,75)
	ax9.set_ylim(5,63)

	ax9.tick_params(axis='both', labelleft=False, labelbottom=False, left=False, bottom=False)
	ax9.set_xlabel(r'Spectrum aperture', fontsize=12)
	ax9.text(10, 10, '100 pc', fontsize=10, color='red')

	rect = Rectangle((15,18), ap_radius*2, 1.5, color='red')
	ax9.add_patch(rect)

	plt.savefig(f'plots/NIFS_{savename}_spec.pdf', dpi=300, bbox_inches='tight')

def spectrum_figure_He(spectrum, cont_spec, wave_arr, fit_dict, savename, ap_name='center'):
	fig = plt.figure(figsize=(12,10))
	gs = GridSpec(5,5, wspace=0.25, hspace=0.7)

	ax0 = fig.add_subplot(gs[1:4, 0:5])

	wave_arr = (wave_arr * u.Angstrom).to(u.micron)

	ax0.plot(wave_arr, spectrum*1e16, color='tab:blue', label='NIFS spectrum')
	ax0.plot(wave_arr, cont_spec*1e16, color='tab:orange', label='Stellar continuum')

	ax0.set_xlim(2.005, 2.43)
	ax0.set_xlabel(r'Observed Wavelength ($\mu$m)', size=12)
	ax0.tick_params(axis='both', labelsize=12)

	ax0.set_ylabel(r'Flux (10$^{-16}$ erg/s/cm$^2$/$\AA$)', size=14)

	ax1 = fig.add_subplot(gs[0,0])
	ax2 = fig.add_subplot(gs[0,1])
	ax3 = fig.add_subplot(gs[0,2])
	ax4 = fig.add_subplot(gs[0,3])
	ax5 = fig.add_subplot(gs[0,4])

	ax6 = fig.add_subplot(gs[4,0])
	ax7 = fig.add_subplot(gs[4,1])
	ax8 = fig.add_subplot(gs[4,2])
	ax9 = fig.add_subplot(gs[4,3])
	ax10 = fig.add_subplot(gs[4,4])

	line_ax = [ax1, ax6, ax2, ax7, ax3, ax8, ax4, ax9, ax5, ax10]
	line_name_list_full = [r'H$_2$(1-0)S(2)', 'He I', r'H$_2$(2-1)S(3)', r'H$_2$(1-0)S(1)', r'H$_2$(2-1)S(2)', r'Br$\gamma$', r'H$_2$(1-0)S(0)', r'H$_2$(2-1)S(1)', r'H$_2$(1-0)Q(1)']
	color_list = ['tab:blue', 'tab:purple', 'tab:red', 'tab:orange', 'tab:green', 'tab:pink', 'tab:olive', 'tab:cyan', 'tab:brown']

	for i, line_name in enumerate(line_dict):
		ax = line_ax[i]
		line_wave = line_dict[line_name]
		fit_returns = fit_dict[line_name]
		fit_params = fit_returns[0]
		delta_bic = fit_returns[2]

		ax.set_title(line_name_list_full[i])


		wave_to_vel = u.doppler_optical(line_wave*u.micron)
		wave_vel = wave_arr.to(u.km/u.s, equivalencies=wave_to_vel)
		#spec_kms = spectrum.with_spectral_unit(u.km/u.s, velocity_convention='optical', rest_value=line_wave*u.micron)

		fit_ind1 = np.where(wave_vel > galv-2500* u.km/u.s)
		fit_ind2 = np.where(wave_vel < galv+2500* u.km/u.s)
		fit_ind = np.intersect1d(fit_ind1, fit_ind2)
		fit_spec = spectrum[fit_ind]
		fit_vel = wave_vel[fit_ind].value
		fit_cont = cont_spec[fit_ind]

		if line_name == 'Brgamma':
			mask_ind = [81,83] #exclude these from fit
			ax.axvspan(fit_vel[mask_ind[0]]-galv.value, fit_vel[mask_ind[1]]-galv.value, color='k', alpha=0.2)

		if line_name == 'He I':
			mask_ind = [81,84] 
			ax.axvspan(fit_vel[mask_ind[0]]-galv.value, fit_vel[mask_ind[1]]-galv.value, color='k', alpha=0.2)

		ax.plot(fit_vel - galv.value, fit_spec*1e16, color='tab:blue')

		ax.plot(fit_vel-galv.value, (gauss_sum(fit_vel, *fit_params[0:3]) + fit_cont)*1e16, linestyle='--', color='tab:purple')

		if line_name != len(fit_params) > 3:
			ax.plot(fit_vel-galv.value, (gauss_sum(fit_vel, *fit_params[3:6]) + fit_cont)*1e16, linestyle='-', color='tab:purple')

		ax.plot(fit_vel-galv.value, fit_cont*1e16, linestyle='-', color='tab:orange')
		ax.plot(fit_vel-galv.value, (fit_cont + gauss_sum(fit_vel, *fit_params))*1e16, linestyle='-', color='k')

		bic_print = int(np.round(delta_bic, 0))
		ax.text(0.02, 0.87, rf'$\Delta$BIC={bic_print}', fontsize=10, transform=ax.transAxes)

		ax.set_xlabel('Velocity (km/s)', fontsize=12)

		ax.tick_params(axis='both', labelsize=12)

		if i != 8:
			ax.set_xlim(-900,900)
			ax.set_xticks([-500, 0, 500])
			ax.axvspan(-900,900, color=color_list[i], alpha=0.2)

		else:
			ax.set_xlim(-900, 500)
			ax.set_xticks([-500, 0, 500])
			ax.axvspan(-900,500, color=color_list[i], alpha=0.2)

		obs_wave = line_wave * (1+z)
		ax0.axvspan(obs_wave-5e-3, obs_wave+5e-3, color=color_list[i], alpha=0.2)

	#plt.savefig(f'plots/NIFS_{savename}_spec_noarr.png', dpi=300, bbox_inches='tight')

	xy_1 = [0.102, 0.9]
	xy_ax1 = [0.5, -0.5]
	arrow1 = ConnectionPatch(xyA=xy_1, coordsA=ax0.transAxes, xyB=xy_ax1, coordsB=ax1.transAxes, arrowstyle='-|>', fc=color_list[0])
	fig.add_artist(arrow1)

	xy_2 = [0.2, 0.97]
	#xy_2 = [0.315, 0.97]
	xy_ax2 = [0.5, -0.5]
	arrow2 = ConnectionPatch(xyA=xy_2, coordsA=ax0.transAxes, xyB=xy_ax2, coordsB=ax2.transAxes, arrowstyle='-|>', fc=color_list[2])
	fig.add_artist(arrow2)

	#xy_3 = [0.415, 0.9]
	xy_3 = [0.39, 0.9]
	xy_ax3 = [0.5, -0.5]
	arrow3 = ConnectionPatch(xyA=xy_3, coordsA=ax0.transAxes, xyB=xy_ax3, coordsB=ax3.transAxes, arrowstyle='-|>', fc=color_list[4])
	fig.add_artist(arrow3)

	#xy_4 = [0.61, 0.9]
	xy_4 = [0.55, 0.9]
	xy_ax4 = [0.5, -0.5]
	arrow4 = ConnectionPatch(xyA=xy_4, coordsA=ax0.transAxes, xyB=xy_ax4, coordsB=ax4.transAxes, arrowstyle='-|>', fc=color_list[6])
	fig.add_artist(arrow4)

	#xy_5 = [0.2, 0.1]
	xy_5 = [0.98, 0.9]
	xy_ax5 = [0.5, -0.5]
	arrow5 = ConnectionPatch(xyA=xy_5, coordsA=ax0.transAxes, xyB=xy_ax5, coordsB=ax5.transAxes, arrowstyle='-|>', fc=color_list[8])
	fig.add_artist(arrow5)

	#xy_6 = [0.39, 0.1]
	xy_6 = [0.16, 0.1]
	xy_ax6 = [0.5, 1.3]
	arrow6 = ConnectionPatch(xyA=xy_6, coordsA=ax0.transAxes, xyB=xy_ax6, coordsB=ax6.transAxes, arrowstyle='-|>', fc=color_list[1])
	fig.add_artist(arrow6)

	#xy_7 = [0.55, 0.1]
	xy_7 = [0.315, 0.1]
	xy_ax7 = [0.5, 1.3]
	arrow7= ConnectionPatch(xyA=xy_7, coordsA=ax0.transAxes, xyB=xy_ax7, coordsB=ax7.transAxes, arrowstyle='-|>', fc=color_list[3])
	fig.add_artist(arrow7)

	#xy_8 = [0.98, 0.095]
	xy_8 = [0.415, 0.1]
	xy_ax8 = [0.2, 1.1]
	arrow8 = ConnectionPatch(xyA=xy_8, coordsA=ax0.transAxes, xyB=xy_ax8, coordsB=ax8.transAxes, arrowstyle='-|>', fc=color_list[5])
	fig.add_artist(arrow8)

	xy_9 = [0.61, 0.1]
	xy_ax9 = [0.5, 1.3]
	arrow9 = ConnectionPatch(xyA=xy_9, coordsA=ax0.transAxes, xyB=xy_ax9, coordsB=ax9.transAxes, arrowstyle='-|>', fc=color_list[7])
	fig.add_artist(arrow9)


	maps_fl = fits.open('/Users/jotter/highres_PSBs/ngc1266_NIFS/fit_output/c3_run4_gaussfit_maps.fits')
	maps = maps_fl[0].data
	maps_fl.close()

	H2_flux = maps[11,:,:]
	H2_mask = maps[13,:,:]
	H2_bool = np.where(H2_mask == 1, True, False)
	H2_flux_det = H2_flux.copy()
	H2_flux_det[H2_bool == False] = np.nan

	ax10.imshow(np.log10(H2_flux_det), cmap='gray', vmin=-17.1, vmax=-16)

	if ap_name == 'center':
		ap_center = (42,34)
	if ap_name == 'east':
		ap_center = (22,42)
	if ap_name == 'west':
		ap_center = (62,22)
	ap_radius = 7.78
	ap_patch = Circle(ap_center, radius=ap_radius, fill=None, edgecolor='red', linewidth=1)

	ax10.add_patch(ap_patch)
	ax10.set_xlim(5,75)
	ax10.set_ylim(5,63)

	ax10.tick_params(axis='both', labelleft=False, labelbottom=False, left=False, bottom=False)
	ax10.set_xlabel(r'Spectrum aperture', fontsize=12)
	ax10.text(10, 10, '100 pc', fontsize=10, color='red')

	rect = Rectangle((15,18), ap_radius*2, 1.5, color='red')
	ax10.add_patch(rect)

	line = Line2D([0], [0], label='Gaussian fit', color='tab:purple')
	ax.legend()

	plt.savefig(f'plots/NIFS_{savename}_He_spec.pdf', dpi=300, bbox_inches='tight')


def save_fluxes(fit_dict, savename):

	col_names = ['line name', 'total_flux', 'comp1_flux', 'comp2_flux', 'sum_flux', 'total_flux_err',
							'comp1_flux_err', 'comp2_flux_err', 'comp1_vel', 'comp1_sigma', 'comp2_vel', 'comp2_sig', 'ulim_flag']
	save_tab = Table(names=col_names, dtype=('S', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'bool'))

	for i, line_name in enumerate(fit_dict.keys()):

		save_name = line_names_save[i]
		line_wave = line_dict[line_name]

		fit_returns = fit_dict[line_name]
		fit_params = fit_returns[0]
		fit_cov = fit_returns[1]
		delta_bic = fit_returns[2]
		sum_flux = fit_returns[3]
		spec_std = fit_returns[4]

		vel1 = fit_params[0]
		sig1 = fit_params[2]

		if len(fit_params) < 5:
			ncomp = 1
			vel2 = np.nan
			sig2 = np.nan
		else:
			ncomp = 2
			vel2 = fit_params[3]
			sig2 = fit_params[5]

		obs_wave = line_wave * (1+z) * u.micron
		total_flux, total_flux_err = compute_flux(fit_params, fit_cov, obs_wave, ncomp=ncomp)

		if total_flux < total_flux_err * 3 or line_name == 'He I':
			ulim_flag = True
			if line_name == 'He I' or line_name == 'Brgamma':
				width = 130
			else:
				width = fit_dict['H2(1-0)S(1)'][0][2]

			ulim_flux = np.sqrt(2 * np.pi) * ((width * 3*spec_std))
			total_flux = ulim_flux
			total_flux_err = np.nan
			comp1_flux = np.nan
			comp1_flux_err = np.nan
			comp2_flux = np.nan
			comp2_flux_err = np.nan

		else:
			ulim_flag = False

			comp1_flux, comp1_flux_err = compute_flux(fit_params[0:3], fit_cov[0:3], obs_wave, ncomp=1)
			if ncomp == 2:
				comp2_flux, comp2_flux_err = compute_flux(fit_params[3:6], fit_cov[3:6], obs_wave, ncomp=1)
			else: 
				comp2_flux, comp2_flux_err = np.nan, np.nan

		save_tab.add_row([save_name, total_flux, comp1_flux, comp2_flux, sum_flux, total_flux_err, comp1_flux_err, comp2_flux_err, vel1, sig1, vel2, sig2, ulim_flag])



	#save_tab = Table(data=save_tab_data, names=col_names, dtype=('S', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'bool'))

	save_tab.write(f'/Users/jotter/highres_PSBs/ngc1266_NIFS/fit_output/{savename}.csv', format='csv', overwrite=True)


runID = 'Center_50pc'
spectrum, err, wave_lam = extract_spectrum('circle', [50]*u.pc, None)
fit_results, optimal_template, cont_spec, spec_rebin, wave_rebin = run_stellar_fit(runID, spectrum, err, wave_lam)
fit_dict = fit_spectrum(spec_rebin, cont_spec, wave_rebin, line_dict, 'center_50pc_ppxf')
#spectrum_figure_He(spec_rebin, cont_spec, wave_rebin, fit_dict, 'center_50pc_ppxf')
#spectrum_figure(spec_rebin, cont_spec, wave_rebin, fit_dict, 'center_50pc_ppxf')
save_fluxes(fit_dict, 'center_50pc_ppxf')

runID = 'East_50pc'
pixcent=[22,42]
spectrum, err, wave_lam  = extract_spectrum('circle', [50]*u.pc, pixcent)
fit_results, optimal_template, cont_spec, spec_rebin, wave_rebin = run_stellar_fit(runID, spectrum, err, wave_lam)
fit_dict = fit_spectrum(spec_rebin, cont_spec, wave_rebin, line_dict, 'east_50pc_ppxf')
spectrum_figure(spec_rebin, cont_spec, wave_rebin, fit_dict, 'east_50pc_ppxf', ap_name='east')
save_fluxes(fit_dict, 'east_50pc_ppxf')

runID = 'West_50pc'
pixcent=[64,20]
spectrum, err, wave_lam  = extract_spectrum('circle', [50]*u.pc, pixcent)
fit_results, optimal_template, cont_spec, spec_rebin, wave_rebin = run_stellar_fit(runID, spectrum, err, wave_lam)
fit_dict = fit_spectrum(spec_rebin, cont_spec, wave_rebin, line_dict, 'West2_50pc_ppxf')
spectrum_figure(spec_rebin, cont_spec, wave_rebin, fit_dict, 'West2_50pc_ppxf', ap_name='west')
save_fluxes(fit_dict, 'west2_50pc_ppxf')

runID = 'fov'
spectrum, err, wave_lam = extract_spectrum('fov', [50]*u.pc, None)
fit_results, optimal_template, cont_spec, spec_rebin, wave_rebin = run_stellar_fit(runID, spectrum, err, wave_lam)
fit_dict = fit_spectrum(spec_rebin, cont_spec, wave_rebin, line_dict, 'fov_ppxf')
spectrum_figure(spec_rebin, cont_spec, wave_rebin, fit_dict, 'fov_ppxf')
save_fluxes(fit_dict, 'fov_ppxf')


