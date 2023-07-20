from astropy.io import fits
from scipy.optimize import curve_fit
from scipy import odr
from astropy.table import Table
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from regions import CirclePixelRegion, CircleSkyRegion, PixCoord
from gaussfit_cube import fit_continuum, fit_line, compute_flux
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


def extract_spectrum(ap_shape, ap_dimensions):
	#ap_shape - either "circle" or "annulus"
	#ap_dimensions - [radius] for circle, [inner radius, outer radius] for annulus in kpc


	#convert dimensions from kpc to arcsec
	z = 0.007214
	D_L = cosmo.luminosity_distance(z).to(u.Mpc)
	as_per_kpc = cosmo.arcsec_per_kpc_comoving(z)

	ap_dimensions_as = (ap_dimensions * as_per_kpc).decompose()

	cube_path = '/Users/jotter/highres_PSBs/ngc1266_data/NIFS_data/NGC1266_NIFS_final_trim_wcs.fits'

	cube_fl = fits.open(cube_path)	
	cube_data = cube_fl[1].data
	nifs_wcs = WCS(cube_fl[1].header)
	cube_fl.close()

	cube = SpectralCube(data=cube_data, wcs=nifs_wcs)

	cube_shape = cube_data.shape

	gal_cen = SkyCoord(ra='3:16:00.7311', dec='-2:25:38.844', unit=(u.hourangle, u.degree),
	                  frame='icrs') #from HST H band image, by eye

	center_pix = nifs_wcs.celestial.all_world2pix(gal_cen.ra, gal_cen.dec, 0)

	center_pixcoord = PixCoord(center_pix[0], center_pix[1])

	ap_dimensions_pix = (ap_dimensions_as / (0.043*u.arcsecond)).decompose().value

	if ap_shape == 'circle':
		ap = CirclePixelRegion(center=center_pixcoord, radius=ap_dimensions_pix[0])

	elif ap_shape == 'annulus':
		ap = CircleAnnulusPixelRegion(center=center_pixcoord, inner_radius=ap_dimensions_pix[0], outer_radius=ap_dimensions_pix[1])

	mask_cube = cube.subcube_from_regions([ap])

	spectrum = mask_cube.sum(axis=(1,2))

	return spectrum


def fit_spectrum(spectrum, line_dict):
	#given aperture spectrum and list of lines, fit each one, return dictionary with fit parameters for each

	return_dict = {}

	runID = 'circ_100pc'

	for i, line_name in enumerate(line_dict.keys()):
		line_wave = line_dict[line_name]

		spec_kms = spectrum.with_spectral_unit(u.km/u.s, velocity_convention='optical', rest_value=line_wave*u.micron)

		fit_ind1 = np.where(spec_kms.spectral_axis > galv-2000* u.km/u.s)
		fit_ind2 = np.where(spec_kms.spectral_axis < galv+2000* u.km/u.s)
		fit_ind = np.intersect1d(fit_ind1, fit_ind2)
		fit_spec = spec_kms[fit_ind]
		fit_spec_vel = spec_kms.spectral_axis[fit_ind].value

		cont_fit, cont_std, cont_serr, mask_ind = fit_continuum(fit_spec, fit_spec_vel, line_name, degree=1, mask_ind=None)
		cont_std_err = np.repeat(cont_std, len(fit_spec))

		peak_flux = np.nanmax(fit_spec)
		peak_ind = np.where(fit_spec == peak_flux)[0][0]
		peak_vel = fit_spec_vel[peak_ind]

		if np.abs(peak_vel - galv.value) > 500:
			peak_vel = galv.value
		start = [peak_vel, peak_flux, 200, peak_vel, peak_flux*0.25, 400]

		bounds = ([galv.value - 500, peak_flux * 0.1, 20, galv.value - 500, 0, 50], [galv.value + 500, peak_flux*10, 600, galv.value + 500, peak_flux*10, 600])

		popt, pcov = fit_line(fit_spec, fit_spec_vel, cont_std_err, cont_fit, line_name, start, bounds, (0,0), runID, plot=True, mask_ind=mask_ind)

		fit_spec_wave = spectrum.spectral_axis[fit_ind].to(u.Angstrom)

		obs_wave = line_wave * (1+z) * u.micron
		total_flux, total_flux_err = compute_flux(popt, pcov, obs_wave)

		return_dict[line_name] = [popt, pcov, [total_flux, total_flux_err], cont_fit]

	return return_dict


def spectrum_figure(spectrum, fit_dict):
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
		cont_fit = fit_returns[3]

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

		if i == 5:
			ax.set_yticks([8,10,12])

		if i != 7:
			ax.set_xlim(-1500,1500)
			ax.set_xticks([-1000, 0, 1000])
			ax.axvspan(-1500,1500, color=color_list[i], alpha=0.2)

		else:
			ax.set_xlim(-1500, 500)
			ax.set_xticks([-1000, 0, 500])
			ax.axvspan(-1500,500, color=color_list[i], alpha=0.2)

		obs_wave = line_wave * (1+z)
		ax0.axvspan(obs_wave-5e-3, obs_wave+5e-3, color=color_list[i], alpha=0.2)

	#xy_1 = [2.048, 18]
	xy_1 = [2.048, 28]
	xy_ax1 = [0.5, -0.4]
	arrow1 = ConnectionPatch(xyA=xy_1, coordsA=ax0.transData, xyB=xy_ax1, coordsB=ax1.transAxes, arrowstyle='-|>', fc=color_list[0])
	fig.add_artist(arrow1)

	#xy_2 = [2.14, 30]
	xy_2 = [2.14, 28]
	xy_ax2 = [0.5, -0.4]
	arrow2 = ConnectionPatch(xyA=xy_2, coordsA=ax0.transData, xyB=xy_ax2, coordsB=ax2.transAxes, arrowstyle='-|>', fc=color_list[2])
	fig.add_artist(arrow2)

	#xy_3 = [2.182, 10]
	xy_3 = [2.182, 28]
	xy_ax3 = [0.5, -0.4]
	arrow3 = ConnectionPatch(xyA=xy_3, coordsA=ax0.transData, xyB=xy_ax3, coordsB=ax3.transAxes, arrowstyle='-|>', fc=color_list[4])
	fig.add_artist(arrow3)

	#xy_4 = [2.265, 11]
	xy_4 = [2.265, 28]
	xy_ax4 = [0.5, -0.4]
	arrow4 = ConnectionPatch(xyA=xy_4, coordsA=ax0.transData, xyB=xy_ax4, coordsB=ax4.transAxes, arrowstyle='-|>', fc=color_list[6])
	fig.add_artist(arrow4)

	#xy_5 = [2.087, 8]
	xy_5 = [2.087, 6]
	xy_ax5 = [0.5, 1.2]
	arrow5 = ConnectionPatch(xyA=xy_5, coordsA=ax0.transData, xyB=xy_ax5, coordsB=ax5.transAxes, arrowstyle='-|>', fc=color_list[1])
	fig.add_artist(arrow5)

	#xy_6 = [2.169, 8]
	xy_6 = [2.169, 6]
	xy_ax6 = [0.3, 1.2]
	arrow6 = ConnectionPatch(xyA=xy_6, coordsA=ax0.transData, xyB=xy_ax6, coordsB=ax6.transAxes, arrowstyle='-|>', fc=color_list[3])
	fig.add_artist(arrow6)

	#xy_7 = [2.24, 8]
	xy_7 = [2.24, 6]
	xy_ax7 = [0.7, 1.2]
	arrow7= ConnectionPatch(xyA=xy_7, coordsA=ax0.transData, xyB=xy_ax7, coordsB=ax7.transAxes, arrowstyle='-|>', fc=color_list[5])
	fig.add_artist(arrow7)

	xy_8 = [2.423, 6]
	xy_ax8 = [0.5, 1.2]
	arrow8 = ConnectionPatch(xyA=xy_8, coordsA=ax0.transData, xyB=xy_ax8, coordsB=ax8.transAxes, arrowstyle='-|>', fc=color_list[7])
	fig.add_artist(arrow8)

	plt.savefig('plots/NIFS_spec_fig.png', dpi=300, bbox_inches='tight')


def save_fluxes(fit_dict, savename):

	save_tab = Table()

	for i, line_name in enumerate(fit_dict.keys()):

		fit_returns = fit_dict[line_name]
		fit_params = fit_returns[0]
		fit_cov = fit_returns[1]
		flux, flux_err = fit_returns[2]
		cont_fit = fit_returns[3]

		print(flux, flux_err)

		save_tab[f'{line_name}_flux'] = [flux]
		save_tab[f'{line_name}_flux_err'] = [flux_err]

	save_tab.write(f'/Users/jotter/highres_PSBs/ngc1266_NIFS/fit_output/{savename}.csv', format='csv')


z = 0.007214
galv = np.log(z+1)*const.c.to(u.km/u.s)
line_dict = {'H2(1-0)S(2)':2.0332, 'H2(2-1)S(3)':2.0735, 'H2(1-0)S(1)':2.1213, 'H2(2-1)S(2)':2.1542, 'Brgamma':2.1654, 'H2(1-0)S(0)':2.2230, 'H2(2-1)S(1)':2.2470, 'H2(1-0)Q(1)':2.4066}

spec = extract_spectrum('circle', [100]*u.pc)
fit_dict = fit_spectrum(spec, line_dict)

spectrum_figure(spec, fit_dict)

save_fluxes(fit_dict, 'NIFS_100pc_fluxes')




