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
z = 0.007214
galv = np.log(z+1)*const.c.to(u.km/u.s)

### script to look at amount of telluric over-subtraction for Brgamma in different locations of the cube
## rather than fitting, should sum the channels where the over-subtraction occurs

def fit_continuum(spectrum, wave_vel, degree=1, mask_ind=None):
	## function to do continuum fit for a given line, with hand-selected continuum bands for each line

	line_vel = wave_vel - galv.value
	lower_band_kms = [-1000, -500]
	upper_band_kms = [1000,1700]

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

		cont_fit, residuals_clip_std, cont_serr, mask_ind = fit_continuum(spectrum, wave_vel, degree=1, mask_ind=full_ind)


	return cont_fit, (cont_params)

def compute_flux(popt, pcov, line_wave):
	#take fit parameter output and compute flux and error

	width1 = popt[2] * u.km/u.s
	amp1 = popt[1] #* u.erg/u.s/u.cm**2/u.AA
	wave_width1 = ((width1 / const.c) * line_wave).to(u.angstrom).value

	amp1_err = np.sqrt(pcov[1,1]) #wrong with unit conversion
	width1_err = np.sqrt(pcov[2,2]) * u.km/u.s
	wave_width1_err = ((width1_err / const.c) * line_wave).to(u.angstrom).value

	if len(popt) > 3:
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


def extract_spectrum(ap_shape, ap_dimensions, pix_center, which_cube='orig'):
	#ap_shape - either "circle" or "annulus"
	#ap_dimensions - [radius] for circle, [inner radius, outer radius] for annulus in length, angular, or dimensionless (pixel) units
	#pix_center - center of aperture in pixel coordinates, if None use galaxy center

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

	if which_cube == 'orig':
		cube_path = '/Users/jotter/highres_PSBs/ngc1266_data/NIFS_data/NGC1266_NIFS_final_trim_wcs2.fits'
	else:
		cube_path = '/Users/jotter/highres_PSBs/ngc1266_data/NIFS_data/NGC1266_NIFS_dec6_wcs.fits'

	cube_fl = fits.open(cube_path)	
	cube_data = cube_fl[1].data
	nifs_wcs = WCS(cube_fl[1].header)
	cube_fl.close()

	cube = SpectralCube(data=cube_data, wcs=nifs_wcs)

	cube_shape = cube_data.shape

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



def plot_specs(location_list, sub_inds, savename, which_cube='orig'):
	fig = plt.figure(figsize=(10,10))
	gs = GridSpec(4,4, wspace=0.1, hspace=0.1)


	for i, loc in enumerate(location_list):
		spec = extract_spectrum('circle', [50]*u.pc, loc, which_cube=which_cube)
		spec_kms = spec.with_spectral_unit(u.km/u.s, velocity_convention='optical', rest_value=2.1654*u.micron)

		fit_ind1 = np.where(spec_kms.spectral_axis > galv-2000* u.km/u.s)
		fit_ind2 = np.where(spec_kms.spectral_axis < galv+2000* u.km/u.s)
		fit_ind = np.intersect1d(fit_ind1, fit_ind2)
		fit_spec = spec_kms[fit_ind]
		fit_spec_kms = spec_kms.spectral_axis[fit_ind].value
		fit_spec_ind = fit_ind


		cont_fit, (slope, intercept) = fit_continuum(fit_spec, fit_spec_kms)
		cont_sub_spec = spec - (spec_kms*slope + intercept)

		tel_flux = np.nansum(cont_sub_spec[sub_inds[0]:sub_inds[1]]) * 1e16

		ax = fig.add_subplot(gs[i//4, i%4])

		ax.plot(fit_spec_ind, fit_spec*1e16, linestyle='-', color='k')
		ax.plot(fit_spec_ind, cont_fit*1e16, linestyle='-', color='tab:orange')

		ax.axvline(sub_inds[0])
		ax.axvline(sub_inds[1])

		#ax.set_xlim(775, 825)
		ax.set_xlim(755, 805)

		ax.tick_params(axis='y', labelleft=False)

		if i//4 == 3:
			ax.tick_params(axis='x', labelbottom=True, labelsize=12)
		else:
			ax.tick_params(axis='x', labelbottom=False)

		ax.text(0.05, 0.9, f'({loc[0]}, {loc[1]})', transform=ax.transAxes)
		ax.text(0.05, 0.05, f'{np.round(tel_flux, 3)}', transform=ax.transAxes)


	plt.savefig(f'plots/{savename}.png', dpi=300, bbox_inches='tight')



loc_list = [(22,22), (22, 32), (22,42), (22, 52), 
			(32,22), (32, 32), (32,42), (32, 52), 
			(52,22), (52, 32), (52,42), (52, 52), 
			(62, 22), (62, 32), (62, 42), (62, 52)]

sub_inds = [795, 805]

#plot_specs(loc_list, sub_inds, 'orig_tellsub_plot_1st', which_cube='orig')

sub_inds = [775, 785]
plot_specs(loc_list, sub_inds, 'dec6_tellsub_plot_1st', which_cube='alt')

