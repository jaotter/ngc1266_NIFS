from astropy.io import fits
from scipy.optimize import curve_fit
from scipy import odr
from astropy.table import Table
from astropy.wcs import WCS
from spectral_cube import SpectralCube

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const
import pandas as pd
import os
import glob
import datetime
from time import perf_counter as clock
from matplotlib.lines import Line2D

from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util
import ppxf.miles_util as lib
import ppxf_custom_util_nifs as custom_util

from astropy.stats import mad_std
from scipy.interpolate import interp1d
from astropy.cosmology import FlatLambdaCDM

#from ppxf_ap_spec import load_xsl_temps, output_ppxf_fit_plot, fit_line, measure_spectrum_err, ppxf_fit_stellar

#########
# Script to fit NIFS lines with gaussian

#line_dict = {'H2(1-0)S(2)':2.0332, 'H2(1-0)S(1)':2.1213, 'H2(1-0)S(0)':2.2230, 'H2(2-1)S(1)':2.2470, 'H2(2-1)S(2)':2.1542, 'Brgamma':2.1654, 'H2(2-1)S(3)':2.0735, 'H2(1-0)Q(1)':2.4066}
line_dict = {'H2(1-0)S(1)':2.1213, 'Brgamma_1comp':2.1654, 'Brgamma_2comp':2.1654}
#line_dict = {'H2(1-0)S(1)':2.1213, 'H2(1-0)Q(1)':2.4066}


z = 0.007214         # NGC 1266 redshift, from SIMBAD
galv = np.log(z+1)*const.c.to(u.km/u.s) # estimate of galaxy's velocity


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

	wave_low_ind = np.searchsorted(data_wave, max_start)
	wave_high_ind = np.searchsorted(data_wave, min_end)

	wave_trunc = data_wave[wave_low_ind:wave_high_ind]

	temp_rebin = []

	for i in range(len(temp_paths)):
		interp_func = interp1d(temp_wave_nobin[i], temp_data_nobin[i])
		new_temp = interp_func(wave_trunc)
		temp_rebin.append(new_temp)


	return np.array(temp_rebin).transpose(), wave_trunc #to be consistent with ppxf MILES loading


def output_ppxf_fit_plot(runid, plot_name, pp, good_pix_total, logLam, vel_comp0, z, ncomp, mad_std_residuals, SN_wMadStandardDev, fit_gas):

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

	full_plot_dir = f'{plot_dir}/ppxf_{runid}/'
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



def ppxf_fit_stellar(spectrum, err, moments, start, adegree, mdegree, wave_lam, runID, plot_name=None):
	#cube - unbinned data cube
	#err
	#start
	#moments - ppxf moments for fitting
	#adegree - degree of additive poly
	#mdegree - degree of multiplicative poly
	#wave_lam - wavelength array
	#plot_name - name for output plots

	save_dict = {'star_vel':[], 'star_vel_error':[], 'star_sig':[], 'star_sig_error':[],
	'SN_mad_STD':[], 'SN_STD':[], 'chisq/dof':[], 'contsub_spec':[]}

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

	#find nans, set them to median values and then remove from good pixels
	nan_ind = np.where(np.isnan(spec_rebin))
	spec_rebin[nan_ind] = np.nanmedian(spec_rebin)

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

	#choose line free region to compute S/N
	#2.095 - 2.125
	#2.19 - 2.23 micron

	ind1_1 = np.where(wave_rebin > 20950)[0]
	ind1_2 = np.where(wave_rebin < 21250)[0]
	ind1 = np.intersect1d(ind1_1, ind1_2)

	ind2_1 = np.where(wave_rebin > 21900)[0]
	ind2_2 = np.where(wave_rebin < 22300)[0]
	ind2 = np.intersect1d(ind2_1, ind2_2)

	linefree_ind = np.concatenate((ind1,ind2))

	mad_std_residuals = mad_std(contsub_spec[linefree_ind], ignore_nan=True)

	std_residuals = np.nanstd(contsub_spec[linefree_ind])
	med_spec = np.nanmedian(spec_rebin[linefree_ind])
	SN_wMadStandardDev = med_spec/mad_std_residuals 
	SN_std = med_spec/std_residuals

	save_dict['SN_mad_STD'].append(SN_wMadStandardDev)
	save_dict['SN_STD'].append(SN_std)
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
	#print('============================================================================')
	#print('Desired Delta Chi^2: %.4g' % np.sqrt(2*spec_rebin.size))
	#print('Current Delta Chi^2: %.4g' % (fit_chisq))
	#print('Elapsed time in PPXF: %.2f s' % (clock() - t))
	#print('============================================================================')

	if plot_name is not None:
		output_ppxf_fit_plot(runID, plot_name, pp, good_pix_total, log_wave_rebin, vel_comp0, z, 0, 
							mad_std_residuals, SN_wMadStandardDev, fit_gas=False)


	return save_dict, optimal_template, spec_rebin, bestfit, wave_rebin

def spiral_traverse(matrix): #from codepal
    """
    This function takes a 2D array (matrix) as input and returns a list of all the elements in the matrix
    in spiral order, starting from the center and moving outwards.
    
    Parameters:
    matrix (list of lists): The 2D array to be traversed
    
    Returns:
    list: A list of all the elements in the matrix in spiral order
    """
    # Initialize variables
    result = []
    rows = len(matrix)
    cols = len(matrix[0])
    row_start = 0
    row_end = rows - 1
    col_start = 0
    col_end = cols - 1
    
    # Traverse the matrix in spiral order
    while row_start <= row_end and col_start <= col_end:
        # Traverse right
        for i in range(col_start, col_end + 1):
            result.append(matrix[row_start][i])
        row_start += 1
        
        # Traverse down
        for i in range(row_start, row_end + 1):
            result.append(matrix[i][col_end])
        col_end -= 1
        
        # Traverse left
        if row_start <= row_end:
            for i in range(col_end, col_start - 1, -1):
                result.append(matrix[row_end][i])
            row_end -= 1
        
        # Traverse up
        if col_start <= col_end:
            for i in range(row_end, row_start - 1, -1):
                result.append(matrix[i][col_start])
            col_start += 1
    
    result = np.flip(result)

    return result


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



def compute_flux(popt, cov_params, line_wave, ncomp=2):
	#take fit parameter output and compute flux and error

	amp1 = popt[1] #* u.erg/u.s/u.cm**2/u.AA
	width1 = popt[2] * u.km/u.s
	wave_width1 = ((width1 / const.c) * line_wave).to(u.angstrom).value

	amp1_err = cov_params[1] #wrong with unit conversion
	width1_err = cov_params[2] * u.km/u.s
	wave_width1_err = ((width1_err / const.c) * line_wave).to(u.angstrom).value

	if ncomp == 2:
		amp2 = popt[4] #* u.erg/u.s/u.cm**2/u.AA
		width2 = popt[5] * u.km/u.s
		wave_width2 = ((width2 / const.c) * line_wave).to(u.angstrom).value

		amp2_err = cov_params[4]
		width2_err = cov_params[5] * u.km/u.s
		wave_width2_err = ((width2_err / const.c) * line_wave).to(u.angstrom).value

		total_flux = np.sqrt(2 * np.pi) * ((wave_width1 * amp1) + (wave_width2 * amp2))
		total_flux_err = np.sqrt(2*np.pi) * np.sqrt((wave_width1*amp1_err)**2 + (amp1*wave_width1_err)**2 + (wave_width2*amp2_err)**2 + (amp2*wave_width2_err)**2)

	else:
		total_flux = np.sqrt(2 * np.pi) * ((wave_width1 * amp1))
		total_flux_err = np.sqrt(2*np.pi) * np.sqrt((wave_width1*amp1_err)**2 + (amp1*wave_width1_err)**2)


	return total_flux, total_flux_err





def fit_cube(runID, cube_path='/Users/jotter/highres_PSBs/ngc1266_data/NIFS_data/NGC1266_NIFS_final_trim_wcs.fits', ncomp=2):
	## fit the NIFS cube, fitting each line with the desired number of components
	# loop through starting at the center and spiral outwards, with the previous bin acting as the the starting parameters for the next
	
	hdu = fits.open(cube_path)
	cube_data = hdu[1].data
	h1 = hdu[1].header
	nifs_wcs = WCS(h1)

	cube = SpectralCube(data=cube_data, wcs=nifs_wcs)
	wave_lam = (cube.spectral_axis.to(u.Angstrom))

	#error_cube = hdu[2].data

	#obs_wave = (np.array(h1['CRVAL3']+(np.arange(0, h1['NAXIS3'])*h1['CDELT3'])) * u.meter).to(u.Angstrom)


	#assigning each pixel a bin number
	binNum = np.reshape(np.arange(cube_data.shape[1]*cube_data.shape[2]), (cube_data.shape[1], cube_data.shape[2]))
	x,y = np.meshgrid(np.arange(cube_data.shape[2]), np.arange(cube_data.shape[1]))

	#write code to automate this for all lines in line_list
	save_dict = {'bin_num':[], 'x':[], 'y':[], 'stell_vel':[], 'stell_vel_err':[], 'stell_sig':[], 'stell_sig_err':[], 
				'cont_SN_madstd':[], 'cont_SN_std':[], 'vel_c1':[], 'vel_c1_err':[], 'vel_c2':[], 'vel_c2_err':[], 'sigma_c1':[],
				'sigma_c1_err':[], 'sigma_c2':[], 'sigma_c2_err':[],}
	for line_name in line_dict.keys():
		save_dict[f'{line_name}_flux_c1'] = []
		save_dict[f'{line_name}_flux_err_c1'] = []
		save_dict[f'{line_name}_flux_c2'] = []
		save_dict[f'{line_name}_flux_err_c2'] = []
		save_dict[f'{line_name}_flux_c2'] = []
		save_dict[f'{line_name}_flux_err_c2'] = []
		save_dict[f'{line_name}_flux'] = []
		save_dict[f'{line_name}_flux_err'] = []
		save_dict[f'{line_name}_mask'] = []


	prev_fit_params = None

	#loop_list = binNum.flatten()

	loop_list = spiral_traverse(binNum)

	for i, bn in enumerate(loop_list): 

		print('\n============================================================================')

		b_loc = np.where(binNum == bn)
		x_loc = x[b_loc]
		y_loc = y[b_loc]

		print(f'binNum: {bn}, x,y: {x_loc[0], y_loc[0]}')
		print(f'{np.round(100 * (float(i)/len(loop_list)), 2)}% complete, {i}/{len(loop_list)}')

		if bn % 10 == 0:
			plot_bool = True
			plotname = f'x{x_loc[0]}y{y_loc[0]}'
		else:
			plot_bool = False
			plotname = None

		spectrum = (cube_data[:,y_loc,x_loc] * u.erg / u.cm**2 / u.s / u.AA).squeeze()
		#spectrum_err = np.sqrt(np.abs(error_cube[:,y_loc,x_loc])) #sqrt bc this cube is the variance
		

		if len(spectrum[np.isnan(spectrum)]) > 0.2 * len(spectrum) or len(spectrum[spectrum==0]) > 0.2 * len(spectrum):
			print('bad spectrum, skipping')
			continue

		save_dict['bin_num'].append(bn)
		save_dict['x'].append(x_loc[0])
		save_dict['y'].append(y_loc[0])

		line_name = 'H2(1-0)S(1)'
		line_wave = line_dict[line_name] * u.micron

		#rough error estimation for ppxf fit - will do better after ppxf fit
		rough_err_bounds = [2.09, 2.11] * u.micron
		err_ind1 = np.where(wave_lam > rough_err_bounds[0])
		err_ind2 = np.where(wave_lam < rough_err_bounds[1])
		err_ind = np.intersect1d(err_ind1, err_ind2)
		err = np.nanstd(spectrum[err_ind])

		#ppxf parameters
		moments = 2
		adegree = 10     		# Additive polynomial used to correct the template continuum shape during the fit.
								# Degree = 10 is the suggested order for stellar populations.
		mdegree = 10			# Multiplicative polynomial used to correct the template continuum shape during the fit.
								# Mdegree = -1 tells fit to not include the multiplicative polynomial.

		start = [galv.value, 25]

		#ppxf fit
		print('Running ppxf')
		ppxf_dict, opt_temps_save, spec_rebin, cont_spec, wave_rebin = ppxf_fit_stellar(spectrum.value, err.value, moments, start, 
																						adegree, mdegree, wave_lam.value, runID, plot_name=plotname)

		#save ppxf kinematics
		save_dict['stell_vel'].append(ppxf_dict['star_vel'][0])
		save_dict['stell_vel_err'].append(ppxf_dict['star_vel_error'][0])
		save_dict['stell_sig'].append(ppxf_dict['star_sig'][0])
		save_dict['stell_sig_err'].append(ppxf_dict['star_sig_error'][0])
		save_dict['cont_SN_madstd'].append(ppxf_dict['SN_mad_STD'][0])
		save_dict['cont_SN_std'].append(ppxf_dict['SN_STD'][0])


		contsub_spec = spec_rebin - cont_spec
		cont_serr, cont_std = measure_spectrum_err(contsub_spec, wave_rebin)
		cont_std_err = np.repeat(cont_std, len(spec_rebin))

		wave_to_vel = u.doppler_optical(line_wave)
		wave_vel = (wave_rebin*u.Angstrom).to(u.km/u.s, equivalencies=wave_to_vel) - galv

		fit_ind1 = np.where(wave_vel > -2000* u.km/u.s)
		fit_ind2 = np.where(wave_vel < 2000* u.km/u.s)
		fit_ind = np.intersect1d(fit_ind1, fit_ind2)

		line_spec_cs = spec_rebin[fit_ind] - cont_spec[fit_ind]
		line_cont = cont_spec[fit_ind]
		line_vel = wave_vel[fit_ind]
		line_err = cont_std_err[fit_ind]

		#line_spec_err = np.sqrt(np.abs(spectrum_err[fit_ind].squeeze())) 

		peak_flux = np.nanmax(line_spec_cs)
		peak_ind = np.where(line_spec_cs == peak_flux)[0][0]
		peak_vel = line_vel[peak_ind].value

		if np.abs(peak_vel) > 500:
			peak_vel = 0

		if prev_fit_params is None:
			#initial guess if of 2 components, one wider with 1/3 the flux
			start = [peak_vel, peak_flux, 100, peak_vel, peak_flux*0.25, 200]

			#format is ([lower bounds], [upper bounds])
			bounds = ([-700, peak_flux * 0.1, 20, -700, 0, 50], [700, peak_flux*10, 600, 700, peak_flux*10, 400])

			'''elif np.abs(prev_fit_params[0] - galv.value) > 1000: #if the previous fit guess is getting way off, then re-center
									#initial guess if of 2 components, one wider with 1/3 the flux
									start = [peak_vel, peak_flux, 200, peak_vel, peak_flux*0.25, 400]
						
									#format is ([lower bounds], [upper bounds])
									bounds = ([galv.value - 500, peak_flux * 0.1, 20, galv.value - 500, 0, 50], [galv.value + 500, peak_flux*10, 600, galv.value + 500, peak_flux*10, 600])'''
						
		else:
			component_offset = prev_fit_params[0] - prev_fit_params[3] #offset btwn 2 gaussians
			#comp2_start = prev_fit_params[0] - component_offset/5
			comp2_start = peak_vel - component_offset/5 #changing start to use peak velocity


			#start = [prev_fit_params[0], peak_flux, prev_fit_params[2], comp2_start, peak_flux*0.25, prev_fit_params[5]]
			start = [peak_vel, peak_flux, prev_fit_params[2], comp2_start, peak_flux*0.25, prev_fit_params[5]] #use peak velocity

			#start = [prev_fit_params[0], peak_flux, prev_fit_params[2], prev_fit_params[0], peak_flux*0.25, prev_fit_params[5]]
			#currently these bounds are not very limiting
			width_upper_1 = np.min((prev_fit_params[2]*2, 400))
			width_upper_2 = np.min((prev_fit_params[5]*2, 400))
			width_lower_1 = np.max((prev_fit_params[2]*0.5, 20))
			width_lower_2 = np.max((prev_fit_params[5]*0.5, 20))

			#if start[2] <= 21:
			#	start[2] = 25
			#if start[5] <= 21:
			#	start[5] = 25
			#if start[2] >= 399:
			#	start[2] = 390
			#if start[5] >= 399:
			#	start[5] = 390

			#bounds = ([prev_fit_params[0]-400, peak_flux*0.1, width_lower_1, prev_fit_params[0]-700, peak_flux*0.01, width_lower_2],
			#	[prev_fit_params[0]+400, peak_flux*3, width_upper_1, prev_fit_params[0]+700, peak_flux*3, width_upper_2])
			#bounds = ([prev_fit_params[0]-400, peak_flux*0.1, width_lower_1, comp2_start-600, peak_flux*0.01, width_lower_2],
			#	[prev_fit_params[0]+400, peak_flux*2, width_upper_1, comp2_start+600, peak_flux*2, width_upper_2])
			bounds = ([peak_vel-400, peak_flux*0.1, width_lower_1, comp2_start-600, peak_flux*0.01, width_lower_2],
				[peak_vel+400, peak_flux*2, width_upper_1, comp2_start+600, peak_flux*2, width_upper_2]) #use peak velocity

		ncomp = 2
		popt, pcov = fit_line(line_spec_cs, line_vel.value, line_err, line_name, start, bounds, (x_loc[0],y_loc[0]), runID, ncomp=2, plot=plot_bool)

		#fit_line(line_spec, line_vel.value, cont_std_err, cont_params, line_name, start, bounds, (x_loc[0],y_loc[0]), runID, plot=plot_bool, mask_ind=mask_ind, ncomp=ncomp)

		if popt[2] <= popt[5]: #if 1st component is thinner than 2nd
			comp1_params = popt[0:3]
			comp2_params = popt[3:6]
			comp1_cov = [np.sqrt(pcov[0,0]), np.sqrt(pcov[1,1]), np.sqrt(pcov[2,2])]
			comp2_cov = [np.sqrt(pcov[3,3]), np.sqrt(pcov[4,4]), np.sqrt(pcov[5,5])]
		else:
			comp1_params = popt[3:6]
			comp2_params = popt[0:3]
			comp1_cov = [np.sqrt(pcov[3,3]), np.sqrt(pcov[4,4]), np.sqrt(pcov[5,5])]
			comp2_cov = [np.sqrt(pcov[0,0]), np.sqrt(pcov[1,1]), np.sqrt(pcov[2,2])]
			
		full_params = np.concatenate((comp1_params, comp2_params))
		full_cov = np.concatenate((comp1_cov, comp2_cov))

		total_flux, total_flux_err = compute_flux(full_params, full_cov, line_wave, ncomp=2)
		flux_c1, flux_err_c1 = compute_flux(comp1_params, comp1_cov, line_wave, ncomp=1)
		flux_c2, flux_err_c2 = compute_flux(comp2_params, comp2_cov, line_wave, ncomp=1)

		line_ind1 = np.where(line_vel > -800* u.km/u.s)
		line_ind2 = np.where(line_vel < 800* u.km/u.s)
		line_ind = np.intersect1d(line_ind1, line_ind2)
		sum_flux = np.nansum(line_spec_cs[line_ind])

		'''if sum_flux / total_flux > 3:
									print('Summed flux much larger than gaussian fit, re-fitting')
						
									start = [peak_vel, peak_flux, 100, peak_vel, peak_flux*0.25, 200]
									bounds = ([-700, peak_flux * 0.1, 20, -700, 0, 50], [700, peak_flux*10, 600, 700, peak_flux*10, 400])
						
									popt, pcov = fit_line(line_spec, line_vel.value, cont_std_err, cont_params, line_name, start, bounds, (x_loc[0],y_loc[0]), 
															runID, plot=True, mask_ind=mask_ind, ncomp=ncomp)
						
									total_flux, total_flux_err = compute_flux(popt, pcov, line_wave)
						
									line_ind1 = np.where(line_vel > -800* u.km/u.s)
									line_ind2 = np.where(line_vel < 800* u.km/u.s)
									line_ind = np.intersect1d(line_ind1, line_ind2)
									sum_flux = np.nansum(line_spec[line_ind])'''

		save_dict[f'{line_name}_flux'].append(total_flux)
		save_dict[f'{line_name}_flux_err'].append(total_flux_err)
		save_dict[f'{line_name}_flux_c1'].append(flux_c1)
		save_dict[f'{line_name}_flux_err_c1'].append(flux_err_c1)
		save_dict[f'{line_name}_flux_c2'].append(flux_c2)
		save_dict[f'{line_name}_flux_err_c2'].append(flux_err_c2)

		if total_flux >= total_flux_err * 3:
			save_dict[f'{line_name}_mask'].append(1)
		else:
			save_dict[f'{line_name}_mask'].append(0)

		save_dict['vel_c1'].append(comp1_params[0])
		save_dict['vel_c1_err'].append(comp1_cov[0])
		save_dict['vel_c2'].append(comp2_params[0])
		save_dict['vel_c2_err'].append(comp2_cov[0])

		save_dict['sigma_c1'].append(comp1_params[2])
		save_dict['sigma_c1_err'].append(comp1_cov[2])
		save_dict['sigma_c2'].append(comp2_params[2])
		save_dict['sigma_c2_err'].append(comp2_cov[2])

		prev_fit_params = full_params

		for line_name in line_dict.keys():
			if line_name == 'H2(1-0)S(1)':
				continue

			line_wave = line_dict[line_name] * u.micron

			#cube_vel_obs = cube.with_spectral_unit(u.km/u.s, velocity_convention='optical', rest_value=line_wave)
			#wave_vel = cube_vel_obs.spectral_axis - galv

			wave_to_vel = u.doppler_optical(line_wave)
			wave_vel = (wave_rebin*u.Angstrom).to(u.km/u.s, equivalencies=wave_to_vel) - galv

			fit_ind1 = np.where(wave_vel > -2000* u.km/u.s)
			fit_ind2 = np.where(wave_vel < 2000* u.km/u.s)
			fit_ind_line = np.intersect1d(fit_ind1, fit_ind2)

			if line_name == 'Brgamma_1comp' or line_name == 'Brgamma_2comp':
				mask_ind = [81,83] #exclude these from fit_ind
				fit_ind = np.concatenate((fit_ind[0:mask_ind[0]], fit_ind[mask_ind[1]:-1]))


			line_spec_cs = spec_rebin[fit_ind_line] - cont_spec[fit_ind_line]
			line_cont = cont_spec[fit_ind_line]
			line_vel = wave_vel[fit_ind_line]
			line_err = cont_std_err[fit_ind_line]


			'''if line_name == 'H2(1-0)Q(1)': #fixing situations where the last number of points are all zeros
									
													line_spec_cs = np.trim_zeros(line_spec_cs, 'b')
													line_vel = line_vel[:len(line_spec)]
									
													if len(line_vel) == 0:
														save_dict[f'{line_name}_mask'].append(0)
														save_dict[f'{line_name}_flux'].append(np.nan)
														save_dict[f'{line_name}_flux_err'].append(np.nan)
														continue
													elif line_vel[-1] < 350*u.km/u.s: #only fit where there is enough data at the end
														save_dict[f'{line_name}_mask'].append(0)
														save_dict[f'{line_name}_flux'].append(np.nan)
														save_dict[f'{line_name}_flux_err'].append(np.nan)
														continue'''

			peak_flux = np.nanmax(line_spec_cs)
			
			if np.abs(peak_flux) < 1e-30: #there is nothing here, not even noise
				save_dict[f'{line_name}_mask'].append(0)
				save_dict[f'{line_name}_flux'].append(np.nan)
				save_dict[f'{line_name}_flux_err'].append(np.nan)
				save_dict[f'{line_name}_flux_c1'].append(np.nan)
				save_dict[f'{line_name}_flux_err_c1'].append(np.nan)
				save_dict[f'{line_name}_flux_c2'].append(np.nan)
				save_dict[f'{line_name}_flux_err_c2'].append(np.nan)
				continue

			if peak_flux <= 2*cont_std: #skip fitting and go straight to upper limit calculation
				if line_name == 'Brgamma_1comp' or 'Brgamma_2comp':
						ulim_width = 150 #km/s, from Davis12 spectrum
				else:
					ulim_width = np.max((prev_fit_params[2], prev_fit_params[5])) #larger width of the two gaussians

				ulim_amp = 3 * cont_serr
				ulim_flux = np.sqrt(2 * np.pi) * (ulim_width * ulim_amp)

				save_dict[f'{line_name}_mask'].append(0)
				save_dict[f'{line_name}_flux'].append(ulim_flux)
				save_dict[f'{line_name}_flux_err'].append(np.nan)
				save_dict[f'{line_name}_flux_c1'].append(ulim_flux)
				save_dict[f'{line_name}_flux_err_c1'].append(np.nan)
				save_dict[f'{line_name}_flux_c2'].append(ulim_flux)
				save_dict[f'{line_name}_flux_err_c2'].append(np.nan)

				continue

			#use H2_212 fit for initial conditions
			#initial guess is of 2 components, one wider with 1/4 the flux

			start = [prev_fit_params[0], peak_flux, prev_fit_params[2], prev_fit_params[0], peak_flux*0.25, prev_fit_params[5]]
			#currently these bounds are not very limiting
			width_upper_1 = np.min((prev_fit_params[2]*2, 400))
			width_upper_2 = np.min((prev_fit_params[5]*2, 400))
			width_lower_1 = np.max((prev_fit_params[2]*0.5, 20))
			width_lower_2 = np.max((prev_fit_params[5]*0.5, 20))
			bounds = ([prev_fit_params[0]-400, peak_flux*0.1, width_lower_1, prev_fit_params[0]-700, peak_flux*0.01, width_lower_2],
						[prev_fit_params[0]+400, peak_flux*3, width_upper_1, prev_fit_params[0]+700, peak_flux*3, width_upper_2])


			#if line_name == 'H2(1-0)Q(1)': #need custom fitting for this line bc otherwise sometimes it goes out of bounds w prev fit guess
			#peak_flux = np.nanmax(line_spec)
			#peak_ind = np.where(line_spec == peak_flux)[0][0]
			#peak_vel = line_vel[peak_ind].value

			#start = [peak_vel, peak_flux, 200, peak_vel, peak_flux*0.25, 400]

			#bounds = ([peak_vel - 500, peak_flux * 0.1, 20, peak_vel - 500, 0, 50], [line_vel[-1].value, peak_flux*10, 600, line_vel[-1].value, peak_flux*10, 600])

			if line_name == 'Brgamma_1comp':
				start = start[0:3]
				bounds = (bounds[0][0:3], bounds[1][0:3])
				ncomp = 1
			else:
				ncomp = 2

			popt, pcov = fit_line(line_spec_cs, line_vel.value, line_err, line_name, start, bounds, (x_loc[0],y_loc[0]), runID, ncomp=ncomp, plot=plot_bool)


			if ncomp == 1:
				flux_c1 = np.nan
				flux_err_c1 = np.nan
				flux_c2 = np.nan
				flux_err_c2 = np.nan

				comp1_params = popt[0:3]
				comp1_cov = [np.sqrt(pcov[0,0]), np.sqrt(pcov[1,1]), np.sqrt(pcov[2,2])]

				total_flux, total_flux_err = compute_flux(comp1_params, comp1_cov, line_wave, ncomp=1)

			if ncomp == 2:
				if popt[2] <= popt[5]: #if 1st component is thinner than 2nd
					comp1_params = popt[0:3]
					comp2_params = popt[3:6]
					comp1_cov = [np.sqrt(pcov[0,0]), np.sqrt(pcov[1,1]), np.sqrt(pcov[2,2])]
					comp2_cov = [np.sqrt(pcov[3,3]), np.sqrt(pcov[4,4]), np.sqrt(pcov[5,5])]
				else:
					comp1_params = popt[3:6]
					comp2_params = popt[0:3]
					comp1_cov = [np.sqrt(pcov[3,3]), np.sqrt(pcov[4,4]), np.sqrt(pcov[5,5])]
					comp2_cov = [np.sqrt(pcov[0,0]), np.sqrt(pcov[1,1]), np.sqrt(pcov[2,2])]
					
				full_params = np.concatenate((comp1_params, comp2_params))
				full_cov = np.concatenate((comp1_cov, comp2_cov))

				total_flux, total_flux_err = compute_flux(full_params, full_cov, line_wave, ncomp=2)
				flux_c1, flux_err_c1 = compute_flux(comp1_params, comp1_cov, line_wave, ncomp=1)
				flux_c2, flux_err_c2 = compute_flux(comp2_params, comp2_cov, line_wave, ncomp=1)


			if total_flux >= total_flux_err * 3:#if 3-sigma detection
				save_dict[f'{line_name}_flux'].append(total_flux)
				save_dict[f'{line_name}_flux_err'].append(total_flux_err)
				save_dict[f'{line_name}_flux_c1'].append(flux_c1)
				save_dict[f'{line_name}_flux_err_c1'].append(flux_err_c1)
				save_dict[f'{line_name}_flux_c2'].append(flux_c2)
				save_dict[f'{line_name}_flux_err_c2'].append(flux_err_c2)
				save_dict[f'{line_name}_mask'].append(1)

			else: #compute upper limit
				if line_name == 'Brgamma_1comp' or line_name == 'Brgamma_2comp':
					ulim_width = 150 #km/s, from Davis12 spectrum
				else:
					ulim_width = np.max((prev_fit_params[2], prev_fit_params[5])) #larger width of the two gaussians

				ulim_amp = 3 * cont_serr
				ulim_flux = np.sqrt(2 * np.pi) * (ulim_width * ulim_amp)

				save_dict[f'{line_name}_mask'].append(0)
				save_dict[f'{line_name}_flux'].append(ulim_flux)
				save_dict[f'{line_name}_flux_err'].append(np.nan)
				save_dict[f'{line_name}_flux_c1'].append(np.nan)
				save_dict[f'{line_name}_flux_err_c1'].append(np.nan)
				save_dict[f'{line_name}_flux_c2'].append(np.nan)
				save_dict[f'{line_name}_flux_err_c2'].append(np.nan)

	csv_save_name = f'/Users/jotter/highres_PSBs/ngc1266_NIFS/fit_output/{runID}_gaussfit.csv'

	run_dict_df = pd.DataFrame.from_dict(save_dict)
	run_dict_df.to_csv(csv_save_name, index=False, header=True)

	print(f'Saved emission line fit csv to {csv_save_name}')


def csv_to_maps(csv_path, save_name):


	fit_tab = Table.read(csv_path, format='csv')


	cube_file = '/Users/jotter/highres_PSBs/ngc1266_data/NIFS_data/NGC1266_NIFS_jun28_wcs.fits'
	#cube_file = '/Users/jotter/highres_PSBs/ngc1266_data/NIFS_data/reduced_cubes/man_offset2/20141110_obs49_merged_wcs_trim.fits'
	cube_fl = fits.open(cube_file)
	cube_header = cube_fl[1].header
	cube_wcs = WCS(cube_header).celestial
	cube_data = cube_fl[1].data
	cube_fl.close()

	map_shape = cube_data.shape[1:3]

	binNum_2D = np.full(map_shape, np.nan)

	n_maps = len(line_dict.keys())*3 + 14
	map_cube = np.full((n_maps,map_shape[0],map_shape[1]), np.nan)

	for ind, bn in enumerate(fit_tab['bin_num']):
		x_loc = fit_tab['x'][ind]
		y_loc = fit_tab['y'][ind]

		binNum_2D[y_loc, x_loc] = bn

		map_cube[0, y_loc, x_loc] = fit_tab['stell_vel'][ind]
		map_cube[1, y_loc, x_loc] = fit_tab['stell_vel_err'][ind]
		map_cube[2, y_loc, x_loc] = fit_tab['stell_sig'][ind]
		map_cube[3, y_loc, x_loc] = fit_tab['stell_sig_err'][ind]

		map_cube[4, y_loc, x_loc] = fit_tab['vel_c1'][ind]
		map_cube[5, y_loc, x_loc] = fit_tab['vel_c1_err'][ind]
		map_cube[6, y_loc, x_loc] = fit_tab['vel_c2'][ind]
		map_cube[7, y_loc, x_loc] = fit_tab['vel_c2_err'][ind]

		map_cube[8, y_loc, x_loc] = fit_tab['sigma_c1'][ind]
		map_cube[9, y_loc, x_loc] = fit_tab['sigma_c1_err'][ind]
		map_cube[10, y_loc, x_loc] = fit_tab['sigma_c2'][ind]
		map_cube[11, y_loc, x_loc] = fit_tab['sigma_c2_err'][ind]

		map_cube[12, y_loc, x_loc] = fit_tab['cont_SN_madstd'][ind]
		map_cube[13, y_loc, x_loc] = fit_tab['cont_SN_std'][ind]

		for map_ind, line_name in enumerate(line_dict.keys()):

			line_flux = fit_tab[f'{line_name}_flux'][ind]
			line_flux_err = fit_tab[f'{line_name}_flux_err'][ind]
			line_mask = fit_tab[f'{line_name}_mask'][ind]

			map_cube[map_ind*3+14, y_loc, x_loc] = line_flux
			map_cube[map_ind*3+1+14, y_loc, x_loc] = line_flux_err
			map_cube[map_ind*3+2+14, y_loc, x_loc] = line_mask

	maps_header = cube_wcs.to_header()

	maps_header['DESC0'] = 'Stellar velocity'
	maps_header['DESC1'] = 'Stellar velocity error'
	maps_header['DESC2'] = 'Stellar sigma'
	maps_header['DESC3'] = 'Stellar sigma error'

	maps_header['DESC4'] = 'Velocity component 1'
	maps_header['DESC5'] = 'Velocity component 1 error'
	maps_header['DESC6'] = 'Velocity component 2'
	maps_header['DESC7'] = 'Velocity component 2 error'

	maps_header['DESC8'] = 'Sigma component 1'
	maps_header['DESC9'] = 'Sigma component 1 error'
	maps_header['DESC10'] = 'Sigma component 2'
	maps_header['DESC11'] = 'Sigma component 2 error'

	maps_header['DESC12'] = 'Continuum SN (mad_std)'
	maps_header['DESC13'] = 'Continuum SN (std)'

	for map_ind, line_name in enumerate(line_dict.keys()):
		maps_header[f'DESC{map_ind*3+14}'] = f'{line_name}_flux'
		maps_header[f'DESC{map_ind*3+15}'] = f'{line_name}_flux_err'
		maps_header[f'DESC{map_ind*3+16}'] = f'{line_name}_mask'

	maps_header['FLUX_UNIT'] = 'erg/s/cm**2'

	new_hdu = fits.PrimaryHDU(map_cube, header=maps_header)
	hdulist = fits.HDUList([new_hdu])

	savefile = f'/Users/jotter/highres_PSBs/ngc1266_NIFS/fit_output/{save_name}.fits'

	hdulist.writeto(savefile, overwrite=True)

	print(f'saved to {savefile}')

