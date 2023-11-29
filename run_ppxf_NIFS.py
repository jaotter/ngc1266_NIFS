#script to run ppxf on MUSE data, adapted from Celia Mulcahey's interactive notebook

import glob
import datetime
import os
from time import perf_counter as clock
from astropy.io import fits
import numpy as np

from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util
import ppxf.miles_util as lib
import ppxf_custom_util_nifs as custom_util

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from astropy.stats import mad_std 
import pandas as pd
from scipy.interpolate import interp1d

#Constants, wavelength in Ang in vacuum------------------------------------------- 

c = 299792.458

H2_203 = 20332 #in angstroms
H2_212 = 21213
H2_222 = 22230
H2_224 = 22470
H2_215 = 21542
Brgamma = 21654
CIV = 20780

def load_Winge_temps(data_wave, temp_folder="/Users/jotter/ppxf_files/Winge2009/*.fits"):
	temp_paths = glob.glob(temp_folder)

	temp_starts = []
	temp_delts = []
	temp_ends = []

	temp_data_nobin = []
	temp_wave_nobin = []

	for tp in temp_paths:
		temp_fl = fits.open(tp)
		temp_head = temp_fl[0].header

		temp_data = temp_fl[0].data

		temp_start = temp_head['CRVAL1']
		temp_delt = temp_head['CDELT1']
		temp_len = len(temp_data)
		temp_end = temp_start + temp_delt * temp_len
		temp_starts.append(temp_start)
		temp_delts.append(temp_delt)
		temp_ends.append(temp_end)

		temp_wave = np.arange(temp_start, temp_end, temp_delt)

		if len(temp_wave) == len(temp_data) + 1:
		    temp_wave = temp_wave[:-1]

		temp_data_nobin.append(temp_data)
		temp_wave_nobin.append(temp_wave)

	max_start = np.max(temp_starts)
	min_end = np.min(temp_ends)

	wave_low_ind = np.searchsorted(data_wave, max_start)
	wave_high_ind = np.searchsorted(data_wave, min_end)-1

	wave_trunc = data_wave[wave_low_ind:wave_high_ind]

	temp_rebin = []

	for i in range(len(temp_paths)):
		interp_func = interp1d(temp_wave_nobin[i], temp_data_nobin[i])
		new_temp = interp_func(wave_trunc)
		temp_rebin.append(new_temp)

	return np.array(temp_rebin).transpose(), wave_trunc #to be consistent with ppxf MILES loading



def ANR(gas_dict, gas_name, emline, gal_lam, gas_bestfit, mad_std_residuals, velocity):
#    '''
#    Function caluclates the amplitude and amplitude-to-residual (A/rN) of specific emission-line feature following Sarzi+2006. No returns,
#    the function simply stores the amplitude and A/rN in the dictionary called gas_dict
    
#    Arguments:
#        gas_dict - dictionary where values extracting using pPPXF are stored
#        gas_name - emission-line feature name as it appears in gas_dict 
#        emline - Wavelength in vaccuum of emission-line feature in Angstroms
#        rest_gas - Spectrum, in rest frame
#        gas_bestfit - pPXF gas best fit
#        mad_std_residuals - median absolute deviation of the residuals 
#		 velocity - fitted gas velocity to get center of line

#    '''

	emline_obs = (velocity/c + 1) * emline
	emline_loc = np.where((gal_lam>emline_obs-5)&(gal_lam<emline_obs+5))
	emline_amp = np.nanmax(gas_bestfit[emline_loc])
	emline_ANR = emline_amp/mad_std_residuals 
	if gas_dict is not None:
		gas_dict[gas_name+'_amplitude'].append(emline_amp)
		gas_dict[gas_name+'_ANR'].append(emline_ANR)
	else:
		return emline_ANR


def output_ppxf_fit_plot(plot_name, pp, good_pix_total, logLam, vel_comp0, z, ncomp,
						bin_number, mad_std_residuals, SN_wMadStandardDev, fit_gas):

	print(f'Saving ppxf plot for bin number {bin_number}')
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

	bin_stell_vel = vel_comp0 - z*c
	fig.text(0.7, 0.93, f'Bin stellar velocity: {int(np.round(bin_stell_vel,0))} km/s')

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
	ax1.legend(handles=legend_elements, loc='upper left', fontsize='small', ncol=2)


	plt.sca(ax2)
	pp.plot()
	ax2.set_xlim(2.04,2.10)
	xticks = ax2.get_xticks()
	#ax2.set_xticks(xticks, labels=np.array(np.round(xticks*1e4, -1), dtype='int'))
	ax2.set_ylabel('')
	ax2.set_xlabel('')
	ax2.set_title(r'Zoom-in on NaD', fontsize = 12)
	
	plt.sca(ax3)
	pp.plot()
	ax3.set_xlim(2.3,2.4)
	xticks = ax3.get_xticks()
	#ax3.set_xticks(xticks, labels=np.array(np.round(xticks*1e4, -1), dtype='int'))
	ax3.set_title(r'Zoom-in on Ca II absorption', fontsize = 12)
	ax3.set_yticklabels([])
	ax3.set_xlabel('')
	ax3.set_ylabel('')

	full_plot_dir = f'{plot_dir}{plot_name}'
	plot_fl = f'{full_plot_dir}/ppxf_stellarfit_bin{bin_number}.png'


	
	if os.path.exists(full_plot_dir) == False:
		os.mkdir(full_plot_dir)
	plt.savefig(plot_fl, dpi = 300) 
	plt.close()

	print(f'Saved ppxf plot to {plot_name}')

def ppxf_fit_stellar(cube, error_cube, moments, adegree, mdegree, wave_lam, plot_every=0,
					plot_name=None, prev_vmap_path=None, individual_bin=None):
	#cube - unbinned data cube
	#error_cube - unbinned error cube
	#moments - ppxf moments for fitting
	#adegree - degree of additive poly
	#mdegree - degree of multiplicative poly
	#wave_lam - wavelength array
	#plot_every - if 0, don't output any plots, otherwise output a plot every [n] bins
	#plot_name - name for output plots
	#prev_fit_path - if supplied, use previous fit info
	#individual_spaxel - int - if not None, bin number of single bin to fit
	#galaxy parameters

	z = 0.007214         # NGC 1266 redshift, from SIMBAD
	galv = np.log(z+1)*c # estimate of galaxy's velocity

	save_dict = {'bin_num':[], 'star_vel':[], 'star_vel_error':[], 'star_sig':[], 'star_sig_error':[],
	'SN_mad_STD':[], 'chisq/dof':[], 'contsub_spec':[]}

	# MUSE spectral resolution, in Angstroms
	FWHM_gal = 2.51
	
	wave_lam_rest = wave_lam/(1+z)

	#preparing stellar templates
	stars_templates, wave_trunc_rest = load_Winge_temps(wave_lam_rest)
	wave_trunc = wave_trunc_rest * (1+z)
	wave_trunc_rest_range = wave_trunc_rest[[0,-1]]
	wave_trunc_range = wave_trunc[[0,-1]]

	cube_fit_ind = np.where((wave_lam_rest > wave_trunc_rest_range[0]) & (wave_lam_rest < wave_trunc_rest_range[1]))[0] #only fit rest-frame area covered by templates

	#shorten all spectra to only be within fitting area - only fit central 22" which is 110 pixels
	cube_trunc = cube[cube_fit_ind,:,:] 
	error_cube_trunc = error_cube[cube_fit_ind,:,:]

	#create cont sub empty cube 
	contsub_cube = np.full(cube_trunc.shape, fill_value=np.nan)

	#rebinning the wavelength to get the velscale
	example_spec = cube_trunc[:,65,65] #only using this for log-rebinning
	example_spec_rebin, log_wave_trunc, velscale_trunc = util.log_rebin(wave_trunc_range, example_spec)


	#assigning each pixel a bin number
	binNum = np.reshape(np.arange(cube_trunc.shape[1]*cube_trunc.shape[2]), (cube_trunc.shape[1], cube_trunc.shape[2]))
	x,y = np.meshgrid(np.arange(cube.shape[2]), np.arange(cube.shape[1]))


	#for saving optimal templates for each bin
	n_bins = len(np.unique(binNum)) 
	optimal_templates_save = np.empty((stars_templates.shape[0], n_bins))

	lam_temp = wave_trunc_rest
	lamRange_temp = [lam_temp[0], lam_temp[-1]]

	
	templates = stars_templates
	gas_component = None
	gas_names = None
	component = 0

	#velocity difference between templates and data, templates start around 3500 and data starts closer to 4500
	dv = c*(np.log(lam_temp[0]/wave_trunc[0])) # eq.(8) of Cappellari (2017)

	#if previous voronoi binned stellar velocity map supplied, use it for initial conditions
	if prev_vmap_path is not None:
		pvmap_fl = fits.open(prev_vmap_path)
		prev_vmap = pvmap_fl[1].data[1,:,:]
		prev_sigmap = pvmap_fl[1].data[2,:,:]
	else:
		prev_vmap = None

	loop_list = np.unique(binNum)
	if individual_bin is not None:
		loop_list = [individual_bin]

	for bn in loop_list: 
		save_dict['bin_num'].append(bn)

		b_loc = np.where(binNum == bn)
		x_loc = x[b_loc]
		y_loc = y[b_loc]

		spectrum = cube_trunc[:,y_loc,x_loc]
		err_spectrum = error_cube_trunc[:,y_loc,x_loc]

		print('\n============================================================================')
		print('binNum: {}'.format(bn))
		#also use previous optimal stellar template if fitting gas
		#choose starting values - just use galaxy velocity if no previous fit included

		if prev_vmap is not None:
			prev_vel = prev_vmap[y_loc, x_loc][0]
			prev_sig = prev_sigmap[y_loc, x_loc][0]

			start_vals = [prev_vel, prev_sig]

		else:
			start_vals = [galv, 25]
		
		start = start_vals
		bounds = None

		templates /= np.median(templates)

		### take mean of spectra
		#gal_lin = np.nansum(spectra, axis=1)/n_spec
		#noise_lin = np.sqrt(np.nansum(np.abs(err_spectra), axis=1))/n_spec #add error spectra in quadrature, err is variance so no need to square

		gal_lin = spectrum[:,0]
		noise_lin = err_spectrum[:,0]

		#noise_lin = noise_lin/np.nanmedian(noise_lin)
		noise_lin[np.isinf(noise_lin)] = np.nanmedian(noise_lin)
		noise_lin[noise_lin == 0] = np.nanmedian(noise_lin)
		noise_lin[np.isnan(noise_lin)] = np.nanmedian(noise_lin)

		galaxy, logLam, velscale = util.log_rebin(wave_trunc_range, gal_lin)
		#log_noise, logLam_noise, velscale_noise = util.log_rebin(lamRange, noise_lin) # I don't think this needs to be log-rebinned also

		#maskreg = (5880,5950) #galactic and extragalactic NaD in this window, observed wavelength
		#reg1 = np.where(np.exp(logLam) < maskreg[0])
		#reg2 = np.where(np.exp(logLam) > maskreg[1])
		#good_pixels = np.concatenate((reg1, reg2), axis=1)
		#good_pixels = good_pixels.reshape(good_pixels.shape[1])

		good_pix_total = custom_util.determine_goodpixels(logLam, lamRange_temp, z) #uses z to get redshifted line wavelengths, so logLam should be observer frame
		#good_pix_total = np.intersect1d(goodpix_util, good_pixels)

		#find nans, set them to median values and then remove from good pixels
		nan_ind = np.where(np.isnan(galaxy))

		if len(nan_ind[0]) > len(galaxy)/4:
			save_dict['SN_mad_STD'].append(np.nan)
			save_dict['star_vel'].append(np.nan) 
			save_dict['star_vel_error'].append(np.nan)
			save_dict['star_sig'].append(np.nan)
			save_dict['star_sig_error'].append(np.nan)
			save_dict['chisq/dof'].append(np.nan)
			save_dict['contsub_spec'].append(np.repeat(np.nan, len(galaxy)))

			continue

		galaxy[nan_ind] = np.nanmedian(galaxy)
		noise_lin[nan_ind] = np.nanmedian(noise_lin)

		goodpix_ind = np.where(good_pix_total == nan_ind)
		good_pix_total = np.delete(good_pix_total, goodpix_ind)

		noise_lin = np.abs(noise_lin)

		#CALLING PPXF HERE!
		t = clock()

		pp = ppxf(templates, galaxy, noise_lin, velscale, start,
					plot=False, moments=moments, degree= adegree, mdegree=mdegree, vsyst=dv,
					clean=False, lam=np.exp(logLam)/(1+z), #regul= 1/0.1 , reg_dim=reg_dim,
					component=component, gas_component=gas_component, bounds=bounds,
					goodpixels = good_pix_total, global_search=False)

		bestfit = pp.bestﬁt
		residuals = galaxy - bestfit

		param0 = pp.sol
		error = pp.error

		mad_std_residuals = mad_std(residuals, ignore_nan=True)    
		med_galaxy = np.nanmedian(galaxy) #this will stand in for signal probably will be ~1
		SN_wMadStandardDev = med_galaxy/mad_std_residuals 
		save_dict['SN_mad_STD'].append(SN_wMadStandardDev)
		print('S/N w/ mad_std: '+str(SN_wMadStandardDev))

		vel_comp0 = param0[0]
		vel_error_comp0 = error[0]*np.sqrt(pp.chi2)
		sig_comp0 = param0[1]
		sig_error_comp0 = error[1]*np.sqrt(pp.chi2)

		optimal_template = templates @ pp.weights
		optimal_templates_save[:,bn] = optimal_template


		save_dict['star_vel'].append(vel_comp0) 
		save_dict['star_vel_error'].append(vel_error_comp0)
		save_dict['star_sig'].append(sig_comp0)
		save_dict['star_sig_error'].append(sig_error_comp0)
		save_dict['chisq/dof'].append(pp.chi2)
		save_dict['contsub_spec'].append(bestfit)

		fit_chisq = (pp.chi2 - 1)*galaxy.size
		print('============================================================================')
		print('Desired Delta Chi^2: %.4g' % np.sqrt(2*galaxy.size))
		print('Current Delta Chi^2: %.4g' % (fit_chisq))
		print('Elapsed time in PPXF: %.2f s' % (clock() - t))
		print('============================================================================')


		if plot_every > 0 and bn % plot_every == 0:

			output_ppxf_fit_plot(plot_name, pp, good_pix_total, logLam, vel_comp0, z, 0, 
								bn, mad_std_residuals, SN_wMadStandardDev, fit_gas=False)


	return save_dict, optimal_templates_save, contsub_cube



def run_stellar_fit(runID, cube_path, prev_vmap_path=None, plot_every=0, individual_bin=None):
	#run stellar fit, optionally also fit gas or mask it out
	#runID - name identifier for the run (e.g. Nov25_1comp)
	#cube_path - path to data cube
	#vorbin_path - path to voronoi bin text file
	#prev_cmap_path - if supplied, uses initial fitting conditions

	gal_out_dir = 'ppxf_output/'

	# Reading in the cube
	hdu = fits.open(cube_path)
	cube = hdu[1].data
	h1 = hdu[1].header

	error_cube = hdu[2].data

	#ppxf parameters
	moments = 2
	adegree = 10     		# Additive polynomial used to correct the template continuum shape during the fit.
							# Degree = 10 is the suggested order for stellar populations.
	mdegree = 10			# Multiplicative polynomial used to correct the template continuum shape during the fit.
							# Mdegree = -1 tells fit to not include the multiplicative polynomial.

	wave_lam = np.array(h1['CRVAL3']+(np.arange(0, h1['NAXIS3'])*h1['CDELT3']))

	plot_name = f'stellarfit_{runID}'
	csv_save_name = f'{gal_out_dir}stellarfit_{runID}_nobin.csv'

	run_dict, opt_temps_save, contsub_cube = ppxf_fit_stellar(cube, error_cube, moments, adegree, mdegree, wave_lam,
						plot_every=plot_every, plot_name=plot_name, prev_vmap_path=prev_vmap_path, individual_bin=individual_bin)
	run_dict_df = pd.DataFrame.from_dict(run_dict)
	run_dict_df.to_csv(csv_save_name, index=False, header=True)

	print(f'Saved ppxf stellar fit csv to {csv_save_name}')

	opt_temps_save_path = csv_save_name[:-4]+'_templates.npy'
	np.save(opt_temps_save_path, opt_temps_save)

	print(f'Saved optimal templates to {opt_temps_save_path}')

	hdu[1].data = contsub_cube

	hdu.writeto(f'../ngc1266_data/NIFS_data/NGC1266_NIFS_cube_contsub_{runID}.fits', overwrite=True)




cube_path = "../ngc1266_data/NIFS_data/NGC1266_NIFS_cube_wcs_try2.fits"
run_id_stellar = 'Mar23'
prev_vmap_path = None

run_stellar_fit(run_id_stellar, cube_path, prev_vmap_path=prev_vmap_path, plot_every=100)



