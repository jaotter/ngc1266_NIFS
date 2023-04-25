from astropy.io import fits
from scipy.optimize import curve_fit
from scipy import odr
from astropy.table import Table
from astropy.wcs import WCS

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const
import pandas as pd
import os

#########
# Script to fit NIFS lines with gaussian

line_dict = {'H2(1-0)S(2)':2.0332, 'H2(1-0)S(1)':2.1213, 'H2(1-0)S(0)':2.2230, 'H2(2-1)S(1)':2.2470, 'H2(2-1)S(2)':2.1542, 'Brgamma':2.1654}#, 'CIV':2.0780}
#lines to make maps for, leaving off CIV for now (need to find reference and also idk what to use it for)

z = 0.007214         # NGC 1266 redshift, from SIMBAD
galv = np.log(z+1)*const.c.to(u.km/u.s) # estimate of galaxy's velocity

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


def fit_continuum(spectrum, wave_vel, line_name, degree=1):
	## function to do continuum fit for a given line, with hand-selected continuum bands for each line

	line_vel = wave_vel - galv.value

	#chosen by hand for H2_212
	lower_band_kms = [-1700,-1000]
	upper_band_kms = [1000,1700]

	lower_ind1 = np.where(line_vel > lower_band_kms[0])
	lower_ind2 = np.where(line_vel < lower_band_kms[1])
	lower_ind = np.intersect1d(lower_ind1, lower_ind2)

	upper_ind1 = np.where(line_vel > upper_band_kms[0])
	upper_ind2 = np.where(line_vel < upper_band_kms[1])
	upper_ind = np.intersect1d(upper_ind1, upper_ind2)

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

	return cont_fit, residuals_std


def fit_line(spectrum_fit, wave_vel_fit, spectrum_err_fit, cont_fit, line_name, start, bounds, bin_num, runID, plot=False):
	## function to fit line to continuum-subtracted spectrum, with start values, bounds, and varying number of gaussian components

	contsub_spec = spectrum_fit - cont_fit
	#spectrum_err_fit = None

	popt, pcov = curve_fit(gauss_sum, wave_vel_fit, contsub_spec, sigma=spectrum_err_fit, p0=start, bounds=bounds, absolute_sigma=True, maxfev=5000)

	print(f'{line_name} fitted for bin {bin_num}.')

	if plot == True:
		fig = plt.figure(figsize=(6,8))
		
		gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=(3,1), hspace=0)
		ax0 = fig.add_subplot(gs[0,0])
		ax1 = fig.add_subplot(gs[1,0], sharex=ax0)

		ax0.plot(wave_vel_fit-galv.value, spectrum_fit, color='tab:blue', linestyle='-', marker='.', label='Data')
		ax0.plot(wave_vel_fit-galv.value, gauss_sum(wave_vel_fit, *popt[0:3]) + cont_fit, linestyle='--', color='tab:purple', label='Gaussian Component 1')
		ax0.plot(wave_vel_fit-galv.value, gauss_sum(wave_vel_fit, *popt[3:6]) + cont_fit, linestyle='-', color='tab:purple', label='Gaussian Component 2')
		ax0.plot(wave_vel_fit-galv.value, cont_fit, linestyle='-', color='tab:orange', label='Continuum')
		ax0.plot(wave_vel_fit-galv.value, cont_fit + gauss_sum(wave_vel_fit, *popt), linestyle='-', color='k', label='Full Fit')

		ax0.legend()

		residuals = spectrum_fit - gauss_sum(wave_vel_fit, *popt) - cont_fit

		ax1.plot(wave_vel_fit-galv.value, residuals, color='tab:red', marker='p', linestyle='-')
		ax1.axhline(0)

		ax0.set_ylabel('Flux (erg/s/cm^2/A)')
		ax1.set_ylabel('Residuals (erg/s/cm^2/A)')

		ax1.set_xlabel('Velocity (km/s)')

		ax0.set_title(f'{line_name} fit for bin {bin_num}' )
		plt.subplots_adjust(hspace=0)

		savename = f'{line_name}_{bin_num}_fit.png'
		savepath = f'/Users/jotter/highres_PSBs/ngc1266_NIFS/plots/gaussfit/{runID}/'
		if os.path.exists(savepath) == False:
			os.mkdir(savepath)
		plt.savefig(f'{savepath}{savename}')
		print(f'Figure saved as {savename}')

		plt.close()

	return popt, pcov

def fit_cube(runID, cube_path='/Users/jotter/highres_PSBs/ngc1266_data/NIFS_data/NGC1266_NIFS_cube_wcs_try2.fits', ncomp=2):
	## fit the NIFS cube, fitting each line with the desired number of components
	# loop through starting at the center and spiral outwards, with the previous bin acting as the the starting parameters for the next
	
	hdu = fits.open(cube_path)
	cube = hdu[1].data
	h1 = hdu[1].header

	error_cube = hdu[2].data

	obs_wave = np.array(h1['CRVAL3']+(np.arange(0, h1['NAXIS3'])*h1['CDELT3'])) * u.angstrom


	#assigning each pixel a bin number
	binNum = np.reshape(np.arange(cube.shape[1]*cube.shape[2]), (cube.shape[1], cube.shape[2]))
	x,y = np.meshgrid(np.arange(cube.shape[2]), np.arange(cube.shape[1]))

	#write code to automate this for all lines in line_list
	save_dict = {'bin_num':[], 'x':[], 'y':[]}
	for line_name in line_dict.keys():
		save_dict[f'{line_name}_flux'] = []
		save_dict[f'{line_name}_flux_err'] = []


	prev_fit_params = None

	loop_list = binNum.flatten()

	loop_list = spiral_traverse(binNum)

	for bn in loop_list: 

		print('\n============================================================================')
		print('binNum: {}'.format(bn))

		if bn % 10 == 0:
			plot_bool = True
		else:
			plot_bool = False

		b_loc = np.where(binNum == bn)
		x_loc = x[b_loc]
		y_loc = y[b_loc]

		spectrum = cube[:,y_loc,x_loc]
		spectrum_err = np.sqrt(np.abs(error_cube[:,y_loc,x_loc])) #sqrt bc this cube is the variance

		if len(spectrum[np.isnan(spectrum)]) > 0.3 * len(spectrum) or len(spectrum[spectrum==0]) > 0.3 * len(spectrum):
			continue

		save_dict['bin_num'].append(bn)
		save_dict['x'].append(x_loc[0])
		save_dict['y'].append(y_loc[0])

		line_name = 'H2(2-1)S(1)'
		line_wave = line_dict[line_name] * u.micron

		wave_to_vel = u.doppler_optical(line_wave)

		wave_vel = obs_wave.to(u.km/u.s, equivalencies=wave_to_vel)
		fit_ind1 = np.where(wave_vel > galv-2000* u.km/u.s)
		fit_ind2 = np.where(wave_vel < galv+2000* u.km/u.s)
		fit_ind = np.intersect1d(fit_ind1, fit_ind2)

		line_spec = spectrum[fit_ind].squeeze() * 1e-16
		line_vel = wave_vel[fit_ind].squeeze()
		line_spec_err = np.sqrt(np.abs(spectrum_err[fit_ind].squeeze())) * 1e-16

		peak_flux = np.nanmax(line_spec)

		if prev_fit_params is None:
			#initial guess if of 2 components, one wider with 1/3 the flux
			start = [galv.value, peak_flux, 200, galv.value, peak_flux*0.25, 400]
			#format is ([lower bounds], [upper bounds])
			bounds = ([galv.value - 500, peak_flux * 0.1, 20, galv.value - 500, 0, 50], [galv.value + 500, peak_flux*10, 600, galv.value + 500, peak_flux*10, 600])

		else:
			start = [prev_fit_params[0], peak_flux, prev_fit_params[2], prev_fit_params[3], peak_flux*0.25, prev_fit_params[5]]
			#currently these bounds are not very limiting
			width_upper_1 = np.min((prev_fit_params[2]*2, 600))
			width_upper_2 = np.min((prev_fit_params[5]*2, 600))
			width_lower_1 = np.max((prev_fit_params[2]*0.5, 20))
			width_lower_2 = np.max((prev_fit_params[5]*0.5, 20))
			bounds = ([prev_fit_params[0]-500, peak_flux*0.01, width_lower_1, prev_fit_params[3]-500, peak_flux*0.01, width_lower_2],
				[prev_fit_params[0]+500, peak_flux*10, width_upper_1, prev_fit_params[3]+500, peak_flux*10, width_upper_2])

		line_spec_err[line_spec_err == 0] = np.nanmedian(line_spec_err)

		cont_fit, cont_std = fit_continuum(line_spec, line_vel.value, line_name, degree=1)


		popt, pcov = fit_line(line_spec, line_vel.value, line_spec_err, cont_fit, line_name, start, bounds, bn, runID, plot=plot_bool)

		total_flux = np.sum(gauss_sum(line_vel.value, *popt))

		save_dict[f'{line_name}_flux'].append(total_flux)
		save_dict[f'{line_name}_flux_err'].append(np.nan)

		prev_fit_params = popt

		for line_name in line_dict.keys():
			if line_name == 'H2(2-1)S(1)':
				continue

			line_wave = line_dict[line_name] * u.micron

			wave_to_vel = u.doppler_optical(line_wave)

			wave_vel = obs_wave.to(u.km/u.s, equivalencies=wave_to_vel)
			fit_ind1 = np.where(wave_vel > galv-2000* u.km/u.s)
			fit_ind2 = np.where(wave_vel < galv+2000* u.km/u.s)
			fit_ind = np.intersect1d(fit_ind1, fit_ind2)

			line_spec = spectrum[fit_ind].squeeze()
			line_vel = wave_vel[fit_ind].squeeze()
			line_spec_err = np.sqrt(np.abs(spectrum_err[fit_ind].squeeze()))

			#perform linear continuum subtraction
			cont_fit, cont_std = fit_continuum(line_spec, line_vel.value, line_name, degree=1)

			contsub_spec = line_spec - cont_fit
			peak_flux = np.nanmax(contsub_spec)
			
			if peak_flux <= 3*cont_std + 1e-4: #figure out upper limits later
				save_dict[f'{line_name}_flux'].append(np.nan)
				save_dict[f'{line_name}_flux_err'].append(np.nan)

				continue

			#use H2_212 fit for initial conditions
			#initial guess if of 2 components, one wider with 1/3 the flux
			start = [prev_fit_params[0], peak_flux, prev_fit_params[2], prev_fit_params[3], peak_flux, prev_fit_params[5]]
			#format is ([lower bounds], [upper bounds])
			width_upper_1 = np.min((prev_fit_params[2]*2, 600))
			width_upper_2 = np.min((prev_fit_params[5]*2, 600))
			width_lower_1 = np.max((prev_fit_params[2]*0.5, 20))
			width_lower_2 = np.max((prev_fit_params[5]*0.5, 20))
			bounds = ([prev_fit_params[0]-600, peak_flux * 0.01, width_lower_1, prev_fit_params[3]-600, 0, width_lower_2], 
						[prev_fit_params[0]+600, peak_flux*10, width_upper_1, prev_fit_params[3]+600, peak_flux*10, width_upper_2])

			line_spec_err[line_spec_err == 0] = np.nanmedian(line_spec_err)
			popt, pcov = fit_line(line_spec, line_vel.value, line_spec_err, cont_fit, line_name, start, bounds, bn, runID, plot=plot_bool)

			width1 = popt[2] * u.km/u.s
			amp1 = popt[1]
			wave_width1 = ((width1 / const.c) * line_wave).to(u.angstrom).value
			width2 = popt[5] * u.km/u.s
			amp2 = popt[4]
			wave_width2 = ((width2 / const.c) * line_wave).to(u.angstrom).value

			amp1_err = np.sqrt(pcov[1,1])
			width1_err = np.sqrt(pcov[2,2]) * u.km/u.s
			wave_width1_err = ((width1_err / const.c) * line_wave).to(u.angstrom).value
			amp2_err = np.sqrt(pcov[4,4])
			width2_err = np.sqrt(pcov[5,5]) * u.km/u.s
			wave_width2_err = ((width2_err / const.c) * line_wave).to(u.angstrom).value

			total_flux = np.sqrt(2 * np.pi) * ((wave_width2 * amp1) + (wave_width2 * amp2))

			total_flux_err = np.sqrt(2*np.pi) * np.sqrt((wave_width1*amp1_err)**2 + (amp1*wave_width1_err)**2 + (wave_width2*amp2_err)**2 + (amp2*wave_width2_err)**2)

			#total_flux_old = np.sum(gauss_sum(line_vel.value, *popt))

			save_dict[f'{line_name}_flux'].append(total_flux)
			save_dict[f'{line_name}_flux_err'].append(total_flux_err)


	csv_save_name = f'/Users/jotter/highres_PSBs/ngc1266_NIFS/fit_output/{runID}_gaussfit.csv'

	run_dict_df = pd.DataFrame.from_dict(save_dict)
	run_dict_df.to_csv(csv_save_name, index=False, header=True)

	print(f'Saved emission line fit csv to {csv_save_name}')


def csv_to_maps(csv_path, save_name):


	fit_tab = Table.read(csv_path, format='csv')


	cube_file = '/Users/jotter/highres_PSBs/ngc1266_data/NIFS_data/NGC1266_NIFS_cube_wcs_try2.fits'
	cube_fl = fits.open(cube_file)
	cube_header = cube_fl[1].header
	cube_wcs = WCS(cube_header).celestial
	cube_data = cube_fl[1].data
	cube_fl.close()

	map_shape = cube_data.shape[1:3]

	binNum_2D = np.full(map_shape, np.nan)

	n_maps = len(line_dict.keys())*2
	map_cube = np.full((n_maps,map_shape[0],map_shape[1]), np.nan)
	#H2_212_flux = np.full(map_shape, np.nan)
	#H2_212_flux_err = np.full(map_shape, np.nan)

	for ind, bn in enumerate(fit_tab['bin_num']):
		x_loc = fit_tab['x'][ind]
		y_loc = fit_tab['y'][ind]

		binNum_2D[y_loc, x_loc] = bn

		for map_ind, line_name in enumerate(line_dict.keys()):

			line_flux = fit_tab[f'{line_name}_flux'][ind]
			line_flux_err = fit_tab[f'{line_name}_flux_err'][ind]

			map_cube[map_ind*2, y_loc, x_loc] = line_flux
			map_cube[map_ind*2+1, y_loc, x_loc] = line_flux_err

	maps_header = cube_wcs.to_header()

	for map_ind, line_name in enumerate(line_dict.keys()):

		maps_header[f'DESC{map_ind*2}'] = f'{line_name}_flux'
		maps_header[f'DESC{map_ind*2+1}'] = f'{line_name}_flux_err'

	maps_header['FLUX_UNIT'] = 'erg/s/cm**2'

	new_hdu = fits.PrimaryHDU(map_cube, header=maps_header)
	hdulist = fits.HDUList([new_hdu])

	savefile = f'/Users/jotter/highres_PSBs/ngc1266_NIFS/fit_output/{save_name}.fits'

	hdulist.writeto(savefile, overwrite=True)

	print(f'saved to {savefile}')


fit_cube(runID='run3')

csv_path = '/Users/jotter/highres_PSBs/ngc1266_NIFS/fit_output/run3_gaussfit.csv'
save_name = 'run3_gaussfit_maps'

csv_to_maps(csv_path, save_name)

