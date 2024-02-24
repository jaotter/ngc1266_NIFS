from astropy.table import Table

import numpy as np


line_dict = {'H2(1-0)S(2)':2.0332, 'H2(2-1)S(3)':2.0735, 'H2(1-0)S(1)':2.1213, 'H2(2-1)S(2)':2.1542, 'Brgamma':2.1654, 'H2(1-0)S(0)':2.2230, 'H2(2-1)S(1)':2.2470, 'H2(1-0)Q(1)':2.4066}


table_dir = '/Users/jotter/highres_PSBs/ngc1266_NIFS/fit_output/'
tables = ['NIFS_50pc_fluxes_cent_mask.csv', 'NIFS_50pc_fluxes_x22y42_mask.csv', 'NIFS_50pc_fluxes_x62y22_mask.csv', 'NIFS_fov_spectrum_fluxes.csv']
table_labels = ['Center', 'West', 'East', 'FOV']

table_rows = []
for tab_name in tables:
	tab = Table.read(table_dir + tab_name)
	row = tab[0]
	table_rows.append(row)

name_col = []
wave_col = []
cent_col = []
west_col = []
east_col = []
fov_col = []

flux_cols = [cent_col, west_col, east_col, fov_col]
rnd_num_list = [1,2,2,0]

for line_name in line_dict.keys():
	name_col.append(line_name)
	wave_col.append(line_dict[line_name])
	for i, tab_row in enumerate(table_rows):

		flux = tab_row[line_name+'_flux'] * 1e16
		flux_err = tab_row[line_name+'_flux_err'] * 1e16

		if flux_err * 3 > flux:
			upper_lim_rnd = np.round(flux_err * 3, rnd_num)
			flux_str = f'<{upper_lim_rnd}'

		else:
			rnd_num = rnd_num_list[i]
			flux_err_rnd = np.round(flux_err, rnd_num)
			flux_rnd = np.round(flux, rnd_num)
			flux_str = f'{flux_rnd} $\pm$ {flux_err_rnd}'

		col_list = flux_cols[i]
		col_list.append(flux_str)

newtable = Table((name_col, wave_col, cent_col, west_col, east_col, fov_col),
				names=('Line', 'Wavelength', 'Center Flux', 'West Flux', 'East Flux', 'FOV Flux'))

newtable.write('/Users/jotter/highres_PSBs/ngc1266_NIFS/flux_table_paper.txt', format='latex', overwrite=True)