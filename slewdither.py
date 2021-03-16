import os
from collections import OrderedDict

import numpy as np


class SlewDitherPattern:
    """creates a slew dither pattern for LMI observaitons"""
    def __init__(
            self, observation_list, dither_step_size=30, ra_min_offset=0, ra_max_offset=None,  dec_min_offset=0,
            dec_max_offset=None,
            max_offset=None
    ):
        """

        Parameters
        ----------
        observation_list : list of tuples
            each tuple should have the format:
                ('title':str, 'ra':str, 'dec':str, 'exposureTime':int or float, 'exposure_count': int, 'filter':str)
        dither_step_size : int
        ra_min_offset : int
        ra_max_offset : int
        dec_min_offset : int
        dec_max_offset : int
        max_offset : int
        """
        self.pattern_columns = [
            'title', 'ra', 'dec', 'exposureTime', 'numExposures', 'filter', 'dRA', 'dDec', 'commandOption'
        ]
        self.pattern = None
        self.observation_list = observation_list
        # columns=['title', 'ra', 'dec', 'exposureTime', 'exposure_count', 'filter']
        self.dither_step_size = dither_step_size
        self.ra_min_offset = ra_min_offset
        self.ra_max_offset = ra_max_offset
        self.dec_min_offset = dec_min_offset
        self.dec_max_offset = dec_max_offset
        self.max_offset = max_offset
        self.header = OrderedDict()
        self.header['title'] = True
        self.header['ra'] = True
        self.header['dec'] = True
        self.header['exposureTime'] = True
        self.header['numExposures'] = True
        self.header['filter'] = True
        self.header['subframe'] = False
        self.header['muRA'] = False
        self.header['muDec'] = False
        self.header['epoch'] = False
        self.header['dRA'] = True
        self.header['dDec'] = True
        self.header['rotatorPA'] = False
        self.header['rotatorFrame'] = False
        self.header['eta'] = False
        self.header['comment'] = False
        self.header['commandOption'] = True
        self.header_string = ''
        self.gen_header_string()
        self.gen_pattern()

    def gen_header_string(self):
        formatted_header = ['{}={}'.format(k, str(v).lower()) for k, v in self.header.items()]
        self.header_string = '#' + ' '.join(formatted_header) + '\n#\n'

    def gen_pattern(self):
        # 'title', 'ra', 'dec', 'exposureTime', 'numExposures', 'filter', 'dRA', 'dDec', 'commandOption'
        previous_ra = None
        previous_dec = None
        title_column = []
        ra_column = []
        dec_column = []
        exposure_time_column = []
        num_exposures_column = []
        filter_column = []
        dra_column = []
        ddec_column = []
        command_option_column = []

        for observation in self.observation_list:
            _title, _ra, _dec, _exposureTime, _exposure_count, _filter = observation
            title_column += ['"{}"'.format(_title) for i in range(0, _exposure_count)]
            ra_column += [_ra for i in range(0, _exposure_count)]
            dec_column += [_dec for i in range(0, _exposure_count)]
            exposure_time_column += [_exposureTime for i in range(0, _exposure_count)]
            num_exposures_column += [1 for i in range(0, _exposure_count)]
            filter_column += [_filter for i in range(0, _exposure_count)]

            dither_pattern = generate_dither_pattern(
                _exposure_count, self.dither_step_size, self.ra_min_offset, self.ra_max_offset, self.dec_min_offset,
                self.dec_max_offset, self.max_offset
            )

            _dra, _ddec = dither_pattern.transpose().tolist()
            dra_column += _dra
            ddec_column += _ddec

            if _ra == previous_ra and _dec == previous_dec:
                command_option_column += ['Dither' for i in range(0, _exposure_count)]
            else:
                command_option_column += ['Slew'] + ['Dither' for i in range(1, _exposure_count)]
            previous_ra = _ra
            previous_dec = _dec

        self.pattern = list(zip(
                title_column, ra_column, dec_column, exposure_time_column, num_exposures_column, filter_column,
                dra_column, ddec_column, command_option_column
            ))

    def save_pattern(self, output_file='pattern.txt'):
        lines = [' '.join([str(i) for i in line]) for line in self.pattern]
        _data = '\n'.join(lines)
        with open(output_file, 'w') as f2:
            f2.write(self.header_string + _data + '\n')


def generate_dither_pattern(
    exposure_count, dither_step,
    ra_min_offset=0, ra_max_offset=None,  dec_min_offset=0, dec_max_offset=None, max_offset=None
):
    assert isinstance(exposure_count, int)
    # assert isinstance(dither_step, float)
    grid_size = np.sqrt(exposure_count)
    ra_grid_size = np.int(grid_size)
    if grid_size.is_integer():
        dec_grid_size = ra_grid_size
    else:
        ra_grid_size = ra_grid_size + 1
        dec_grid_size = ra_grid_size
    dra_array = np.arange(ra_min_offset, ra_grid_size * dither_step + 1, dither_step)
    ddec_array = np.arange(dec_min_offset, dec_grid_size * dither_step + 1, dither_step)
    if ra_max_offset is not None:
        dra_array = dra_array[dra_array < ra_max_offset]
    if dec_max_offset is not None:
        ddec_array = ddec_array[ddec_array < dec_max_offset]
    offset_coords_first_quad = np.array(np.meshgrid(dra_array, ddec_array)).T.reshape(-1, 2)

    if max_offset is not None:
        distances = distance(offset_coords_first_quad[:, 0], offset_coords_first_quad[:, 1])
        offset_coords_first_quad = offset_coords_first_quad[distances < max_offset]

    offset_coords_second_quad = offset_coords_first_quad.copy()
    offset_coords_second_quad[:, 0] = -offset_coords_second_quad[:, 0]
    offset_coords_third_quad = offset_coords_first_quad.copy()
    offset_coords_third_quad = -offset_coords_third_quad
    offset_coords_fourth_quad = offset_coords_first_quad.copy()
    offset_coords_fourth_quad[:, 0] = -offset_coords_fourth_quad[:, 1]
    offset_coords_combined = np.concatenate(
        (offset_coords_first_quad, offset_coords_second_quad, offset_coords_third_quad, offset_coords_fourth_quad),
        axis=0
    )
    if offset_coords_combined.shape[0] < exposure_count:
        raise UserWarning(
            'Requested {} exposures, but only {} unique locations exist with current domain'.format(
                exposure_count, offset_coords_combined.shape[0]
            )
        )
    indices = np.arange(0, offset_coords_combined.shape[0])
    np.random.shuffle(indices)
    output_indices = indices[0:exposure_count]
    return offset_coords_combined[output_indices]


def distance(dra, ddec):
    return np.sqrt(dra ** 2 + ddec ** 2)


if __name__ == '__main__':
    obslists = [
        # [('FRB180301', '06:12:43.40', '+04:33:44.8', 25, 40, 'SL-r')],

        [('GRB110402A', '13:09:36.54', '+61:15:10.3', 150, 10, 'SL-i')],

        [('GRB151228', '14:16:03.76', '-17:39:56.0', 150, 10, 'SL-r')],

        [
            ("GRB160408A", '08:10:29.32', '71:07:40.5', 200, 10, 'SL-r'),
            ("GRB160408A", '08:10:29.32', '71:07:40.5', 200, 10, 'SL-g')
        ],

        [
            ("GRB160601A", '15:39:44.41', '+64:32:28.8', 150, 8, 'SL-i'),
            ("GRB160601A", '15:39:44.41', '+64:32:28.8', 100, 15, 'SL-z'),
            ("GRB160601A", '15:39:44.41', '+64:32:28.8', 150, 8, 'SL-g')
        ],

        [("GRB171007A", '09:02:24.13', '+42:49:08.9', 100, 20, 'SL-z')],

        [
            ("GRB180618A", '11:19:45.84', '+73:50:13.5', 100, 18, 'SL-z'),
            ("GRB180618A", '11:19:45.84', '+73:50:13.5', 100, 15, 'Yish')
        ],

        [
            ("GRB191031D", '18:53:09.57', '+47:38:38.5', 100, 5, 'SL-z'),
            ("GRB191031D", '18:53:09.57', '+47:38:38.5', 150, 3, 'SL-g'),
            ("GRB191031D", '18:53:09.57', '+47:38:38.5', 100, 6, 'Yish')],

        [("GRB200907B", '05:56:06.97', '+06:54:22.5', 150, 10, 'SL-r')]
    ]
    # observation_list, dither_step_size = 30, ra_min_offset = 0, ra_max_offset = None, dec_min_offset = 0,
    # dec_max_offset = None,
    # max_offset = None
    names = [
        'Mar15_Slew_dither_FRB180301_33min.txt',
        'Mar15_Slew_dither_GRB110402A_i_30min.txt',
        'Mar15_Slew_dither_GRB151128_30in.txt',
        'Mar15_Slew_dither_GRB160408A_75m+in.txt',
        'Mar15_Slew_dither_GRB160601A_izg_79min.txt',
        'Mar15_Slew_dither_GRB171007A_z.txt',
        'Mar15_Slew_dither_GRB180618A_zy_69min.txt',
        'Mar15_Slew_dither_GRB191031D_gzy_32min.txt',
        'Mar15_Slew_dither_GRB200907B_30in.txt'
    ]

    for name, obslist in zip(names, obslists):
        a = SlewDitherPattern(obslist, 30, 40, 131, 40, 131)
        a.save_pattern(name)
