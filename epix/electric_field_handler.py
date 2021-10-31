from scipy.interpolate import RegularGridInterpolator as RGI
import pandas as pd
import numpy as np
import os
import gzip
import json


class MyElectricFieldHandler:
    def __init__(self, field_map=""):
        """
        The field map, defined over a regular grid, should be
        a .csv or .json.gz file. 

        Structure of the csv file:
        Columns 'r', 'z' and 'E', with lenghts in cm and field in V/cm.
        The elements are delimited by a comma. 

        Structure of the json.gz file:
        Contains the 'r' and 'z' coordinates in cm under the key
          'coordinate_system' :   [['r', [r_min, r_max, n_r]],
                                  [['z', [z_min, z_max, n_z]]
        and the field values in V/cm under the key
          'map' : [value1, value2, value3, ...] with length equal to n_r * n_z        
        """
        self.map = field_map
        if os.path.isfile(self.map):
            self._load_field()
            self._get_coordinates()
            self._build_interpolator()
        else:
            raise ValueError(f'Cannot open "{self.map}". It is not a valid file'
                             ' for the electric field map.')

    def _load_field(self):
        file_ending = self.map.split('.')[-1]

        if file_ending == 'csv':
            _field = pd.read_csv(self.map)
            _field = pd.DataFrame(_field.groupby(['r']).aggregate({'z': list, 'E': list}))
            _field = _field.explode(['z', 'E'])
            _field = _field.reset_index()
            self.field = _field.applymap(float)
        elif file_ending == 'gz':
            with gzip.open(self.map, 'rb') as f:
                field_map = json.load(f)

            csys = field_map['coordinate_system']
            grid = [np.linspace(left, right, points)
                    for _, (left, right, points) in csys]
            csys = np.array(np.meshgrid(*grid, indexing='ij'))
            axes = np.roll(np.arange(len(grid) + 1), -1)
            csys = np.transpose(csys, axes)
            csys = np.array(csys).reshape((-1, len(grid)))

            self.field = pd.DataFrame()
            self.field["r"] = np.array(csys)[:, 0]
            self.field["z"] = np.array(csys)[:, 1]
            self.field["E"] = np.array(field_map['map'])
        else:
            raise ValueError(f'Cannot open "{self.map}". File extension is not valid'
                             ' for the electric field map. Use .csv or .json.gz')

    def _get_coordinates(self):
        self.R = np.unique(self.field['r'])
        self.Z = np.unique(self.field['z'])

    def _build_interpolator(self):
        e_tmp = np.reshape(np.array(self.field.E),
                           (len(self.R), len(self.Z)))
        self.interpolator = RGI([self.R, self.Z],
                                e_tmp,
                                bounds_error=False,
                                fill_value=None)

    def get_field(self, x, y, z, outside_map=np.nan):
        """
        Function which returns the electric field at a certain position
        according to an efield map.

        Args:
            x (np.array): x coordinate of the interaction in cm
            y (np.array): y coordinate of the interaction in cm
            z (np.array): z coordinate of the interaction in cm

        Kwargs:
            outside_map (float): Default value to be used if interaction
                was not within the range of the map. Default np.nan
        :return:
        """
        r = np.sqrt(x ** 2 + y ** 2)
        efield = self.interpolator((r, z))
        efield[np.isnan(efield)] = outside_map
        return efield
