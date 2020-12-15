from scipy.interpolate import RegularGridInterpolator as RGI
import pandas as pd
import numpy as np
import os


class MyElectricFieldHandler:
    def __init__(self, field_map=""):
        """
        The field map should be a text file with the following
        structure: "r z E", with length in cm and field in V/cm.
        The elements are delimited by a white space. Header lines
        should start with a '#'.

        The map has to be defined over a regular grid and it has to
        be given as follows:
        0.0 -97.2 300
        0.1 -97.2 305
        0.2 -97.2 298
        ...
        """
        self.map = field_map
        if os.path.isfile(self.map):
            self._load_field()
            self._get_coordinates()
            self._build_interpolator()

    def _load_field(self):
        self.field = pd.read_csv(self.map,
                                 comment='#',
                                 header=None,
                                 delim_whitespace=True,
                                 names=['r', 'z', 'E'])

    def _get_coordinates(self):
        self.R = np.unique(self.field['r'])
        self.Z = np.unique(self.field['z'])

    def _build_interpolator(self):
        e_tmp = np.reshape(np.array(self.field.E),
                           (len(self.Z), len(self.R)))
        self.interpolator = RGI([self.Z, self.R],
                                e_tmp,
                                bounds_error=False,
                                fill_value=None)
    
    def get_field(self, x, y, z, outside_map=np.nan):
        """
        Function which returns the electric field at a certain position
        according to an efield map.

        Args:
            x (np.array): x coordinate of the interaction in cm
            y (np.array): x coordinate of the interaction in cm
            z (np.array): x coordinate of the interaction in cm

        Kwargs:
            outsie_map (float): Default value to be used if interaction
                was not within in the range of the map. Default np.nan
        :return:
        """
        r = np.sqrt(x**2+y**2)
        efield = self.interpolator((z, r))
        efield[np.isnan(efield)] = outside_map
        return efield
