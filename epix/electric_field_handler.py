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
        Columns "r" "z" and "E", with lenght in cm and field in V/cm.
        The elements are delimited by a ",". 

        Structure of the json.gz file:
        Contains the "r" and "z" coordinates in cm under the key
        "coordinate_system" and the field in V/cm under the key "map".
        """
        self.map = field_map
        if os.path.isfile(self.map):
            self._load_field()
            self._get_coordinates()
            self._build_interpolator()
        else:
            raise ValueError(f'Cannot open "{self.map}". It is not a valid file'
                             ' for the electirc field map.')

    def _load_field(self):
        file_ending = self.map.split(".")[-1]

        if file_ending == "csv":
            self.field = pd.read_csv(self.map)
        elif file_ending == "gz":
            with gzip.open(self.map, 'rb') as f:
                field_map = json.load(f)
            self.field = pd.DataFrame()
            self.field["r"] = np.array(field_map["coordinate_system"])[:,0]
            self.field["z"] = np.array(field_map["coordinate_system"])[:,1]
            self.field["E"] = np.array(field_map["map"])
        else:
            raise ValueError(f'Cannot open "{self.map}". File extension is not valid'
                             ' for the electric field map. Use .csv or .json.gz')

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
            y (np.array): y coordinate of the interaction in cm
            z (np.array): z coordinate of the interaction in cm

        Kwargs:
            outside_map (float): Default value to be used if interaction
                was not within the range of the map. Default np.nan
        :return:
        """
        r = np.sqrt(x**2+y**2)
        efield = self.interpolator((z, r))
        efield[np.isnan(efield)] = outside_map
        return efield
