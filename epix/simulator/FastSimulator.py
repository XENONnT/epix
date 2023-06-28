import time
import numpy as np
import pandas as pd
import os
from .GenerateEvents import GenerateEvents
from .helpers import Helpers

# 2023-02-19: configuration_files:
#   'nv_pmt_qe':'nveto_pmt_qe.json',
#   'photon_area_distribution':'XENONnT_SR0_spe_distributions_20210713_no_noise_scaled.csv',
#   's1_pattern_map': 'XENONnT_s1_xyz_patterns_LCE_MCvf051911_wires.pkl',
#   's2_pattern_map': 'XENONnT_s2_xy_patterns_GXe_LCE_corrected_qes_MCv4.3.0_wires.pkl',

def monitor_time(prev_time, task):
    t = time.time()
    print(f'It took {(t - prev_time):.4f} sec to {task}')
    return t


class FastSimulator:
    """Simulator class for epix to go from  epix instructions to fully processed data"""

    def __init__(self, instructions_epix, config, resource):
        self.config = config
        self.ge = GenerateEvents()
        self.resource = resource
        self.simulation_functions = sorted(
            # get the functions of GenerateEvents in order to run through them all
            [getattr(self.ge, field) for field in dir(self.ge)
             if hasattr(getattr(self.ge, field), "order")
             ],
            # sort them by their order
            key=(lambda field: field.order)
        )
        self.instructions_epix = instructions_epix

    def cluster_events(self, ):
        """Events have more than 1 s1/s2. Here we throw away all of them except the largest 2
         Take the position to be that of the main s2
         And strax wants some start and endtime, so we make something up
        
        First do the macro clustering. Clustered instructions will be flagged with amp=-1,
        so we can safely throw those out"""
        start_time = time.time()

        Helpers.macro_cluster_events(self.instructions_epix, self.config, self.resource)
        self.instructions_epix = self.instructions_epix[self.instructions_epix['amp'] != -1]

        event_numbers = np.unique(self.instructions_epix['event_number'])
        ins_size = len(event_numbers)
        instructions = np.zeros(ins_size, dtype=Helpers.get_dtype()['events_tpc'])

        for ix in range(ins_size):
            i = instructions[ix]
            inst = self.instructions_epix[self.instructions_epix['event_number'] == event_numbers[ix]]
            inst_s1 = inst[inst['type'] == 1]
            inst_s2 = inst[inst['type'] == 2]

            s1 = np.argsort(inst_s1['amp'])
            s2 = np.argsort(inst_s2['amp'])

            if len(s1) < 1 or len(s2) < 1:
                continue

            i['s1_area'] = np.sum(inst_s1['amp'])

            i['s2_area'] = inst_s2[s2[-1]]['amp']
            i['e_dep'] = inst_s2[s2[-1]]['e_dep']

            if len(s2) > 1:
                i['alt_s2_area'] = inst_s2[s2[-2]]['amp']
                i['alt_e_dep'] = inst_s2[s2[-2]]['e_dep']
                i['alt_s2_x_true'] = inst_s2[s2[-2]]['x']
                i['alt_s2_y_true'] = inst_s2[s2[-2]]['y']
                i['alt_s2_z_true'] = inst_s2[s2[-2]]['z']

            i['x_true'] = inst_s2[s2[-1]]['x']
            i['y_true'] = inst_s2[s2[-1]]['y']
            i['z_true'] = inst_s2[s2[-1]]['z']

            i['x_pri'] = inst_s2[s2[-1]]['x_pri']
            i['y_pri'] = inst_s2[s2[-1]]['y_pri']
            i['z_pri'] = inst_s2[s2[-1]]['z_pri']

            i['g4id'] = inst_s2[s2[-1]]['g4id']

            # Strax wants begin and endtimes
            i['time'] = ix * 1000
            i['endtime'] = ix * 1000 + 1

        instructions = instructions[~((instructions['time'] == 0) & (instructions['endtime'] == 0))]
        self.instructions = instructions
        print(f'It took {(time.time() - start_time):.4f} sec to macro cluster events')

    def simulate(self, ):
        for func in self.simulation_functions:
            start = time.time()
            func(i=self.instructions,
                 config=self.config,
                 resource=self.resource)
            monitor_time(start, func.__name__)

    def run_simulator(self, ):
        self.cluster_events()
        # TODO this for debug purposes - not super nice. We should have a fastsim_truth data type
        file_name = self.config['epix_config']['file_name'].split('/')[-1]
        file_name = file_name.split('.')[0]
        file_name += '_instruction_after_macro_clustering.csv'
        save_epix = self.config['epix_config'].get('save_epix', False)
        if save_epix:
            if isinstance(save_epix, str):
                # assume save epix as path to store
                epix_path = os.path.join(self.config['epix_config']['save_epix'], file_name)
            else:
                # if save epix True store in normal path
                epix_path = os.path.join(self.config['epix_config']['path'], file_name)
            print(f'Saving epix instruction after macro clustering: {epix_path}')
            df = pd.DataFrame(self.instructions)
            df.to_csv(epix_path, index=False)
        self.simulate()
        return self.instructions




