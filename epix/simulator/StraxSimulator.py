import strax
import straxen
import wfsim.load_resource
import epix
import numpy as np
import pandas as pd
import os
from copy import deepcopy
from immutabledict import immutabledict
import inspect
import pickle
from .GenerateNveto import NVetoUtils
from .FastSimulator import FastSimulator
from .helpers import Helpers
import warnings

# We should think a bit more to detector_config_override
# and tell fast_sim to look into epix_args
# also because one entry of self.config is epix_config
@strax.takes_config(
    strax.Option('detector', default='XENONnT', help='Detector model'),
    strax.Option('detector_config_override', default='', help='For the electric field, otherwise 200 V/cm'),
    strax.Option('epix_config', default=dict(), help='Configuration file for epix', ),
    strax.Option('configuration_files', default=dict(), help='Files required for simulating'),
    strax.Option('fax_config', help='Fax configuration to load'),
    strax.Option('fax_config_overrides', help='Fax configuration to override', default=None),
    strax.Option('fax_config_override_from_cmt', default=None, infer_type=False,
                 help="Dictionary of fax parameter names (key) mapped to CMT config names (value) "
                      "where the fax parameter values will be replaced by CMT"),
    strax.Option('xy_resolution', default=0, help='xy position resolution (cm)'),
    strax.Option('z_resolution', default=0, help='xy position resolution (cm)'),
    strax.Option('nv_spe_resolution', default=0.40, help='nVeto SPE resolution'),
    strax.Option('nv_spe_res_threshold', default=0.50, help='nVeto SPE acceptance threshold'),
    strax.Option('nv_max_time_ns', default=1e7, help='nVeto maximum time for the acceptance of PMT hits in event'),
    strax.Option('s2_clustering_algorithm', default='mlp', help='clustering algorithm for S2, [ nsort | naive | mlp ]'),
)
class StraxSimulator(strax.Plugin):
    provides = ('events_tpc', 'events_nveto')
    depends_on = ()
    data_kind = immutabledict(zip(provides, provides))
    dtype = Helpers.get_dtype()

    def load_resources(self):
        """First load the config through wfsim, then we add some things we'd like to have"""
        self.resource = wfsim.load_resource.load_config(self.config)
        # add mean photon area distribution with avg values.
        self.resource.mean_photon_area_distribution = epix.Helpers.average_spe_distribution(
            self.resource.photon_area_distribution)
        self.resource.fdc_map = wfsim.load_resource.make_map(self.config['field_distortion_correction_map'],
                                                             fmt='json.gz')
        working_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        self.resource.nn_weights = pickle.load(open(f'{working_dir}/nn_weights.p', 'rb+'))
        if 'nv_pmt_qe' in self.config['configuration_files'].keys():
                    self.resource.nv_pmt_qe = straxen.get_resource(
                        self.config['configuration_files']['nv_pmt_qe'], fmt='json')
        else:
            warnings.warn('The configuration_files should not exist!'
                         'Everything should come by one config!'
                         'Since nv_pmt_qe config is missing in configuration_files, '
                         'the default one nveto_pmt_qe.json will be used')
            self.resource.nv_pmt_qe = straxen.get_resource('nveto_pmt_qe.json', fmt='json')

    def setup(self):
        print('Setup')
        overrides = self.config['fax_config_overrides']
        self.config.update(straxen.get_resource(self.config['fax_config'], fmt='json'))
        if overrides is not None:
            self.config.update(overrides)
        if self.config['fax_config_override_from_cmt'] is not None:
            for fax_field, cmt_option in self.config['fax_config_override_from_cmt'].items():
                if fax_field in ['fdc_3d', 's1_lce_correction_map'] and self.config.get('default_reconstruction_algorithm', False):
                    cmt_option = tuple(['suffix', self.config['default_reconstruction_algorithm'], *cmt_option])
                cmt_value = straxen.get_correction_from_cmt(self.run_id, cmt_option)
                self.config[fax_field] = cmt_value

        # TODO: this should not be needed here. Where is this info normally taken from?
        if 'gains' not in self.config.keys():
            self.config['gains'] = [1] * 494
        if 'n_top_pmts' not in self.config.keys():
            self.config['n_top_pmts'] = 253
        if 'n_tpc_pmts' not in self.config.keys():
            self.config['n_tpc_pmts'] = 494
        print('Loading resources')
        self.load_resources()

    def get_nveto_data(self, ):
        print('Getting nveto data')
        file_loader = epix.io.file_loader(directory=self.config['epix_config']['path'],
                                          file_name=self.config['epix_config']['file_name'])
        file_tree, _ = file_loader._get_ttree()

        if 'pmthitID' in file_tree.keys():
            nv_hits = NVetoUtils.get_nv_hits(ttree=file_tree,
                                             pmt_nv_json_dict=self.resource.nv_pmt_qe,
                                             nveto_dtype=self.dtype['events_nveto'],
                                             SPE_Resolution=self.config['nv_spe_resolution'],
                                             SPE_ResThreshold=self.config['nv_spe_res_threshold'],
                                             max_time_ns=self.config['nv_max_time_ns'],
                                             batch_size=10000)
            return nv_hits

    def save_epix_instruction(self, epix_ins, save_epix, config):
        file_name = config['epix_config']['file_name'].split('/')[-1]
        file_name = file_name.split('.')[0]
        file_name += '_instructions.csv'
        if isinstance(save_epix, str):
            # assume save epix as path to store
            epix_path = os.path.join(config['epix_config']['save_epix'], file_name)
        else:
            # if save epix True store in normal path
            epix_path = os.path.join(config['epix_config']['path'], file_name)
        print(f'Saving epix instruction: {epix_path}')
        df = pd.DataFrame(epix_ins)
        df.to_csv(epix_path, index=False)

    def get_epix_instructions(self, ):
        epix_config = deepcopy(self.config['epix_config'])
        fn = epix_config.get('file_name', '')
        if fn.endswith('.csv'):
            csv_file_path = os.path.join(epix_config['path'], epix_config['file_name'])
            print(f'Loading epix instructions from csv-file from {csv_file_path}')
            epix_ins = pd.read_csv(csv_file_path)
            epix_ins = np.array(epix_ins.to_records(index=False))
        elif fn.endswith('.root'):
            print('Generating epix instructions from root-file')

            detector = epix.init_detector(self.config['detector'].lower(), self.config['detector_config_override'])
            epix_config['detector_config'] = detector

            outer_cylinder = getattr(epix.detectors, 'xenonnt')
            _, outer_cylinder = outer_cylinder()
            epix_config['outer_cylinder'] = outer_cylinder
            epix_ins = epix.run_epix.main(epix_config, return_wfsim_instructions=True)
        else:
            print('No valid file_name! must be .root (Geant4 file) or .csv (epix instructions)')
            return
        return epix_ins

    def compute(self):
        print('Compute')
        #simulated_data_nveto = self.get_nveto_data()
        simulated_data_nveto = None
        self.epix_instructions = self.get_epix_instructions()
        save_epix = self.config['epix_config'].get('save_epix', False)
        if save_epix:
            self.save_epix_instruction(self.epix_instructions, save_epix, self.config)

        self.Simulator = FastSimulator(instructions_epix=self.epix_instructions,
                                   config=self.config,
                                   resource=self.resource)
        simulated_data = self.Simulator.run_simulator()

        simulated_data_chunk = self.chunk(
            start=simulated_data['time'][0],
            end=simulated_data['endtime'][-1],
            data=simulated_data,
            data_type='events_tpc')

        # write empty chunk if nveto data isn't there	        return {'events_tpc':simulated_data_chunk}
        if simulated_data_nveto is None or len(simulated_data_nveto['time']) < 1:
            simulated_data_nveto_chunk = self.chunk(
                start=0,
                end=1,
                data=simulated_data_nveto,
                data_type='events_nveto')
        else:
            simulated_data_nveto_chunk = self.chunk(
                start=np.floor(simulated_data_nveto['time'][0]).astype(np.int64),
                end=np.ceil(np.max(simulated_data_nveto['endtime'])).astype(np.int64),
                data=simulated_data_nveto,
                data_type='events_nveto')

        return {'events_tpc': simulated_data_chunk,
                'events_nveto': simulated_data_nveto_chunk}

    def is_ready(self, chunk):
        # For this plugin we'll smash everything into 1 chunk, should be oke
        return True if chunk == 0 else False

    def source_finished(self):
        return True