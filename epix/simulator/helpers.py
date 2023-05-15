import wfsim
from scipy.interpolate import interp1d
import numpy as np
import numba


# Numba and classes still are not a match made in heaven
@numba.njit
def _merge_these_clusters_nsort(amp1, r1, z1, amp2, r2, z2, conf):
    sensitive_volume_ztop = 0  # it's the ground mesh, the top liquid level is at 2.7; // mm
    max_s2_area = max(amp1, amp2)
    if max_s2_area > 5000:
        SeparationDistanceIntercept = 0.00024787 * 5000. + 3.4056346550312973
        SeparationDistanceSlope = 5.5869678412887262e-07 * 5000. + 0.0044792968
    else:
        SeparationDistanceIntercept = \
            0.00024787 * max_s2_area + 3.4056346550312973
        SeparationDistanceSlope = \
            5.5869678412887262e-07 * max_s2_area + 0.0044792968
    SeparationDistance = \
        SeparationDistanceIntercept - \
        SeparationDistanceSlope * (-sensitive_volume_ztop + (z1 + z2) * 0.5)
    return z1 - z2 < SeparationDistance


@numba.njit
def _merge_these_clusters_nt_res_naive(amp1, r1, z1, amp2, r2, z2, conf):
    sensitive_volume_ztop = 0  # [cm]
    SeparationDistance = 1.6  # [cm], the worst case from [[weiss:analysis:he:zresoultion_zdependence]]
    return np.abs(z1 - z2) < SeparationDistance

@numba.njit
def get_nn_prediction(inp, w0, w1, w2, w3, w4, w5, w6, w7, w8, w9):
    y      = np.dot(inp, w0) + w1
    y[y<0] = 0
    y = np.dot(y, w2) + w3
    y[y < 0] = 0
    y = np.dot(y, w4) + w5
    y[y < 0] = 0
    y = np.dot(y, w6) + w7
    y[y < 0] = 0
    y = np.dot(y, w8) + w9
    y = 1 / (1+np.exp(-y))
    return y[0]


def _merge_these_clusters_nt_res_jaron(amp1, r1, z1, amp2, r2, z2, conf):
    dt_correction = 0.8988671508131871
    width_correction = 0.854918851953897
    fm = conf['field_map']
    dm = conf['diffusion_map']
    v1 = 0.1*fm([r1, z1], map_name='drift_speed_map')[0] # cm/us
    #print(f'{v1 =}')
    v2 = 0.1*fm([r2, z2], map_name='drift_speed_map')[0] # cm/us
    #print(f'{v2 =}')
    dt1 = -z1*dt_correction/v1+conf['dt_gate']/1000 # us
    #print(f'{dt1 =}')
    dt2 = -z2*dt_correction/v2+conf['dt_gate']/1000 # us
    #print(f'{dt2 =}')
    diff1 = 1e3*dm([r1,z1])[0] # cm2/us
    #print(f'{diff1 =}')
    diff2 = 1e3*dm([r2,z2])[0] # cm2/us
    #print(f'{diff2 =}')
    width1 = width_correction*1.348*np.sqrt(2*diff1*dt1/v1**2) # us
    #print(f'{w1 =}')
    width2 = width_correction*1.348*np.sqrt(2*diff2*dt2/v2**2) # us
    #print(f'{w2 =}')
    delta_t = (dt2-dt1) # us
    #print(f'{delta_t =}')
    split_param = delta_t/(width1+width2)
    #print(f'{split_param = }')
    survival1 = conf['field_map']([r1,z1], map_name='survival_probability_map')[0]
    survival2 = conf['field_map']([r2,z2], map_name='survival_probability_map')[0]
    e_lifetime = conf['e_lifetime'] / 1000 # us
    amp1_corr = int(conf['e_extraction_yield'] * survival1 * np.exp(-dt1/e_lifetime) * amp1)
    amp2_corr = int(conf['e_extraction_yield'] * survival2 * np.exp(-dt2/e_lifetime) * amp2)
    #print(f'{amp1_corr = }')
    #print(f'{amp2_corr = }')
    lower = min(amp1_corr, amp2_corr)
    higher = max(amp1_corr, amp2_corr)
    w0, w1, w2, w3, w4, w5, w6, w7, w8, w9 = conf['nn_weights']
    X = np.array((split_param, higher, lower), dtype=np.float32)
    y = get_nn_prediction(X, w0, w1, w2, w3, w4, w5, w6, w7, w8, w9)
    #print(f'{y = }')
    return y > np.random.random()


class Helpers:
    @staticmethod
    def assign_order(order):
        """
        This is a trick to have the calculation function be executed in a particular order
        """
        def do_assignment(to_func):
            to_func.order = order
            return to_func

        return do_assignment

    @staticmethod
    def average_spe_distribution(spe_shapes):
        """We take the spe distribution from all channels and take the average to be our spe distribution to draw photon areas from"""
        uniform_to_pe_arr = []
        for ch in spe_shapes.columns[1:]:  # skip the first element which is the 'charge' header
            if spe_shapes[ch].sum() > 0:
                # mean_spe = (spe_shapes['charge'].values * spe_shapes[ch]).sum() / spe_shapes[ch].sum()
                scaled_bins = spe_shapes['charge'].values  # / mean_spe
                cdf = np.cumsum(spe_shapes[ch]) / np.sum(spe_shapes[ch])
            else:
                # if sum is 0, just make some dummy axes to pass to interpolator
                cdf = np.linspace(0, 1, 10)
                scaled_bins = np.zeros_like(cdf)

            grid_cdf = np.linspace(0, 1, 2001)
            # For Inverse_transform_sampling methode
            grid_scale = interp1d(cdf, scaled_bins,
                                  bounds_error=False,
                                  fill_value=(scaled_bins[0], scaled_bins[-1]))(grid_cdf)

            uniform_to_pe_arr.append(grid_scale)
        spe_distribution = np.mean(uniform_to_pe_arr, axis=0)
        return spe_distribution

    @staticmethod
    def macro_cluster_events(nn_weights, instructions, config):
        """Loops over all instructions, checks if it's an S2 and if there is another s2 within the same event
            within the macro cluster distance, if it is they are merged."""

        print(f"\n macro_cluster_events --> s2_clustering_algorithm == {config['s2_clustering_algorithm']} . . .")

        merge_config = {}
        if config['s2_clustering_algorithm'] == 'bdt':
            _merge_clusters = _merge_these_clusters_nt_res_jaron
            merge_config['dt_gate'] = config['drift_time_gate']
            merge_config['e_lifetime'] = config['electron_lifetime_liquid']
            merge_config['e_extraction_yield'] = config['electron_extraction_yield']
            merge_config['field_map'] = wfsim.load_resource.make_map(config['field_dependencies_map'],
                                                     fmt='json.gz', method='WeightedNearestNeighbors')
            merge_config['diffusion_map'] = wfsim.load_resource.make_map(config['diffusion_longitudinal_map'],
                                                      fmt='json.gz', method='WeightedNearestNeighbors')
            merge_config['nn_weights'] = nn_weights
        elif config['s2_clustering_algorithm'] == 'naive':
            _merge_clusters = _merge_these_clusters_nt_res_naive
        elif config['s2_clustering_algorithm'] == 'nsort':
            _merge_clusters = _merge_these_clusters_nsort
        else:
            return
        for ix1, _ in enumerate(instructions):
            if instructions[ix1]['type'] != 2:
                continue
            #print(f"{ix1} is an S2 of event {instructions[ix1]['event_number']}. Other peaks in this event:")
            #print(instructions[instructions['event_number'] == instructions[ix1]['event_number']])
            for ix2 in range(1, len(instructions[ix1:])): # why was  + 1): ?
                #print(ix1 + ix2, end=' ')
                if instructions[ix1]['event_number'] != instructions[ix1 + ix2]['event_number']:
                    #print('belongs to another event.')
                    break
                if instructions[ix1 + ix2]['type'] != 2:
                    #print('isn\'t an S2')
                    continue
                #print(f'is an S2 of the same event. Check merge...')
                r1 = np.sqrt(instructions[ix1]['x'] ** 2 + instructions[ix1]['y'] ** 2)
                r2 = np.sqrt(instructions[ix1 + ix2]['x'] ** 2 + instructions[ix1 + ix2]['y'] ** 2)
                if _merge_clusters(instructions[ix1]['amp'], r1, instructions[ix1]['z'],
                                   instructions[ix1 + ix2]['amp'], r2, instructions[ix1 + ix2]['z'],
                                   merge_config):
                    #print(f'I will merge {ix1} and {ix1+ix2}.')
                    amp1 = instructions[ix1]['amp']
                    amp2 = instructions[ix1 + ix2]['amp']
                    amp_total = int((instructions[ix1]['amp'] + instructions[ix1 + ix2]['amp']))
                    instructions[ix1 + ix2]['x'] = (instructions[ix1]['x'] * amp1 + instructions[ix1 + ix2]['x'] * amp2) / amp_total
                    instructions[ix1 + ix2]['y'] = (instructions[ix1]['y'] * amp1+ instructions[ix1 + ix2]['y'] * amp2) / amp_total
                    instructions[ix1 + ix2]['z'] = (instructions[ix1]['z'] + instructions[ix1 + ix2]['z']) / 2

                    # primary position is one
                    instructions[ix1 + ix2]['x_pri'] = instructions[ix1]['x_pri']
                    instructions[ix1 + ix2]['y_pri'] = instructions[ix1]['y_pri']
                    instructions[ix1 + ix2]['z_pri'] = instructions[ix1]['z_pri']

                    instructions[ix1 + ix2]['amp'] = amp_total
                    instructions[ix1]['amp'] = -1  # flag to throw this instruction away later
                    instructions[ix1 + ix2]['e_dep'] = (instructions[ix1]['e_dep'] + instructions[ix1 + ix2]['e_dep'])
                    instructions[ix1]['e_dep'] = -1  # flag to throw this instruction away later
                    break
                #print(f'I won\'t merge {ix1} and {ix1+ix2}.')

    @staticmethod
    def get_s1_area_with_spe(spe_distribution, num_photons):
        """
            :params: spe_distribution, the spe distribution to draw photon areas from
            :params: num_photons, number of photons to draw from spe distribution
        """
        s1_area_spe = []
        for n_ph in num_photons:
            s1_area_spe.append(np.sum(spe_distribution[
                                          (np.random.random(n_ph) * len(spe_distribution)).astype(np.int64)]))

        return np.array(s1_area_spe)

    @staticmethod
    def get_s1_light_yield(n_photons, positions, s1_lce_map, config):
        """See WFsim.s1.get_n_photons"""
        return wfsim.S1.get_n_photons(n_photons=n_photons,
                                      positions=positions,
                                      s1_lce_correction_map=s1_lce_map,
                                      config=config)

    @staticmethod
    def get_s2_light_yield(positions, config, resource):
        """See WFsim.s2.get_s2_light_yield"""
        return wfsim.S2.get_s2_light_yield(positions=positions,
                                           config=config,
                                           resource=resource)

    @staticmethod
    def get_s2_charge_yield(n_electron, xy, z, config, resource):
        """See wfsim.s2.get_electron_yield"""
        return wfsim.S2.get_electron_yield(n_electron=n_electron,
                                           xy_int=xy,
                                           z_int=z,
                                           config=config,
                                           resource=resource)

    @staticmethod
    def get_drift_time(z, xy, config, resource):
        """See wfsim.S2.get_s2_drift_time_params"""
        dt_mean, _ = wfsim.S2.get_s2_drift_time_params(z, xy, config, resource)
        return dt_mean
