import numpy as np
import nestpy
import pickle


class BBF_quanta_generator:
    def __init__(self):
        self.er_par_dict = {
            'W': 0.013509665661431896,
            'Nex/Ni': 0.08237994367314523,
            'py0': 0.12644250072199228,
            'py1': 43.12392476032283,
            'py2': -0.30564651066249543,
            'py3': 0.937555814189728,
            'py4': 0.5864910020458629,
            'rf0': 0.029414125811261564,
            'rf1': 0.2571929264699089,
            'fano': 0.059
        }
        self.nr_par_dict = {
            "W": 0.01374615297291325,
            "alpha": 0.9376149722771664,
            "zeta": 0.0472,
            "beta": 311.86846286764376,
            "gamma": 0.015772527423653895,
            "delta": 0.0620,
            "kappa": 0.13762801393921467,
            "eta": 6.387273512457444,
            "lambda": 1.4102590741165675,
            "fano": 0.059
        }
        self.ERs = [7, 8, 11]
        self.NRs = [0, 1]
        self.unknown = [12]
        self.get_quanta_vectorized = np.vectorize(self.get_quanta, excluded="self")

    def update_ER_params(self, new_params):
        self.er_par_dict.update(new_params)

    def update_NR_params(self, new_params):
        self.nr_par_dict.update(new_params)

    def get_quanta(self, interaction, energy, field):
        if int(interaction) in self.ERs:
            return self.get_ER_quanta(energy, field, self.er_par_dict)
        elif int(interaction) in self.NRs:
            return self.get_NR_quanta(energy, field, self.nr_par_dict)
        elif int(interaction) in self.unknown:
            return 0, 0, 0
        else:
            raise RuntimeError("Unknown nest ID: {:d}, {:s}".format(
                int(interaction),
                str(nestpy.INTERACTION_TYPE(int(interaction)))))

    ####
    def ER_recomb(self, energy, field, par_dict):
        W = par_dict['W']
        ExIonRatio = par_dict['Nex/Ni']

        Nq = energy / W
        Ni = Nq / (1. + ExIonRatio)
        Nex = Nq - Ni

        TI = par_dict['py0'] * np.exp(-energy / par_dict['py1']) * field ** par_dict['py2']
        Recomb = 1. - np.log(1. + TI * Ni / 4.) / (TI * Ni / 4.)
        FD = 1. / (1. + np.exp(-(energy - par_dict['py3']) / par_dict['py4']))

        return Recomb * FD

    def ER_drecomb(self, energy, par_dict):
        return par_dict['rf0'] * (1. - np.exp(-energy / par_dict['py1']))

    def NR_quenching(self, energy, par_dict):
        alpha = par_dict['alpha']
        beta = par_dict['beta']
        gamma = par_dict['gamma']
        delta = par_dict['delta']
        kappa = par_dict['kappa']
        eta = par_dict['eta']
        lam = par_dict['lambda']
        zeta = par_dict['zeta']

        e = 11.5 * energy * 54. ** (-7. / 3.)
        g = 3. * e ** 0.15 + 0.7 * e ** 0.6 + e

        return kappa * g / (1. + kappa * g)

    def NR_ExIonRatio(self, energy, field, par_dict):
        alpha = par_dict['alpha']
        beta = par_dict['beta']
        gamma = par_dict['gamma']
        delta = par_dict['delta']
        kappa = par_dict['kappa']
        eta = par_dict['eta']
        lam = par_dict['lambda']
        zeta = par_dict['zeta']

        e = 11.5 * energy * 54. ** (-7. / 3.)

        return alpha * field ** (-zeta) * (1. - np.exp(-beta * e))

    def NR_Penning_quenching(self, energy, par_dict):
        alpha = par_dict['alpha']
        beta = par_dict['beta']
        gamma = par_dict['gamma']
        delta = par_dict['delta']
        kappa = par_dict['kappa']
        eta = par_dict['eta']
        lam = par_dict['lambda']
        zeta = par_dict['zeta']

        e = 11.5 * energy * 54. ** (-7. / 3.)
        g = 3. * e ** 0.15 + 0.7 * e ** 0.6 + e

        return 1. / (1. + eta * e ** lam)

    def NR_recomb(self, energy, field, par_dict):
        alpha = par_dict['alpha']
        beta = par_dict['beta']
        gamma = par_dict['gamma']
        delta = par_dict['delta']
        kappa = par_dict['kappa']
        eta = par_dict['eta']
        lam = par_dict['lambda']
        zeta = par_dict['zeta']

        e = 11.5 * energy * 54. ** (-7. / 3.)
        g = 3. * e ** 0.15 + 0.7 * e ** 0.6 + e

        HeatQuenching = self.NR_quenching(energy, par_dict)
        PenningQuenching = self.NR_Penning_quenching(energy, par_dict)

        ExIonRatio = self.NR_ExIonRatio(energy, field, par_dict)

        xi = gamma * field ** (-delta)
        Nq = energy * HeatQuenching / par_dict['W']
        Ni = Nq / (1. + ExIonRatio)

        return 1. - np.log(1. + Ni * xi) / (Ni * xi)

    ###
    def get_ER_quanta(self, energy, field, par_dict):
        Nq_mean = energy / par_dict['W']
        Nq = np.clip(np.round(np.random.normal(Nq_mean, np.sqrt(Nq_mean * par_dict['fano']))), 0, np.inf).astype(
            np.int64)

        Ni = np.random.binomial(Nq, 1. / (1. + par_dict['Nex/Ni']))

        recomb = self.ER_recomb(energy, field, par_dict)
        drecomb = self.ER_drecomb(energy, par_dict)
        true_recomb = np.clip(np.random.normal(recomb, drecomb), 0., 1.)

        Ne = np.random.binomial(Ni, 1. - true_recomb)
        Nph = Nq - Ne
        Nex = Nq - Ni
        return Nph, Ne, Nex

    def get_NR_quanta(self, energy, field, par_dict):
        Nq_mean = energy / par_dict['W']
        Nq = np.round(np.random.normal(Nq_mean, np.sqrt(Nq_mean * par_dict['fano']))).astype(np.int64)

        quenching = self.NR_quenching(energy, par_dict)
        Nq = np.random.binomial(Nq, quenching)

        ExIonRatio = self.NR_ExIonRatio(energy, field, par_dict)
        Ni = np.random.binomial(Nq, ExIonRatio / (1. + ExIonRatio))

        penning_quenching = self.NR_Penning_quenching(energy, par_dict)
        Nex = np.random.binomial(Nq - Ni, penning_quenching)

        recomb = self.NR_recomb(energy, field, par_dict)
        if recomb < 0 or recomb > 1:
            return None, None

        Ne = np.random.binomial(Ni, 1. - recomb)
        Nph = Ni + Nex - Ne
        return Nph, Ne, Nex


class NEST_quanta_generator:

    def __init__(self):
        self.nc = nestpy.NESTcalc(nestpy.DetectorExample_XENON10())
        ## not sure if nestpy RNG issue was solved, so randomize NEST internal state
        for i in range(np.random.randint(100)):
            self.nc.GetQuanta(self.nc.GetYields(energy=np.random.uniform(10, 100)))

    def get_quanta(self, interaction, energy, field):
        y = self.nc.GetYields(
            interaction=nestpy.INTERACTION_TYPE(interaction),
            energy=energy,
            drift_field=field,
        )
        q_ = self.nc.GetQuanta(y)
        return q_.photons, q_.electrons, q_.excitons


class BETA_quanta_generator:

    def __init__(self):
        # ToDo: Should user be able to set custom beta-generator input files?
        cs1_spline_path = 'epix/data_files/cs1_beta.pkl'
        cs2_spline_path = 'epix/data_files/cs2_beta.pkl'
        with open(cs1_spline_path, 'rb') as f:
            self.cs1_spline = pickle.load(f)
        with open(cs2_spline_path, 'rb') as f:
            self.cs2_spline = pickle.load(f)

        # ToDo: Should the numbers be in the config file?
        self.XENONnT_g1 = 0.151  ## v5
        self.XENONnT_g2 = 16.450  ## v5

        self.get_quanta_vectorized = np.vectorize(self.get_quanta, excluded="self")

    def get_quanta(self, interaction, energy, field):
        beta_photons = self.cs1_spline(energy) / self.XENONnT_g1
        beta_electrons = self.cs2_spline(energy) / self.XENONnT_g2
        beta_excitons = 0.0

        return beta_photons, beta_electrons, beta_excitons
