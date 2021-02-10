import numpy as np

class MesserUpper():
    '''Class to hold all the stuff to be applied to the data. 
    The functions will be grouped together and executed by Simulator'''
    

    @staticmethod
    @numba.njit(cache=True,)
    def MakeS1(instructions,
                      s1_light_yield=1,
                      data):
        #Draw number of photons from poisson distribution with p the light yield
        for ix,inst in enumerate(instructions):
            data[ix]['s1_area']= np.random.binomial(inst['amp'], p=light_yield)
        
    @staticmethod
    @numba.njit(cache=True)
    def MakeS2(instructions,
                        drift_velocity_liquid,
                        drift_time_gate,
                        electron_lifetime_liquid,
                        electron_extraction_yield,
                        data):
        #Something electron_livetime, drift_time for a delay and electron extraction efficiency(~1)
        
        drift_time_mean = - instructions['z'] / drift_velocity_liquid+drift_time_gate

        # Absorb electrons during the drift
        electron_lifetime_correction = np.exp(- 1 * drift_time_mean / electron_lifetime_liquid)
        cy = electron_extraction_yield * electron_lifetime_correction
        cy = np.clip(cy, a_min = 0, a_max = 1)

        for ix,inst in enumerate(instructions):
            n_electron = np.random.binomial(n=inst['amp'], p=cy)
            data[ix]['s2_area'] = np.random.gaussian(n=n_electron,p=se_gain)

    @staticmethod
    def CorrectionS1(data):
        m=data['type'==1]
        data[m]['cs1'] = data[m]*#Whatever the correction factors are

    @staticmethod
    def CorrectionS2(data):
        m=data['type'==2]
        data[m]['cs1'] = data[m]*#Whatever the correction factors are

    @staticmethod
    def SmearPositions(instructions,xy_resolution,z_resolution,data):
        for ix, inst in enumerate(instructions):
            data[ix]['x'] = np.random.normal(inst['x'],p=xy_resolution)
            data[ix]['y'] = np.random.normal(inst['y'],p=xy_resolution)
            data[ix]['z'] = np.random.normal(inst['z'],p=z_resolution)
            

class  Simulator():
    '''Simulator class for epix to go from  epix instructions to fully processed data'''

    def __init__(self):
        self.mu = MesserUpper

    def simulate(self,instructions):
        #So this guy will take the instructions, fire the other functions to do all the (sort of ) smearings
        #Do large aray multiplication and returns data. Currently I assume there to be 1 S1 and 1 S2 per event!

