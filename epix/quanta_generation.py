import numpy as np
import nestpy

print(f'Using nestpy version {nestpy.__version__}')

@np.vectorize
def quanta_from_NEST(en, model, E_Field, A, Z, VolumeId):
    """
    Function which uses NEST to yield photons and electrons
    for a given set of parameters.

    Args:
        en (numpy.array): Energy deposit of the interaction [keV]
        model (numpy.array): Nest Id for qunata generation (integers)
        E_field (numpy.array): Field value in the interaction site [V/cm]
        A (numpy.array): Atomic mass number
        Z (numpy.array): Atomic number
        VolumeId (numpy.array): Volume Id of the interaction site

    Returns:
        photons (numpy.array): Number of generated photons
        electrons (numpy.array): Number of generated electrons
    """
    nc = nestpy.NESTcalc(nestpy.VDetector())
    density = 2.862  # g/cm^3
    
    # Fix for Kr83m events.
    # Energies have to be exactly 32.1 keV or 9.1 keV
    max_allowed_energy_difference=1 #keV 
    if model == 11:
        if abs(en - 32.1) > max_allowed_energy_difference:
            en=32.1
        if abs(en - 9.1) > max_allowed_energy_difference:
            en=9.1
            
    y = nc.GetYields(nestpy.INTERACTION_TYPE(model),
                     en,
                     density,
                     E_Field,
                     A,
                     Z,
                     )

    event_quanta = nc.GetQuanta(y, density)

    photons = event_quanta.photons

    if VolumeId == 2: #belowcathode
        electrons = 0
    else:
        electrons = event_quanta.electrons

    return photons, electrons