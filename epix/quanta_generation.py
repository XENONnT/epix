import numpy as np
import warnings
import nestpy

print(f'Using nestpy version {nestpy.__version__}')

@np.vectorize
def quanta_from_NEST(en, model, e_field, A, Z, create_s2, **kwargs):
    """
    Function which uses NEST to yield photons and electrons
    for a given set of parameters.

    Note:
        In case the energy deposit is outside of the range of NEST a -1
        is returned.

    Args:
        en (numpy.array): Energy deposit of the interaction [keV]
        model (numpy.array): Nest Id for qunata generation (integers)
        e_field (numpy.array): Field value in the interaction site [V/cm]
        A (numpy.array): Atomic mass number
        Z (numpy.array): Atomic number
        create_s2 (bool): Specifies if S2 can be produced by interaction,
            in this case electrons are generated.
        kwargs: Additional keyword arguments which can be taken by
            GetYields e.g. density.

    Returns:
        photons (numpy.array): Number of generated photons
        electrons (numpy.array): Number of generated electrons
        excitons (numpy.array): Number of generated excitons
    """
    nc = nestpy.NESTcalc(nestpy.VDetector())
    density = 2.862  # g/cm^3

    # Fix for Kr83m events.
    # Energies have to be very close to 32.1 keV or 9.4 keV
    # See: https://github.com/NESTCollaboration/nest/blob/master/src/NEST.cpp#L567
    # and: https://github.com/NESTCollaboration/nest/blob/master/src/NEST.cpp#L585
    max_allowed_energy_difference = 1  # keV
    if model == 11:
        if abs(en - 32.1) < max_allowed_energy_difference:
            en = 32.1
        if abs(en - 9.4) < max_allowed_energy_difference:
            en = 9.4

    # Some addition taken from
    # https://github.com/NESTCollaboration/nestpy/blob/e82c71f864d7362fee87989ed642cd875845ae3e/src/nestpy/helpers.py#L94-L100
    if model == 0 and en > 2e2:
        warnings.warn(f"Energy deposition of {en} keV beyond NEST validity for NR model of 200 keV - Remove Interaction")
        return -1, -1, -1
    if model == 7 and en > 3e3:
        warnings.warn(f"Energy deposition of {en} keV beyond NEST validity for gamma model of 3 MeV - Remove Interaction")
        return -1, -1, -1
    if model == 8 and en > 3e3:
        warnings.warn(f"Energy deposition of {en} keV beyond NEST validity for beta model of 3 MeV - Remove Interaction")
        return -1, -1, -1

    y = nc.GetYields(interaction=nestpy.INTERACTION_TYPE(model),
                     energy=en,
                     drift_field=e_field,
                     A=A,
                     Z=Z,
                     **kwargs
                     )

    event_quanta = nc.GetQuanta(y)  # Density argument is not use in function...

    photons = event_quanta.photons
    excitons = event_quanta.excitons
    electrons = 0
    if create_s2:
        electrons = event_quanta.electrons

    return photons, electrons, excitons
