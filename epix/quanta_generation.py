import numpy as np
import nestpy

print(f'Using nestpy version {nestpy.__version__}')

@np.vectorize
def quanta_from_NEST(en, model, e_field, A, Z, create_s2, **kwargs):
    """
    Function which returns photons and electrons produced by an
    interaction.

    Note:
        In case the energy deposit is outside of the range of NEST a -1
        is returned.

    Args:
        en (float): Energy of the deposit.
        model (int): NestId to identify the interaction type.
        e_field (float): Electric field at the interaction position.
        A (int): Atomic mass number of energy deposit causing particle.
        Z (int): Atomic number "
        create_s2 (bool): Specifies if S2 can be produced by interaction,
            in this case electrons are generated.
        kwargs: Additional keyword arguments which can be taken by
            GetYields e.g. density.

    Returns:
        np.array: Number of photons
        np.array: Number of electrons
    """
    nc = nestpy.NESTcalc(nestpy.VDetector())
    density = 2.862  # g/cm^3

    # Fix for Kr83m events.
    # Energies have to be exactly 32.1 keV or 9.1 keV
    # See: https://github.com/NESTCollaboration/nest/blob/master/src/NEST.cpp#L567
    # and: https://github.com/NESTCollaboration/nest/blob/master/src/NEST.cpp#L585
    # TODO: There must be something odd with our clustering, that we need
    #  this?
    max_allowed_energy_difference = 1  # keV
    if model == 11:
        if abs(en - 32.1) > max_allowed_energy_difference:
            en = 32.1
        if abs(en - 9.1) > max_allowed_energy_difference:
            en = 9.1

    # Some addition taken from
    # https://github.com/NESTCollaboration/nestpy/blob/e82c71f864d7362fee87989ed642cd875845ae3e/src/nestpy/helpers.py#L94-L100
    if model == 0 and en > 2e2:
        return -1, -1
    if model == 7 and en > 3e3:
        return -1, -1
    if model == 8 and en > 3e3:
        return -1, -1

    y = nc.GetYields(nestpy.INTERACTION_TYPE(model),
                     en,
                     e_field,
                     A,
                     Z,
                     **kwargs
                     )

    event_quanta = nc.GetQuanta(y)  # Density argument is not use in function...

    photons = event_quanta.photons

    electrons = 0
    if create_s2:
        electrons = event_quanta.electrons

    return photons, electrons
