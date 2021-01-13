import numba
import epix

# ------------
# Volumes to be used in run_epic
# ------------
# TODO: Add some sort of config file....
# TPC:
# Defining the volume:
@numba.njit  # Must be a njitted function!
def clyinder_tpc(x, y, z):
    return epix.in_cylinder(x, y, z, epix.z_cathode, epix.z_gate_mesh, epix.sensitive_volume_radius)

# Getting efield in specified volume:
# TODO: EField handling is a bit inconsistent, but no idea how to solve this right now...
# Can be either specified as a constant
efield = 200
# or loaded from file via:
# path = 'ValidPath'
# efield = MyElectricFieldHandler(path)

tpc = epix.SensitiveVolume('tpc', 1, clyinder_tpc, efield, True)

# Below Cathode:
@numba.njit  # Must be a njitted function!
def cylinder_bc(x, y, z):
    return epix.in_cylinder(x, y, z, epix.z_bottom_pmts, epix.z_cathode, epix.sensitive_volume_radius)

below_cathode = epix.SensitiveVolume('below_cathode', 2, cylinder_bc, efield, False)

# Gaseous xenon:
#TODO: Add me

# Defined volumes:
my_volumes = {'tpc': tpc,
              'below_cathode': below_cathode
              }
