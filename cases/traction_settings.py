"""Common numerical settings for all reported traction benchmarks.

Equilibrium benchmarks use velocity damping; the pressure pulse is undamped.
"""

SETTINGS = {
    "boundary_reconstruction": "local_f",
    "lattice_speed": 10.0,
    "omega": 1.8,
    "local_displacement_interval": 1,
    "local_displacement_sweeps": 1,
    "local_displacement_relax": 0.85,
    "local_displacement_boundary_weight": 100.0,
    "local_compatibility_interval": 1,
    "local_compatibility_blend": 0.5,
}
RELAXATION_DAMPING = 10.0


def command_arguments():
    return [
        item
        for key, value in SETTINGS.items()
        for item in ("--" + key.replace("_", "-"), str(value))
    ]


def apply_defaults(parser, *, dynamic=False):
    parser.set_defaults(**SETTINGS, damping_gamma=0.0 if dynamic else RELAXATION_DAMPING)


def configuration():
    return {
        **SETTINGS,
        "relaxation_damping": RELAXATION_DAMPING,
        "dynamic_damping": 0.0,
        "displacement_gradient": "centered_or_second_order_one_sided",
        "one_sided_fallback": "first_order_if_second_interior_node_unavailable",
        "characteristic_coefficient": "dt / collision_omega",
        "characteristic_source": "dt / (2*D) * (1/omega - 1/2) * B",
        "kinematic_dependency_radius": 3,
    }
