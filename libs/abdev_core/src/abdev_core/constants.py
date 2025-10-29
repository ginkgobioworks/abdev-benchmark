"""Constants and configuration for the antibody developability benchmark."""

# Biophysical properties measured in the benchmark
PROPERTY_LIST = [
    "HIC",
    "Tm2",
    "Titer",
    "PR_CHO",
    "AC-SINS_pH7.4",
]

# Whether higher values are better for each assay
ASSAY_HIGHER_IS_BETTER = {
    "HIC": False,
    "Tm2": True,
    "Titer": True,
    "PR_CHO": False,
    "AC-SINS_pH7.4": False,
}

# Dataset names
DATASETS = ["GDPa1", "GDPa1_cross_validation", "heldout_test"]
