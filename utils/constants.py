from pathlib import Path
import sys

if sys.platform == 'win32':
    SSD_PLIF_FOLDER = Path(r'D:\Data\enose-plif\plif')
    SSD_ENOSE_FOLDER = Path(r'D:\Data\enose-plif\enose')
else: # for mac
    SSD_PLIF_FOLDER = Path('/Volumes/T7/data/enose-plif/plif')
    SSD_ENOSE_FOLDER = Path('/Volumes/T7/data/enose-plif/enose')
    SSD_COMBINED_FOLDER = Path('/Volumes/T7/data/enose-plif/enose-plif-combined')


# Local folders
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_DATA_FOLDER = PROJECT_ROOT / 'data'
LOCAL_OUT_FOLDER = PROJECT_ROOT / 'out'

ENOSE_BOUNDS = {
    "LU": (480, 580, 136, 315),
    "LC": (480, 580, -36, 143),
    "LR": (312, 412, -39, 140),
}

# PLIF 2D spaces where we retrieve data
PLIF_USEFUL_BOUNDS = {
    "LC": {
        "A": (537, 546, 93, 113),
        "B": (514, 523, 93, 113),
        "C": (534, 541, 116, 122),
        "D": (519, 526, 116, 122),
    },
    "LR": {
        "A": (370, 379, 96, 115),
        "B": (347, 356, 96, 115),
        "C": (367, 374, 119, 125),
        "D": (352, 359, 119, 125),
    },
    "LU": {
        "A": (538, 547, 271, 290),
        "B": (515, 524, 272, 291),
        "C": (535, 542, 294, 300),
        "D": (520, 527, 295, 301),
    },
}


# In which PLIF area, each MOX gas sensor belongs to
MOX_PLIF_BOUND_RELATIONSHIPS = {
    '1': 'A',
    '2': 'A',
    '3': 'A',
    '4': 'C',
    '5': 'B',
    '6': 'B',
    '7': 'B',
    '8': 'D',
}

LOCATIONS = {
    "r56": 'LC',
    "r57": 'LC',
    "r58": 'LC',
    "r59": 'LC',
    "r60": 'LC',
    "r61": 'LC',
    "r70": 'LR',
    "r71": 'LR',
    "r75": 'LU',
    "r76": 'LU',
}

SPEEDS = {
    "r56": 10,
    "r57": 15,
    "r58": 20,
    "r59": 10,
    "r60": 15,
    "r61": 20,
    "r65": 10,
    "r66": 10,
    "r70": 10,
    "r71": 10,
    "r75": 10,
    "r76": 10,
}


ENOSE_ZONE_SENSOR = {
    'A': f'R_gas_{1}',
    'B': f'R_gas_{5}',
    'C': f'R_gas_{4}',
    'D': f'R_gas_{8}',
}

# Sensing area for each MOX sensor
MOX_SENSING_BOUNDS = {
    location: {
        sensor: PLIF_USEFUL_BOUNDS[location][area]
        for sensor, area in MOX_PLIF_BOUND_RELATIONSHIPS.items()}
    for location in ['LC', 'LR', 'LU']
}

PLIF_REFS = {
    "LC": "r56", # LC
    "LR": "r70", # LR
    "LU": "r75", # LU    
}

# Useful
PLIF_SAMPLING_FREQUENCY = 20
MOX_SAMPLING_FREQUENCY = 1000
INTERP_METHOD ='pchip'

fs_original = 20 # Hz
fs_target = 1000 # Hz
