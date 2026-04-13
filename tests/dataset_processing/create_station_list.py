import os
from BlocklyEQTransformer.utils.hdf5_maker import preprocessor, stationListFromMseed

# # Directories containing mseed files
# mseed_root_dirs = [
#     "data/waveforms_earthquakes_nonoise",
#     "data/waveforms_explosions_nonoise",
#     "data/waveforms_noise_only"
# ]

# Directory containing mseed files
mseed_dir = "data/waveforms_earthquakes_nonoise"

# Get station names and locations
station_names = [entry.name for entry in os.scandir(mseed_dir) if entry.is_dir()]
locations = {
    # HE (Finland)
    "ALAJF": [None, None, None],
    "FIA0": [None, None, None],
    "FIA1": [None, None, None],
    "HEF": [68.4062, 23.6643, None],
    "HEL1": [None, None, None],
    "HEL5": [None, None, None],
    "JOF": [62.9187, 31.3092, None],
    "KAF": [62.1115, 26.3061, None],
    "KEF": [62.1667, 24.8670, None],
    "KEV": [69.75651, 27.00356, None],
    "KIF": [69.0432, 20.8040, None],
    "KPF": [61.8337, 22.0704, None],
    "KU6": [66.02547, 29.89067, None],
    "KUNI": [None, None, None],
    "LAUT": [None, None, None],
    "MEF": [60.2172, 24.3958, None],
    "NIF": [63.390016, 27.810279, None],
    "NUR": [None, None, None],
    "OBF0": [None, None, None],
    "OBF3": [None, None, None],
    "OBF8": [64.4402, 24.5172, None],
    "OUF": [64.3673, 24.7314, None],
    "PVF": [60.54535, 25.85898, None],
    "RAF": [61.0224, 21.7646, None],
    "RMF": [64.2175, 29.9318, None],
    "RSUO": [60.2027, 24.90306, None],
    "RUF": [61.4247, 28.9497, None],
    "SUF": [62.7194, 26.1474, None],
    "TOF": [66.077745, 24.332007, None],
    "TVF": [None, None, None],
    "VAF": [63.0471, 22.6683, None],
    "VJF": [60.5388, 27.5550, None],
    "VRF": [67.7480, 29.6090, None],
    "VUOS": [None, None, None],

    # UP (Sweden)
    "AAL": [60.1780, 19.9936, None],
    "ARJ": [66.242081, 16.97349, None],
    "DEL": [56.4701, 13.8705, None],
    "BRE": [63.8908, 18.5777, None],
    "FOR": [60.3870, 18.1801, None],
    "FIB": [59.9013, 17.3518, None],
    "GRA": [60.3342, 18.5396, None],
    "HUD": [61.7364, 17.1194, None],
    "KAL": [65.8596, 23.3567, None],
    "LUN": [55.6320, 13.4468, None],
    "NRT": [59.6769, 18.6308, None],
    "NYN": [59.0047, 18.0042, None],
    "OUL": [65.0859, 25.8926, None],
    "SGF": [67.44211, 26.52611, None],
    "RNF": [66.61216, 26.01074, None],

    # FN (Northern Finland)
    "KLF": [67.2347, 23.9640, None],
    "KMNF": [69.14895, 26.99611, None],
    "MSF": [65.91131, 29.04019, None],
    "OLKF": [66.3206, 29.4003, None],
    "RAJF": [68.47512, 28.31099, None],
    "RANF": [66.0147, 26.7886, None],
    "SGF": [None, None, None],
}

# Create station_list.json
stationListFromMseed(mseed_dir, locations)

print("station_list.json successfully created!")