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

# Get station names
station_names = [entry.name for entry in os.scandir(mseed_dir) if entry.is_dir()]

# Create dict for a dummy station locations
DUMMY_COORDS = [0.0, 0.0, 0.0]
station_locations = {station: DUMMY_COORDS for station in station_names}

# Create station_list.json
stationListFromMseed(mseed_dir, station_locations)

print("station_list.json successfully created!")