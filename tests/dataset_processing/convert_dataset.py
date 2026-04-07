import os
from BlocklyEQTransformer.utils.hdf5_maker import preprocessor

preprocessor(
    preproc_dir="data/preprocessed",
    mseed_dir="data/waveforms_earthquakes_nonoise",
    stations_json="station_list.json",
    overlap=0.3,
    n_processor=4
)