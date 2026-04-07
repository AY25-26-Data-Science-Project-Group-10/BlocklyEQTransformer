import os
import glob
import h5py
import pandas as pd
from tqdm import tqdm

# Config
INPUT_DIR = "data/waveforms_earthquakes_nonoise_processed_hdfs"
OUTPUT_DIR = "data/waveforms_merged"

MAX_STATIONS = 3

OUTPUT_HDF5 = os.path.join(OUTPUT_DIR, "earthquakes_nonoise.hdf5")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "earthquakes_nonoise.csv")

# Setup
os.makedirs(OUTPUT_DIR, exist_ok=True)

hdf5_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.hdf5")))
csv_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.csv")))

assert len(hdf5_files) == len(csv_files), "Mismatch between HDF5 and CSV files"

# Limit number of stations
if MAX_STATIONS is not None:
    hdf5_files = hdf5_files[:MAX_STATIONS]
    csv_files = csv_files[:MAX_STATIONS]

print(f"Using {len(hdf5_files)} stations")

# Merge
all_trace_names = []
with h5py.File(OUTPUT_HDF5, "w") as hdf_out:
    for hdf5_path, csv_path in tqdm(zip(hdf5_files, csv_files), total=len(hdf5_files)):
        station_name = os.path.basename(hdf5_path).replace(".hdf5", "")

        # Load CSV
        df = pd.read_csv(csv_path)

        with h5py.File(hdf5_path, "r") as hdf_in:
            data_group = hdf_in['data']
            for trace_name in df["trace_name"]:
                if trace_name not in data_group:
                    print(f"Skipping missing trace: {trace_name}")
                    continue

                new_name = f"{station_name}__{trace_name}"
                hdf_out.create_dataset(
                    new_name,
                    data=data_group[trace_name][:]
                )
                all_trace_names.append(new_name)


# Save .csv
merged_df = pd.DataFrame({"trace_name": all_trace_names})
merged_df.to_csv(OUTPUT_CSV, index=False)

print("\nMerge complete!")
print(f"HDF5: {OUTPUT_HDF5}")
print(f"CSV:  {OUTPUT_CSV}")
print(f"Total traces: {len(all_trace_names)}")