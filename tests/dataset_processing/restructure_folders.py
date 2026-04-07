import os
import shutil

# List of root folders to reorganise
root_folders = [
    "data/waveforms_earthquakes_nonoise",
    "data/waveforms_explosions_nonoise",
    "data/waveforms_noise_only"
]

for root in root_folders:
    print(f"Processing {root} ...")
    
    # Walk through the first-level subfolders (time windows)
    for time_folder in os.listdir(root):
        time_path = os.path.join(root, time_folder)
        if not os.path.isdir(time_path):
            continue
        
        # Loop through station subfolders
        for station_folder in os.listdir(time_path):
            station_path = os.path.join(time_path, station_folder)
            if not os.path.isdir(station_path):
                continue
            
            # Target station folder at root level
            target_station_path = os.path.join(root, station_folder)
            os.makedirs(target_station_path, exist_ok=True)
            
            # Move all .mseed files to the station folder
            for fname in os.listdir(station_path):
                if fname.endswith(".mseed"):
                    src = os.path.join(station_path, fname)
                    dst = os.path.join(target_station_path, fname)
                    shutil.move(src, dst)
            
            # Remove empty original folder
            if not os.listdir(station_path):
                os.rmdir(station_path)
        
        # Remove empty time folder
        if not os.listdir(time_path):
            os.rmdir(time_path)

print("Done reorganizing dataset!")