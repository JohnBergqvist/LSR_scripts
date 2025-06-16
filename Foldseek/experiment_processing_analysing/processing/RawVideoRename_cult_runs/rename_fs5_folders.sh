#!/bin/bash

# This script renames folder names that contain 'run_' and are prefixed by anything,
# such as 'fs5_250519_'. It renames them to just 'run_xxxx_timestamp'.

# Used for converting the folder name structure when recording manually
# to the folder structure as if recorded via AirTable. 

# Check for an argument
if [ -z "$1" ]; then
  echo "Usage: $0 /path/to/target/directory"
  exit 1
fi

TARGET_DIR="$1"

# Change to the target directory
cd "$TARGET_DIR" || { echo "Directory not found: $TARGET_DIR"; exit 1; }

for dir in *run_*; do
  if [ -d "$dir" ]; then
    # Extract everything from the last occurrence of 'run_' onward
    new_name="${dir##*run_}"
    new_name="run_$new_name"
    mv "$dir" "$new_name"
    echo "Renamed '$dir' to '$new_name'"
  fi
done
