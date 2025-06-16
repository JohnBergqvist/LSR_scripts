#!/bin/bash

# Check if directory path is provided
if [ -z "$1" ]; then
  echo "Usage: $0 /full/path/to/directory_with_run_folders"
  exit 1
fi

ROOT="$1"

# Ensure the directory exists
if [ ! -d "$ROOT" ]; then
  echo "Error: Directory '$ROOT' does not exist."
  exit 1
fi

# Loop through all subdirectories two levels deep
find "$ROOT" -mindepth 2 -maxdepth 2 -type d | while read -r dir; do
  base=$(basename "$dir")

  # Match and remove the 'cult_*_*_*_*' part if it exists
  if [[ "$base" =~ ^(run_[0-9]+)_cult_[0-9]+_[0-9]+_[a-zA-Z]_[0-9]+_(.*)$ ]]; then
    prefix="${BASH_REMATCH[1]}"
    suffix="${BASH_REMATCH[2]}"
    corrected_name="${prefix}_${suffix}"

    parent_dir=$(dirname "$dir")
    new_path="${parent_dir}/${corrected_name}"

    echo "Renaming:"
    echo "  From: $dir"
    echo "  To:   $new_path"

    mv "$dir" "$new_path"
  fi
done

