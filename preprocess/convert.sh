#!/bin/bash

# Input and output files
input_dir="/netscratch/jeetmal/videos/capture_10_02_2025/Shalini/"
output_dir="/netscratch/jeetmal/videos/capture_10_02_2025/Shalini/videos"


# ORIENTATION="-90" # 05, 04, 03, 02, 01
ORIENTATIONS=("-90" "-90" "-90" "-90" "-90" "-90" "-90" "-90" "-90" "-90" "-90" "-90" "90" "90" "90" "0" "0")
# Get frame rate and orientation
frame_rate=29.97002997
# orientation=$(ffprobe -v error -show_entries stream=side_data_list,rotation -of json $input | grep -o '"rotation": *[-0-9]*' | grep -o '[-0-9]*')
# frame_rate=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1:nokey=1 $input | bc)

mkdir -p $output_dir

for input in "$input_dir"/*.MOV; do

    # Get the base name of the file (without directory and extension)
    base_name=$(basename "$input" .MOV)
    
    # Set the output file path
    output="$output_dir/$base_name.mp4"
        
    ORIENTATION="${ORIENTATIONS[$orientation_index]}"

    # Decide transpose filter based on orientation
    transpose_filter=""
    if [ "$ORIENTATION" == "90" ]; then
        transpose_filter="transpose=1"
    elif [ "$ORIENTATION" == "-90" ]; then
        transpose_filter="transpose=2"
    elif [ "$ORIENTATION" == "180" ]; then
        transpose_filter="hflip,vflip"
    fi

    # Apply ffmpeg transformation
    if [ -n "$transpose_filter" ]; then
        ffmpeg -i $input -vf "$transpose_filter" -r $frame_rate -c:v libx264 -preset veryfast -crf 18 -c:a aac -y $output
    else
        ffmpeg -i $input -r $frame_rate -c:v libx264 -preset veryfast -crf 18 -c:a aac -y $output   
    fi

     # Increment orientation_index for the next file
    orientation_index=$((orientation_index + 1))

    # If the index exceeds the array length, reset to 0 (optional behavior)
    if [ "$orientation_index" -ge "${#ORIENTATIONS[@]}" ]; then
        orientation_index=0
    fi
done

