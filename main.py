import os
import argparse
import logging
import sys
from preprocess.trim_synced_streams import trim_videos
from BiRefNet.demo import background_matting

# Setup logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Prints to console
        logging.FileHandler('output.log')  # Logs to a file named 'output.log'
    ]
)

def run_command(command):
    """Function to run a shell command and check for errors."""
    logging.info(f"Running command: {command}")
    # Use os.system to execute the command and check the return code
    ret_code = os.system(command)
    if ret_code != 0:
        logging.error(f"Error occurred while executing: {command}")
        sys.exit(ret_code)  # Exit the script if command fails

def calibrate_cameras(args):
    # Variables (these would normally come from user input or environment variables)
    base_path = args.root_dir
    pattern = args.pattern
    grid = args.grid
    grid_step = args.grid_step
    intri = args.list[0]
    extri = args.list[1]

    # change the directory to the Easymocap code
    os.chdir('EasyMocap')
    for item in args.list:
        # Command 0: extract individual frames from the video
        logging.info(f"Extracting frames from {item}")
        extract_frames_command = f" python3 scripts/preprocess/extract_video.py {base_path}/{item} --no2d"
        run_command(extract_frames_command)

        # Command 1: Detect the chessboard in all cameras
        logging.info(f"Detecting chessboard in {item}")
        detect_chessboard_command = f"python3 apps/calibration/detect_chessboard.py {base_path}/{item} --out {base_path}/{item}/output/calibration --pattern {pattern} --grid {grid}"
        run_command(detect_chessboard_command)


    # Command 2: Intrinsic camera calibration
    calib_intri_command = f"python3 apps/calibration/calib_intri.py {base_path}/{intri}"
    run_command(calib_intri_command)

    # Command 3: Extrinsic camera calibration
    calib_extri_command = f"python3 apps/calibration/calib_extri.py {base_path}/{extri} --intri {base_path}/{intri}/output/intri.yml"
    run_command(calib_extri_command)

    # Command 4: Check calibration
    check_calib_command = f"python3 apps/calibration/check_calib.py {base_path}/{extri} --out {base_path}/{extri} --mode cube --write --grid_step {grid_step}"
    run_command(check_calib_command)

    # change to the root directory
    os.chdir('..')

def background_mask(args):
    os.chdir('BiRefNet')
    # Variables (these would normally come from user input or environment variables)
    base_path = args.root_dir
    output_folder = os.path.join(args.output, 'background_matting')
    os.mkdir(output_folder, exist_ok=True)

    input_path = os.path.join(base_path, 'videos', 'source')
    background_matting(input_path, output_folder)

def body_segmentation(args):
    # Variables (these would normally come from user input or environment variables)
    base_path = args.root_dir
    os.chdir('bodysegmentation/lite/scripts/demo/torchscript')
    input_path = os.path.join(base_path, 'videos', 'source')
    output_folder = os.path.join(args.output, 'BodySegmentation')
    os.mkdir(output_folder, exist_ok=True)

    # Command 0: Run Sapiens
    sapiens_command = f"bash seg.sh {input_path} {output_folder}"
    run_command(sapiens_command)
    os.chdir('../../../../../..')

def main(args):
    base_path = args.root_dir

    if args.trim:
        # Loop through the items in the list
        list.append('videos')
        for item in args.list:
            if item not in os.listdir(base_path):
                logging.error(f"Item {item} not found in the root directory.")
                sys.exit(1)

            input = os.path.join(base_path, item)
            # Command 0: Trim videos
            trim_videos(input, args.output)
    
    if args.calibrate:
        calibrate_cameras(args)
    
    if args.background_matting:
        background_mask(args)

    if args.sapiens:
        body_segmentation(args)  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data preprocessing for GUAVA')
    parser.add_argument("--root_dir", type=str, required=True, help="Path to the root directory")
    parser.add_argument("--output", type=str, help="Path to the output folder")
    parser.add_argument('--list', nargs='+', default=['intrinsics', 'extrinsics'], help="List of items enclosed in brackets (required)")
    parser.add_argument('--pattern', type=str, default='8,13', help="Pattern in the format '8,13'")
    parser.add_argument('--grid', type=float, default=0.04, help="Grid dimension for the checkerboard")
    parser.add_argument('--grid_step', type=float, default=0.24 ,help="Grid step value for the checkerboard (cube size)")
    parser.add_argument('--trim', action='store_true', help="Trim videos")
    parser.add_argument('--calibrate', action='store_true', help="Calibrate cameras")
    parser.add_argument('--background_matting', action='store_true', help="Run background matting")
    parser.add_argument('--sapiens', action='store_true', help="Run Sapiens")

    args = parser.parse_args()
    main(args)
