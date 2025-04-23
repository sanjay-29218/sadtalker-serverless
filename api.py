import os
import shutil
import uuid
import runpod
from urllib.parse import unquote
from src.gradio_demo import SadTalker
import logging # Import logging
import base64 # Import base64 earlier

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Create required directories
# These might not be strictly necessary if using /tmp and network volume,
# but leaving them doesn't hurt unless they conflict with network volume mounts.
# UPLOAD_DIR = "/workspace/uploads" # Commented out as likely not needed with /tmp
# RESULTS_DIR = "/workspace/results" # Commented out as likely not needed with /tmp
# It's safer to ensure these exist within the container's ephemeral storage if used
# os.makedirs(UPLOAD_DIR, exist_ok=True) # Commented out
# os.makedirs(RESULTS_DIR, exist_ok=True) # Commented out

# --- Checkpoint Path ---
CHECKPOINT_PATH = '/runpod-volume/checkpoints' # Use relative path for local/default execution
CONFIG_PATH = 'src/config'

logging.info(f"Attempting to initialize SadTalker...")
logging.info(f"Using checkpoint_path: {CHECKPOINT_PATH}")
logging.info(f"Using config_path: {CONFIG_PATH}")


# Check if checkpoint path exists and list contents (optional but helpful for debugging)
try:
    if os.path.exists(CHECKPOINT_PATH):
        logging.info(f"Checkpoint path {CHECKPOINT_PATH} exists.")
        # Listing contents might be verbose or cause issues if there are many files/permission problems
        # logging.info(f"Contents of {CHECKPOINT_PATH}: {os.listdir(CHECKPOINT_PATH)}")
    else:
        logging.warning(f"Checkpoint path {CHECKPOINT_PATH} does not exist!")
except Exception as e:
    logging.error(f"Error accessing checkpoint path {CHECKPOINT_PATH}: {str(e)}")


# Initialize SadTalker
sad_talker = None # Ensure sad_talker is defined in this scope
try:
    sad_talker = SadTalker(
        checkpoint_path=CHECKPOINT_PATH,
        config_path=CONFIG_PATH,
        lazy_load=True
    )
    logging.info("SadTalker initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize SadTalker: {str(e)}", exc_info=True) # Log traceback
    # Depending on severity, you might want to exit or prevent the handler from running
    # For now, we'll let it proceed, but the handler will likely fail later.


def handler(job):
    logging.info("Handler started.")

    if sad_talker is None:
         logging.error("SadTalker not initialized, cannot process job.")
         return {"error": "SadTalker model could not be loaded during startup."}


    # Define temporary paths early for easier cleanup
    temp_dir = '/tmp'
    temp_image_path = None
    temp_audio_path = None
    temp_result_dir = None
    video_path = None # Initialize video_path

    try:
        # Get inputs
        job_input = job['input']
        image_data = job_input.get('image')
        audio_data = job_input.get('audio')
        logging.info("Received job input.")

        # Validate inputs
        if not image_data or not audio_data:
            logging.error("Missing image or audio data.")
            return {"error": "Both image and audio are required"}

        # Get processing parameters
        preprocess = job_input.get('preprocess', 'crop')
        still_mode = job_input.get('still_mode', False)
        use_enhancer = job_input.get('use_enhancer', False)
        batch_size = job_input.get('batch_size', 1)
        size = job_input.get('size', 256)
        pose_style = job_input.get('pose_style', 0)
        logging.info(f"Processing parameters: preprocess={preprocess}, still_mode={still_mode}, use_enhancer={use_enhancer}, batch_size={batch_size}, size={size}, pose_style={pose_style}")


        # Create temporary file paths
        job_id = job.get('id', uuid.uuid4()) # Use job id for uniqueness if available
        temp_image_path = os.path.join(temp_dir, f"{job_id}_image.jpg")
        temp_audio_path = os.path.join(temp_dir, f"{job_id}_audio.wav")
        logging.info(f"Using temporary image path: {temp_image_path}")
        logging.info(f"Using temporary audio path: {temp_audio_path}")


        # Save input files
        try:
            logging.info("Decoding and saving input files...")
            with open(temp_image_path, "wb") as f:
                # Simplistic check for base64, might need refinement
                if isinstance(image_data, str): # and is_base64(image_data): # Add a helper is_base64 if needed
                     f.write(base64.b64decode(image_data))
                # TODO: Add handling for URL inputs if needed
                # elif is_url(image_data): download_file(image_data, f)
                else:
                    raise ValueError("Unsupported image data format")


            with open(temp_audio_path, "wb") as f:
                 # Simplistic check for base64, might need refinement
                if isinstance(audio_data, str): # and is_base64(audio_data): # Add a helper is_base64 if needed
                     f.write(base64.b64decode(audio_data))
                # TODO: Add handling for URL inputs if needed
                # elif is_url(audio_data): download_file(audio_data, f)
                else:
                    raise ValueError("Unsupported audio data format")
            logging.info("Input files saved successfully.")


        except Exception as e:
            logging.error(f"Failed to save input files: {str(e)}", exc_info=True)
            # No need to clean up here, finally block will handle it
            return {"error": f"Failed to process input files: {str(e)}"}


        # Define result path within the container's temporary storage first
        temp_result_dir = os.path.join(temp_dir, 'results', str(job_id)) # Unique result dir per job
        os.makedirs(temp_result_dir, exist_ok=True)
        logging.info(f"Using temporary result directory: {temp_result_dir}")


        # Generate video
        try:
            logging.info("Starting video generation...")
            video_path = sad_talker.test(
                source_image=temp_image_path,
                driven_audio=temp_audio_path,
                preprocess=preprocess,
                still_mode=still_mode,
                use_enhancer=use_enhancer,
                batch_size=batch_size,
                size=size,
                pose_style=pose_style,
                result_dir=temp_result_dir
            )
            logging.info(f"Video generation finished. Raw output path: {video_path}")


            if not video_path or not os.path.exists(video_path):
                logging.error("Video generation failed or video path is invalid/missing.")
                # No need to clean up here, finally block will handle it
                return {"error": "Failed to generate video or video path is invalid"}

            # Read the video file as bytes
            logging.info(f"Reading video file from: {video_path}")
            with open(video_path, 'rb') as video_file:
                video_bytes = video_file.read()
            logging.info(f"Video file read successfully ({len(video_bytes)} bytes).")


            # Encode video to base64 for JSON response
            logging.info("Encoding video to base64...")
            video_base64 = base64.b64encode(video_bytes).decode('utf-8')
            logging.info("Video encoded successfully. Preparing success response.")


            # Success case
            return {
                "status": "success",
                "video": video_base64
            }

        except Exception as e:
            logging.error(f"Video generation process failed: {str(e)}", exc_info=True) # Log traceback
            # No need to clean up here, finally block will handle it
            return {"error": f"Video generation failed: {str(e)}"}

    except Exception as e:
        logging.error(f"Unexpected error in handler: {str(e)}", exc_info=True) # Log traceback
        # No need to clean up here, finally block will handle it
        return {"error": f"Unexpected error in handler: {str(e)}"}

    finally:
        # --- Cleanup ---
        logging.info("Executing cleanup...")
        if temp_image_path and os.path.exists(temp_image_path):
            try:
                os.remove(temp_image_path)
                logging.info(f"Removed temp image: {temp_image_path}")
            except Exception as e:
                logging.warning(f"Failed to remove temp image {temp_image_path}: {e}")
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
                logging.info(f"Removed temp audio: {temp_audio_path}")
            except Exception as e:
                 logging.warning(f"Failed to remove temp audio {temp_audio_path}: {e}")
        # Note: video_path points *inside* temp_result_dir, so removing the dir handles it.
        if temp_result_dir and os.path.exists(temp_result_dir):
             try:
                 shutil.rmtree(temp_result_dir)
                 logging.info(f"Removed temp result directory: {temp_result_dir}")
             except Exception as e:
                 logging.warning(f"Failed to remove temp result directory {temp_result_dir}: {e}")
        logging.info("Handler finished.")


if __name__ == "__main__":
    # Check if SadTalker was initialized before starting the server
    if sad_talker:
        logging.info("Starting RunPod serverless handler...")
        runpod.serverless.start({"handler": handler})
    else:
        logging.critical("SadTalker failed to initialize. Server not starting.")
        # Potentially exit or raise an error to prevent RunPod from thinking it started successfully
        # For now, just log critically. RunPod might retry the pod initialization.
        # Consider: raise RuntimeError("SadTalker initialization failed") 