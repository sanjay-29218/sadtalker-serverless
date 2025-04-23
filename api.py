import os
import shutil
import uuid
import runpod
from urllib.parse import unquote
from src.gradio_demo import SadTalker

# Create required directories
UPLOAD_DIR = "/workspace/uploads"  # Changed to workspace
RESULTS_DIR = "/workspace/results" # Changed to workspace
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Initialize SadTalker with network volume path
sad_talker = SadTalker(
    checkpoint_path='/workspace/checkpoints',  # Network volume path
    config_path='src/config', 
    lazy_load=True
)

def handler(job):
    """
    RunPod handler function
    Expects job input with:
    - image: base64 or URL
    - audio: base64 or URL
    - preprocess: str
    - still_mode: bool
    - use_enhancer: bool
    - batch_size: int
    - size: int
    - pose_style: int
    """
    try:
        # Get inputs
        job_input = job['input']
        image_data = job_input.get('image')
        audio_data = job_input.get('audio')
        
        # Validate inputs
        if not image_data or not audio_data:
            return {"error": "Both image and audio are required"}

        # Get processing parameters
        preprocess = job_input.get('preprocess', 'crop')
        still_mode = job_input.get('still_mode', False)
        use_enhancer = job_input.get('use_enhancer', False)
        batch_size = job_input.get('batch_size', 1)
        size = job_input.get('size', 256)
        pose_style = job_input.get('pose_style', 0)

        # Create temporary files
        image_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}.jpg")
        audio_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}.wav")

        # Save input files
        try:
            with open(image_path, "wb") as f:
                f.write(unquote(image_data).encode())
            with open(audio_path, "wb") as f:
                f.write(unquote(audio_data).encode())
        except Exception as e:
            return {"error": f"Failed to save input files: {str(e)}"}

        # Generate video
        try:
            video_path = sad_talker.test(
                source_image=image_path,
                driven_audio=audio_path,
                preprocess=preprocess,
                still_mode=still_mode,
                use_enhancer=use_enhancer,
                batch_size=batch_size,
                size=size,
                pose_style=pose_style,
                result_dir=RESULTS_DIR
            )
            
            if not video_path or not os.path.exists(video_path):
                return {"error": "Failed to generate video"}

            # Read the video file as bytes
            with open(video_path, 'rb') as video_file:
                video_bytes = video_file.read()

            # Clean up
            os.remove(image_path)
            os.remove(audio_path)
            os.remove(video_path)

            # Return video bytes
            return {
                "status": "success",
                "video": video_bytes
            }

        except Exception as e:
            return {"error": f"Video generation failed: {str(e)}"}

    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

    finally:
        # Ensure cleanup
        if 'image_path' in locals() and os.path.exists(image_path):
            os.remove(image_path)
        if 'audio_path' in locals() and os.path.exists(audio_path):
            os.remove(audio_path)
        if 'video_path' in locals() and os.path.exists(video_path):
            os.remove(video_path)

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler}) 