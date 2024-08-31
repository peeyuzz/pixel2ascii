import sys
import os
import time
from urllib.request import urlopen
import argparse
from PIL import Image
import numpy as np
import cv2
from io import BytesIO
import imageio

ASCII_CHARS = " .:-=+*%#@"

def clear_console():
    # Clear command line screen
    os.system('cls' if os.name == 'nt' else 'clear')

def generate_ascii_art(file_path, scale_factor=1, contrast=1):
    # Handle URL
    if file_path.startswith(("http://", "https://")):
        image = Image.open(urlopen(file_path))
        return get_ascii(image, scale_factor, contrast)

    # Handle static images (jpg, png, etc.)
    elif file_path.endswith((".jpg", ".jpeg", ".png")):
        image = Image.open(file_path)
        return get_ascii(image, scale_factor, contrast)

    # Handle GIF and MP4 files
    elif file_path.endswith((".gif", ".mp4")):
        frames = []
        if file_path.endswith(".gif"):
            gif = Image.open(file_path)
            try:
                while True:
                    frames.append(get_ascii(gif.copy(), scale_factor, contrast))
                    gif.seek(gif.tell() + 1)
            except EOFError:
                pass  # End of sequence

        elif file_path.endswith(".mp4"):
            cap = cv2.VideoCapture(file_path)
            while True:
                print("extracting frames..")
                ret, frame = cap.read()
                if not ret:
                    break
                # Convert frame (OpenCV image) to PIL image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                frames.append(get_ascii(pil_image, scale_factor, contrast))
            cap.release()
        
        return frames

def get_ascii(image, scale_factor=1, contrast=1):
    # Ensure scale_factor and contrast are correctly converted
    scale_factor = float(scale_factor)
    contrast = float(contrast)

    original_height, original_width = image.height, image.width
    scaled_height = int(original_height / (scale_factor * 2))
    scaled_width = int(original_width / scale_factor)

    # Resize image to the new dimensions
    resized_image = image.resize((scaled_width, scaled_height))

    # Convert image to RGB if necessary
    if resized_image.mode != 'RGB':
        resized_image = resized_image.convert('RGB')

    pixels = resized_image.load()

    ASCIIED_IMAGE = ""
    for y in range(resized_image.height):
        for x in range(resized_image.width):
            R, G, B = pixels[x, y]
            # Adjust contrast
            avg = (R + G + B) / 3.0
            avg = avg * contrast

            # Calculate ASCII character index
            ASCII_CHARS_INDEX = int(len(ASCII_CHARS) * (avg) / 255.0)
            ASCII_CHARS_INDEX = min(ASCII_CHARS_INDEX, len(ASCII_CHARS) - 1)

            ASCIIED_IMAGE += ASCII_CHARS[ASCII_CHARS_INDEX]
        ASCIIED_IMAGE += "\n"

    return ASCIIED_IMAGE

def animate_frames(frames, fps=60):
    # Display frames one by one to create an animation effect
    frame_duration = 1 / fps
    for frame in frames:
        clear_console()
        print(frame)
        time.sleep(frame_duration)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASCII art generator")
    parser.add_argument("--file-path", required=True, help="Path to the image, video, GIF or URL")
    parser.add_argument("--scale-factor", required=False, help="Scale factor for the image (default = 1)", default=1)
    parser.add_argument("--contrast", required=False, help="Contrast factor for the image (default = 1)", default=1)
    parser.add_argument("--fps", required=False, help="Frames per second for animation (default = 60)", default=60)
    args = parser.parse_args()

    art = generate_ascii_art(args.file_path, args.scale_factor, args.contrast)

    # Handle both single image and list of frames
    if isinstance(art, list):
        animate_frames(art, int(args.fps))
    else:
        print(art)
