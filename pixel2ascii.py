import sys
import os
import time
from urllib.request import urlopen
import argparse
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm  # For progress bar
import logging  # For logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ASCII_CHARS = " .:-=+*%#@"

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

def load_image(file_path):
    """Loads an image from a file path or URL."""
    if file_path.startswith(("http://", "https://")):
        logging.info(f"Loading image from URL: {file_path}")
        return Image.open(urlopen(file_path)), "png"
    elif file_path.endswith((".jpg", ".jpeg", ".png", ".gif")):
        logging.info(f"Loading image from file: {file_path}")
        return Image.open(file_path)
    else:
        raise ValueError("Unsupported file format")

def generate_ascii(image, scale_factor=1, contrast=1):
    """Generates ASCII art for an image."""
    image = image.convert("L")
    image_np = np.array(image, dtype=np.float32) * contrast
    image_np = np.clip(image_np, 0, 255)

    height, width = image_np.shape
    new_width = int(width / scale_factor)
    new_height = int(height / (scale_factor * 2))
    resized_image = cv2.resize(image_np, (new_width, new_height))

    ascii_image = "\n".join(
        "".join(ASCII_CHARS[int(value / 255 * (len(ASCII_CHARS) - 1))] for value in row)
        for row in resized_image
    )

    return ascii_image

def process_video(file_path, scale_factor=1, contrast=1):
    """Processes video file and generates ASCII art for each frame."""
    frames = []
    cap = cv2.VideoCapture(file_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logging.info(f"Processing video: {file_path} with {frame_count} frames")

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(frame_to_ascii, frame, scale_factor, contrast)
            for ret, frame in tqdm(iter(lambda: cap.read(), (False, None)), total=frame_count, desc="Frames processed")
            if ret
        ]
        frames = [future.result() for future in futures]
    cap.release()

    return frames

def frame_to_ascii(frame, scale_factor=1, contrast=1):
    """Converts a single frame to ASCII art."""
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = np.clip(frame_gray * float(contrast), 0, 255)
    new_width = int(frame_gray.shape[1] / scale_factor)
    new_height = int(frame_gray.shape[0] / (scale_factor * 2))
    resized_frame = cv2.resize(frame_gray, (new_width, new_height))

    ascii_frame = "\n".join(
        "".join(ASCII_CHARS[int(value / 255 * (len(ASCII_CHARS) - 1))] for value in row)
        for row in resized_frame
    )

    return ascii_frame

def save_ascii_art(art, file_format, output_path):
    """Saves ASCII art to an output file."""
    font_path = r"c:\WINDOWS\Fonts\CONSOLA.TTF"
    font = ImageFont.truetype(font_path, size=15)

    if isinstance(art, list):
        logging.info(f"Saving ASCII art as {file_format} to {output_path}")
        if file_format == 'gif':
            images = [convert_text_to_image(ascii_frame, font) for ascii_frame in art]
            images[0].save(output_path, save_all=True, append_images=images[1:], duration=100, loop=0)
        elif file_format == 'mp4':
            height = len(art[0].split('\n')) * 20
            width = len(art[0].split('\n')[0]) * 15
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 10.0, (width, height))
            for ascii_frame in art:
                frame = convert_text_to_image(ascii_frame, font)
                out.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
            out.release()
    else:
        convert_text_to_image(art, font).save(output_path)

def convert_text_to_image(text, font):
    """Converts ASCII text to an image."""
    width = len(text.split('\n')[0]) * 15
    height = len(text.split('\n')) * 20
    img = Image.new('RGB', (width, height), color='black')
    d = ImageDraw.Draw(img)
    d.text((0, 0), text, fill='white', font=font)
    return img

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASCII art generator")
    parser.add_argument("--file-path", required=True, help="Path to the image, video, GIF or URL")
    parser.add_argument("--scale-factor", required=False, help="Scale factor for the image (default = 1)", default=1, type=float)
    parser.add_argument("--contrast", required=False, help="Contrast factor for the image (default = 1)", default=1, type=float)
    parser.add_argument("--fps", required=False, help="Frames per second for animation (default = 60)", default=60, type=int)
    parser.add_argument("--save", action="store_true", help="Save the ASCII output")
    args = parser.parse_args()

    try:
        file_format = args.file_path.split('.')[-1]
        if file_format in ["jpg", "jpeg", "png"]:
            image = load_image(args.file_path)
            ascii_art = generate_ascii(image, args.scale_factor, args.contrast)
            print(ascii_art)
        elif file_format in ["gif", "mp4"]:
            ascii_art = process_video(args.file_path, args.scale_factor, args.contrast)
            for frame in ascii_art:
                clear_console()
                print(frame)
                time.sleep(1 / args.fps)
        if args.save:
            save_ascii_art(ascii_art, file_format, f"ascii_output.{file_format}")
            logging.info(f"ASCII art saved to ascii_output.{file_format}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
