#!/usr/bin/env python3

import sys
import os
import time
from urllib.request import urlopen
import argparse
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import logging
import subprocess

__version__ = "1.0.0"

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ASCII_CHARS = " .:-=+*%#@"

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

def load_image(file_path):
    """Loads an image from a file path or URL."""
    try:
        if file_path.startswith(("http://", "https://")):
            logger.info(f"Loading image from URL: {file_path}")
            return Image.open(urlopen(file_path))
        elif file_path.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".jfif")):
            logger.info(f"Loading image from file: {file_path}")
            return Image.open(file_path)
        else:
            raise ValueError("Unsupported file format")
    except Exception as e:
        logger.error(f"Error loading image: {e}")
        raise

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
    logger.info(f"Processing video: {file_path} with {frame_count} frames")

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
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
    try:
        font_path = get_font_path()
        font = ImageFont.truetype(font_path, size=15)

        max_width, max_height = 4096, 2304

        if isinstance(art, list):
            logger.info(f"Saving ASCII art as {file_format} to {output_path}")
            if file_format == 'gif':
                save_as_gif(art, font, output_path)
            elif file_format == 'mp4':
                save_as_mp4(art, font, output_path, max_width, max_height)
        else:
            save_as_image(art, font, output_path)

    except Exception as e:
        logger.error(f"Error saving ASCII art: {e}")
        raise

def get_font_path():
    """Returns the appropriate font path based on the operating system."""
    if os.name == 'nt':  # Windows
        return r"c:\WINDOWS\Fonts\CONSOLA.TTF"
    elif os.name == 'posix':  # macOS and Linux
        font_paths = [
            '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf',  # Linux
            '/Library/Fonts/Courier New.ttf',  # macOS
        ]
        for path in font_paths:
            if os.path.exists(path):
                return path
    raise FileNotFoundError("Suitable font not found. Please install DejaVu Sans Mono or Courier New.")

def save_as_gif(frames, font, output_path):
    """Saves ASCII frames as an animated GIF."""
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(convert_text_to_image, ascii_frame, font) for ascii_frame in frames]
        images = list(tqdm(executor.map(lambda f: f.result(), futures), total=len(futures), desc="Converting frames"))
    
    images[0].save(output_path, save_all=True, append_images=images[1:], duration=17, loop=0)
    logger.info(f"GIF saved successfully to {output_path}")

def save_as_mp4(frames, font, output_path, max_width, max_height):
    """Saves ASCII frames as an MP4 video."""
    height = len(frames[0].split('\n')) * 15
    width = len(frames[0].split('\n')[0]) * 7

    if width > max_width or height > max_height:
        width, height = resize_dimensions(width, height, max_width, max_height)

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    fps = 30.0
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)
    out.set(cv2.VIDEOWRITER_PROP_QUALITY, 40)

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_frame, ascii_frame, font, width, height) for ascii_frame in frames]
        for future in tqdm(futures, desc="Saving frames to MP4", unit="frame"):
            out.write(future.result())

    out.release()

    compress_video(output_path)
    logger.info(f"Compressed MP4 saved successfully to {output_path}")

def save_as_image(art, font, output_path):
    """Saves ASCII art as a single image."""
    img = convert_text_to_image(art, font)
    img.save(output_path)
    logger.info(f"ASCII art saved as image to {output_path}")

def convert_text_to_image(text, font):
    """Converts ASCII text to an image."""
    lines = text.split('\n')
    width = max(len(line) for line in lines) * 7
    height = len(lines) * 15
    img = Image.new('RGB', (width, height), color='black')
    d = ImageDraw.Draw(img)
    d.text((0, 0), text, fill='white', font=font)
    return img

def resize_dimensions(width, height, max_width, max_height):
    """Calculates the new dimensions to fit within the maximum while maintaining aspect ratio."""
    aspect_ratio = width / height

    if width > max_width:
        width = max_width
        height = int(width / aspect_ratio)

    if height > max_height:
        height = max_height
        width = int(height * aspect_ratio)

    return width, height

def process_frame(ascii_frame, font, width, height):
    """Converts an ASCII frame to an image and resizes if necessary."""
    frame = convert_text_to_image(ascii_frame, font)
    
    if frame.size[0] > width or frame.size[1] > height:
        frame = frame.resize((width, height), Image.LANCZOS)
    
    return cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

def compress_video(input_path):
    """Compresses the video using FFmpeg."""
    compressed_output_path = input_path.replace('.mp4', '_compressed.mp4')
    ffmpeg_command = f'ffmpeg -i {input_path} -vcodec libx264 -crf 28 {compressed_output_path}'
    try:
        subprocess.run(ffmpeg_command, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.replace(compressed_output_path, input_path)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error compressing video: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="ASCII art generator")
    parser.add_argument("file_path", help="Path to the image, video, GIF or URL")
    parser.add_argument("--scale-factor", type=float, default=1, help="Scale factor for the image (default: 1)")
    parser.add_argument("--contrast", type=float, default=1, help="Contrast factor for the image (default: 1)")
    parser.add_argument("--fps", type=int, default=60, help="Frames per second for animation (default: 60)")
    parser.add_argument("--save", action="store_true", help="Save the ASCII output")
    parser.add_argument("--output", help="Output file path (required if --save is used)")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args()

    if args.save and not args.output:
        parser.error("--output is required when --save is used")

    try:
        file_format = args.file_path.split('.')[-1].lower() if not args.file_path.startswith(("http://", "https://")) else "jpg"
        
        if file_format in ["jpg", "jpeg", "png", "jfif"] or args.file_path.startswith(("http://", "https://")):
            image = load_image(args.file_path)
            ascii_art = generate_ascii(image, args.scale_factor, args.contrast)
            print(ascii_art)
        elif file_format in ["gif", "mp4"]:
            ascii_art = process_video(args.file_path, args.scale_factor, args.contrast)
            for frame in ascii_art:
                clear_console()
                print(frame)
                time.sleep(1 / args.fps)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        if args.save:
            save_ascii_art(ascii_art, file_format, args.output)
            logger.info(f"ASCII art saved to {args.output}")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()