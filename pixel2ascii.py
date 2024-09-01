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

ASCII_CHARS = " .:-=+*%#@"

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

def generate_ascii_art(file_path, scale_factor=1, contrast=1):
    if file_path.startswith(("http://", "https://")):
        image = Image.open(urlopen(file_path))
        return get_ascii(image, scale_factor, contrast), "png"

    elif file_path.endswith((".jpg", ".jpeg", ".png")):
        image = Image.open(file_path)
        return get_ascii(image, scale_factor, contrast), file_path.split('.')[-1]

    elif file_path.endswith((".gif", ".mp4")):
        frames = []
        if file_path.endswith(".gif"):
            gif = Image.open(file_path)
            try:
                while True:
                    frames.append(get_ascii(gif.copy(), scale_factor, contrast))
                    gif.seek(gif.tell() + 1)
            except EOFError:
                pass  

        elif file_path.endswith(".mp4"):
            cap = cv2.VideoCapture(file_path)
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    futures.append(executor.submit(process_frame, frame, scale_factor, contrast))

                for future in futures:
                    frames.append(future.result())
            cap.release()
        
        return frames, file_path.split('.')[-1]

def process_frame(frame, scale_factor, contrast):
    return frame_to_ascii(frame, scale_factor, contrast)

def frame_to_ascii(frame, scale_factor=1, contrast=1):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    frame_gray = np.clip(frame_gray * float(contrast), 0, 255)

    new_width = int(frame_gray.shape[1] / int(scale_factor))
    new_height = int(frame_gray.shape[0] / (int(scale_factor) * 2))
    resized_frame = cv2.resize(frame_gray, (new_width, new_height))

    ascii_frame = "\n".join(
        "".join(ASCII_CHARS[int(value / 255 * (len(ASCII_CHARS) - 1))] for value in row)
        for row in resized_frame
    )

    return ascii_frame

def get_ascii(image, scale_factor=1, contrast=1):
    scale_factor = float(scale_factor)
    contrast = float(contrast)

    image = image.convert("L")

    image_np = np.array(image, dtype=np.float32)
    image_np = np.clip(image_np * contrast, 0, 255)

    original_height, original_width = image_np.shape
    scaled_height = int(original_height / (int(scale_factor) * 2))
    scaled_width = int(original_width / int(scale_factor))
    resized_image = cv2.resize(image_np, (scaled_width, scaled_height))

    ascii_image = "\n".join(
        "".join(ASCII_CHARS[int(value / 255 * (len(ASCII_CHARS) - 1))] for value in row)
        for row in resized_image
    )

    return ascii_image

def animate_frames(frames, fps=60):
    frame_duration = 1 / fps
    for frame in frames:
        clear_console()
        print(frame)
        time.sleep(frame_duration)

def save_ascii_art(art, file_format, output_path):
    if isinstance(art, list):
        if file_format == 'gif':
            images = []
            font = ImageFont.load_default()
            for ascii_frame in art:
                img = Image.new('RGB', (len(ascii_frame.split('\n')[0]) * 10, len(ascii_frame.split('\n')) * 15), color='white')
                d = ImageDraw.Draw(img)
                d.text((0, 0), ascii_frame, fill='black', font=font)
                images.append(img)
            images[0].save(output_path, save_all=True, append_images=images[1:], duration=100, loop=0)
        elif file_format == 'mp4':
            height, width = len(art[0].split('\n')) * 15, len(art[0].split('\n')[0]) * 10
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 10.0, (width, height))
            for ascii_frame in art:
                img = np.zeros((height, width, 3), dtype=np.uint8)
                img.fill(255)  
                for i, line in enumerate(ascii_frame.split('\n')):
                    cv2.putText(img, line, (0, i*15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
                out.write(img)
            out.release()
    else:
        font = ImageFont.load_default()
        img = Image.new('RGB', (len(art.split('\n')[0]) * 10, len(art.split('\n')) * 15), color='white')
        d = ImageDraw.Draw(img)
        d.text((0, 0), art, fill='black', font=font)
        img.save(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASCII art generator")
    parser.add_argument("--file-path", required=True, help="Path to the image, video, GIF or URL")
    parser.add_argument("--scale-factor", required=False, help="Scale factor for the image (default = 1)", default=1)
    parser.add_argument("--contrast", required=False, help="Contrast factor for the image (default = 1)", default=1)
    parser.add_argument("--fps", required=False, help="Frames per second for animation (default = 60)", default=60)
    parser.add_argument("--save", action="store_true", help="Save the ASCII output")
    args = parser.parse_args()

    art, file_format = generate_ascii_art(args.file_path, args.scale_factor, args.contrast)

    if args.save:
        output_path = f"ascii_output.{file_format}"
        save_ascii_art(art, file_format, output_path)
        print(f"ASCII art saved to {output_path}")

    # Handle both single image and list of frames
    if isinstance(art, list):
        animate_frames(art, int(args.fps))
    else:
        print(art)