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
        return Image.open(urlopen(file_path))
    elif file_path.endswith((".jpg", ".jpeg", ".png", ".gif", ".jfif")):
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

# def save_ascii_art(art, file_format, output_path):
#     """Saves ASCII art to an output file."""
#     font_path = r"c:\WINDOWS\Fonts\CONSOLA.TTF"
#     font = ImageFont.truetype(font_path, size=15)

#     if isinstance(art, list):
#         logging.info(f"Saving ASCII art as {file_format} to {output_path}")
#         if file_format == 'gif':
#             images = [convert_text_to_image(ascii_frame, font) for ascii_frame in art]
#             images[0].save(output_path, save_all=True, append_images=images[1:], duration=100, loop=0)
#         elif file_format == 'mp4':
#             height = len(art[0].split('\n')) * 20
#             width = len(art[0].split('\n')[0]) * 15
#             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#             out = cv2.VideoWriter(output_path, fourcc, 10.0, (width, height))
#             for ascii_frame in art:
#                 frame = convert_text_to_image(ascii_frame, font)
#                 out.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
#             out.release()
#     else:
#         convert_text_to_image(art, font).save(output_path)

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def save_ascii_art(art, file_format, output_path):
    """Saves ASCII art to an output file."""
    font_path = r"c:\WINDOWS\Fonts\CONSOLA.TTF"
    font = ImageFont.truetype(font_path, size=15)

    # Define maximum dimensions for MPEG-4
    max_width, max_height = 4096, 2304

    if isinstance(art, list):
        logging.info(f"Saving ASCII art as {file_format} to {output_path}")
        if file_format == 'gif':
            # Use ThreadPoolExecutor to parallelize the conversion of ASCII frames to images
            with ThreadPoolExecutor() as executor:
                futures = []
                # Submit tasks to convert ASCII frames to images in parallel
                for ascii_frame in art:
                    futures.append(executor.submit(convert_text_to_image, ascii_frame, font))

                # Collect the converted images with tqdm progress bar
                images = []
                for future in tqdm(futures, desc="Converting ASCII frames to images", unit="frame"):
                    images.append(future.result())

            # Save the images as an animated GIF
            images[0].save(output_path, save_all=True, append_images=images[1:], duration=17, loop=0)

            logging.info(f"GIF saved successfully to {output_path}")

        elif file_format == 'mp4':
            # Calculate the initial frame size
            height = len(art[0].split('\n')) * 15
            width = len(art[0].split('\n')[0]) * 7

            # Check and adjust dimensions if they exceed MPEG-4 limits
            if width > max_width or height > max_height:
                width, height = resize_dimensions(width, height, max_width, max_height)

            # Use H.264 codec for better compression
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            # out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))  # Reduced FPS to 30 for smaller size
            fps = 30.0
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)
            out.set(cv2.VIDEOWRITER_PROP_QUALITY, 40)

            # Use ThreadPoolExecutor to parallelize frame processing
            with ThreadPoolExecutor() as executor:
                # Process and save frames with tqdm progress bar
                futures = []
                for ascii_frame in art:
                    futures.append(executor.submit(process_frame, ascii_frame, font, width, height))

                for future in tqdm(futures, desc="Saving frames to MP4", unit="frame"):
                    frame = future.result()
                    out.write(frame)

            out.release()

            # Use ffmpeg to further compress the output
            compressed_output_path = output_path.replace('.mp4', '_compressed.mp4')
            ffmpeg_command = f'ffmpeg -i {output_path} -vcodec libx264 -crf 28 {compressed_output_path}'
            os.system(ffmpeg_command)

        # elif file_format == 'mp4':
    #     # Calculate the initial frame size
    #     height = len(art[0].split('\n')) * 15
    #     width = len(art[0].split('\n')[0]) * 7

    #     # Check and adjust dimensions if they exceed MPEG-4 limits
    #     if width > max_width or height > max_height:
    #         width, height = resize_dimensions(width, height, max_width, max_height)

    #     fps = 30

    #     # FFmpeg command for compressed output
    #     ffmpeg_command = [
    #         'ffmpeg',
    #         '-y',  # Overwrite output file if it exists
    #         '-f', 'rawvideo',
    #         '-vcodec', 'rawvideo',
    #         '-s', f'{width}x{height}',
    #         '-pix_fmt', 'bgr24',
    #         '-r', str(fps),
    #         '-i', '-',  # Input from pipe
    #         '-c:v', 'libx264',
    #         '-preset', 'medium',  # Adjust preset for speed/compression trade-off
    #         '-crf', '23',  # Adjust CRF value for quality/size trade-off
    #         '-y',
    #         output_path
    #     ]

    #     # Start FFmpeg process
    #     process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE)

    #     def process_frame(ascii_frame, font, width, height):
    #         img = np.ones((height, width, 3), dtype=np.uint8) * 255
    #         font_scale = 0.4
    #         font_thickness = 1
    #         for i, line in enumerate(ascii_frame.split('\n')):
    #             cv2.putText(img, line, (0, (i+1)*15), font, font_scale, (0,0,0), font_thickness)
    #         return img

    #     font = cv2.FONT_HERSHEY_SIMPLEX

    #     # Use ThreadPoolExecutor to parallelize frame processing
    #     with ThreadPoolExecutor() as executor:
    #         futures = []
    #         for ascii_frame in art:
    #             futures.append(executor.submit(process_frame, ascii_frame, font, width, height))

    #         for future in tqdm(futures, desc="Processing and compressing frames", unit="frame"):
    #             frame = future.result()
    #             process.stdin.write(frame.tobytes())

    #     # Close FFmpeg process
    #     process.stdin.close()
    #     process.wait()

    #     if process.returncode != 0:
    #         print(f"Error during FFmpeg encoding. Return code: {process.returncode}")
    #     else:
    #         print(f"Compressed MP4 video saved successfully to {output_path}")

    #         # logging.info(f"Compressed MP4 video saved successfully to {compressed_output_path}")
    # else:
    #     # Convert single ASCII art to image and save
    #     img = convert_text_to_image(art, font)
    #     img.save(output_path)
    #     logging.info(f"ASCII art saved as image to {output_path}")

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
    
    # Resize frame if necessary
    if frame.size[0] > width or frame.size[1] > height:
        frame = frame.resize((width, height), Image.LANCZOS)
    
    # Convert PIL Image to OpenCV format
    return cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)


def convert_text_to_image(text, font):
    """Converts ASCII text to an image."""
    width = len(text.split('\n')[0]) * 7
    height = len(text.split('\n')) * 15
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
        file_format = args.file_path.split('.')[-1] if not args.file_path.startswith(("http://", "https://")) else "jpg"
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
        if args.save:
        # Determine the output path
            if args.file_path.startswith(("http://", "https://")):
                # If the input is a URL, save the file as 'asciied_' + URL in the current directory
                output_path = "asciied_" + args.file_path.replace("http://", "").replace("https://", "").replace("/", "_")
            else:
                # If the input is a local file, append 'asciied_' to the start of the file name
                output_path = os.path.join(
                    os.path.dirname(args.file_path), 
                    "asciied_" + os.path.basename(args.file_path)
                )
            
            save_ascii_art(ascii_art, file_format, output_path)
            logging.info(f"ASCII art saved to {output_path}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")



