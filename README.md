# ASCII Converter CLI Tool

Convert images, GIFs, and videos to ASCII art using this command-line tool.

## Features

- Convert static images (JPG, JPEG, PNG, JFIF) to ASCII art
- Convert GIFs and videos (MP4) to ASCII art animations
- Support for loading images from URLs
- Adjustable scale factor and contrast
- Save output as text, image, GIF, or MP4

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/peeyuzz/ascii-converter-cli.git
   cd ascii-converter-cli
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Ensure you have FFmpeg installed on your system for video processing.

## Usage

```
python ascii_converter.py [-h] [--scale-factor SCALE_FACTOR] [--contrast CONTRAST] [--fps FPS] [--save] [--output OUTPUT] [--version] file_path
```

### Arguments

- `file_path`: Path to the input image, video, GIF, or URL
- `--scale-factor`: Scale factor for the image (default: 1)
- `--contrast`: Contrast factor for the image (default: 1)
- `--fps`: Frames per second for animation (default: 60)
- `--save`: Save the ASCII output
- `--output`: Output file path (required if --save is used)
- `--version`: Show the version number and exit

### Examples

1. Convert an image to ASCII and display it:
   ```
   python ascii_converter.py path/to/image.jpg
   ```

2. Convert an image to ASCII, adjust scale and contrast, and save the output:
   ```
   python ascii_converter.py path/to/image.jpg --scale-factor 0.5 --contrast 1.2 --save --output output.txt
   ```

3. Convert a video to ASCII animation and save as MP4:
   ```
   python ascii_converter.py path/to/video.mp4 --scale-factor 0.3 --fps 30 --save --output output.mp4
   ```

4. Convert a GIF to ASCII animation and save as GIF:
   ```
   python ascii_converter.py path/to/animation.gif --scale-factor 0.5 --save --output output.gif
   ```

5. Convert an image from a URL:
   ```
   python ascii_converter.py https://example.com/image.jpg --save --output output.txt
   ```

## Requirements

- Python 3.6+
- Pillow
- NumPy
- OpenCV
- tqdm
- FFmpeg (for video processing)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

If you encounter any problems or have any questions, please open an issue in the GitHub repository.