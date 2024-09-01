# ASCII Art Generator

Welcome to the ASCII Art Generator! This Python script converts images, GIFs, and videos into ASCII art. You can customize the output with various parameters such as scale factor, contrast, and frame rate for animations. This README will guide you through installation, usage, and understanding the code.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Parameters](#parameters)
- [Code Overview](#code-overview)
- [License](#license)

## Features

- Convert images (JPEG, PNG) to ASCII art.
- Convert GIFs and MP4 videos to ASCII art frames.
- Customize output with scale factor and contrast adjustments.
- Save ASCII art as an image or animated GIF/MP4.
- Display ASCII art in the console.

## Requirements

To run this script, you need to have Python 3.x installed along with the following packages:

- `Pillow` for image processing
- `NumPy` for numerical operations
- `OpenCV` for video processing

You can install the required packages using pip:

```bash
pip install pillow numpy opencv-python
```

## Installation

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/yourusername/ascii-art-generator.git
   cd ascii-art-generator
   ```

2. Ensure that all dependencies are installed as mentioned above.

## Usage

To use the ASCII Art Generator, run the script from the command line with the required parameters. Here’s the basic command structure:

```bash
python ascii_art_generator.py --file-path <path_to_image_or_video> [options]
```

### Example Commands

- Convert an image to ASCII art:

  ```bash
  python ascii_art_generator.py --file-path path/to/image.jpg
  ```

- Convert a GIF to ASCII art and save it:

  ```bash
  python ascii_art_generator.py --file-path path/to/animation.gif --save
  ```

- Convert a video to ASCII art with a specified frame rate:

  ```bash
  python ascii_art_generator.py --file-path path/to/video.mp4 --fps 30 --save
  ```

## Parameters

| Parameter       | Description                                          | Default Value |
|------------------|------------------------------------------------------|---------------|
| `--file-path`    | Path to the image, GIF, video, or URL.              | Required      |
| `--scale-factor`  | Scale factor for the ASCII art output.               | 1             |
| `--contrast`      | Contrast factor for the image processing.            | 1             |
| `--fps`           | Frames per second for animated output.               | 60            |
| `--save`          | Flag to save the ASCII art output to a file.        | False         |

## Code Overview

### Key Functions

- **`clear_console()`**: Clears the console for a clean display of ASCII art frames.
  
- **`generate_ascii_art(file_path, scale_factor=1, contrast=1)`**: Main function that handles image, GIF, and video inputs, returning ASCII art.

- **`process_frame(frame, scale_factor, contrast)`**: Processes individual video frames into ASCII art.

- **`frame_to_ascii(frame, scale_factor=1, contrast=1)`**: Converts a single video frame to ASCII art.

- **`get_ascii(image, scale_factor=1, contrast=1)`**: Converts an image to ASCII art.

- **`animate_frames(frames, fps=60)`**: Displays animated ASCII art frames in the console.

- **`save_ascii_art(art, file_format, output_path)`**: Saves the ASCII art output as an image or video file.

### Main Execution

The script's entry point is defined in the `if __name__ == "__main__":` block, which parses command-line arguments and initiates the ASCII art generation process.

## License

This project is licensed under the MIT License. Feel free to use, modify, and distribute this code as per the license terms.

---

Thank you for using the ASCII Art Generator! If you have any questions or suggestions, please feel free to open an issue in the repository. Happy coding!