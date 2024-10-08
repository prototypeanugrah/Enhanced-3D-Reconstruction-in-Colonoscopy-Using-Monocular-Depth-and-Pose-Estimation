# Video Depth Estimation for Hyper-Kvasir Dataset

## Project Overview

This project focuses on applying depth estimation techniques to endoscopic videos from the Hyper-Kvasir dataset. It uses the DepthAnythingV2 model to generate depth maps for each frame of the video, providing valuable insights into the spatial structure of the gastrointestinal tract.

## Dataset

The project uses the Hyper-Kvasir dataset, which is a comprehensive collection of gastrointestinal endoscopy images and videos. The dataset is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) License.


## Features

- Video processing: Extracts frames from .avi video files in the Hyper-Kvasir dataset.
- Depth estimation: Applies the DepthAnythingV2 model to generate depth maps for each frame.
- Visualization: Converts depth maps to heatmaps for better visualization.
- Output: Generates a new video with depth heatmaps.

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download the DepthAnythingV2 model checkpoints and place them in the `checkpoints` directory.

## Usage

Run the main script to process a video:
  ```
  python main.py
  ```

This script will:
1. Load the DepthAnythingV2 model
2. Process the first .avi video found in the "hyper-kvasir-videos/videos" directory
3. Generate depth maps for each frame
4. Create a new video with depth heatmaps

## Code Structure

- `main.py`: The main script that orchestrates the video processing pipeline.
- `data_processing.py`: Contains functions for video handling and data processing.
- `model_processing.py`: Handles the loading and application of the DepthAnythingV2 model.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

- The Hyper-Kvasir dataset: [Hyper-Kvasir](https://osf.io/mkzcq/)
- DepthAnythingV2 model: [DepthAnythingV2](https://huggingface.co/depth-anything)

## Contributing

Contributions to this project are welcome. Please feel free to submit a Pull Request.

## Contact

For any questions or concerns, please open an issue in this repository.
