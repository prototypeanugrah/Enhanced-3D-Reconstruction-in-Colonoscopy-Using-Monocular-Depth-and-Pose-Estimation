# Video Depth Estimation for Hyper-Kvasir Dataset

## Project Overview

This project focuses on applying depth estimation techniques to endoscopic videos from the Hyper-Kvasir dataset. It uses the DepthAnythingV2 model to generate depth maps for each frame of the video, providing valuable insights into the spatial structure of the gastrointestinal tract.

## Dataset

The project uses the Hyper-Kvasir dataset, which is a comprehensive collection of gastrointestinal endoscopy images and videos. The dataset is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) License.

## Features

- Video processing: Extracts frames from .avi and .mp4 video files in the Hyper-Kvasir dataset.
- Depth estimation: Applies the DepthAnythingV2 model to generate depth maps for each frame.
- Visualization: Converts depth maps to heatmaps and creates overlay videos for better visualization.
- Dataset splitting: Automatically splits the dataset into train, validation, and test sets.
- Training: Supports fine-tuning of the DepthAnythingV2 model using PyTorch Lightning.

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/prototypeanugrah/Endoscopy-3D-Modeling.git
   cd your-repo-name
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Data Preparation

Split the dataset into train, validation, and test sets:
   ```
   python dataset_split.py
   ```

### Training

To fine-tune the model:
   ```
   python train.py --input_dir datasets/hyper-kvasir --output_dir output --model_size small --epochs 10 --batch_size 4 --learning_rate 1e-4
   ```

Arguments:
- `--input_dir`: Path to the directory containing input videos
- `--output_dir`: Path to save the fine-tuned model
- `--model_size`: Size of the DepthAnythingV2 model (small, base, large)
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--learning_rate`: Learning rate for training

### Inference

To process videos and generate depth maps:
   ```
   python main_lightning.py --input input_path --output output_path --model_size small --limit 10
   ```

Arguments:
- `--input`: Path to input video file or folder containing videos
- `--output`: Path to output directory
- `--model_size`: Size of the depth estimation model (small, base, large)
- `--limit`: Limit the number of videos to process (optional)

## Code Structure

- `main_lightning.py`: Main script for video processing and depth estimation
- `train.py`: Script for fine-tuning the DepthAnythingV2 model
- `dataset_split.py`: Script for splitting the dataset
- `custom_dataset.py`: Custom dataset class for video frames
- `data_processing.py`: Functions for video handling and data processing
- `model_processing.py`: Functions for loading and applying the DepthAnythingV2 model
- `lightning_model.py`: PyTorch Lightning module for the DepthAnythingV2 model

## Dataset

This project uses the Hyper-Kvasir dataset, a comprehensive collection of gastrointestinal endoscopy and videos. The dataset is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) License.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

- The Hyper-Kvasir dataset: [Hyper-Kvasir](https://osf.io/mkzcq/)
- DepthAnythingV2 model: [DepthAnythingV2](https://huggingface.co/depth-anything)

## Contributing

Contributions to this project are welcome. Please feel free to submit a Pull Request.

## Contact

For any questions or concerns, please open an issue in this repository.
