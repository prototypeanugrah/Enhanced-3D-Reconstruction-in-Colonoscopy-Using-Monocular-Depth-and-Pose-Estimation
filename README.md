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
- Evaluation: Includes metrics for evaluating the performance of the depth estimation model.


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
   python train.py
   --input_dir <input_dir>
   --output_dir <output_dir>
   --model_size <model_size>
   --epochs <num_epochs>
   --batch_size <batch_size>
   --learning_rate <learning_rate>
   -s <use_scheduler>
   -w <warmup_steps>
   -ld <log_dir>
   ```

Arguments:
- `--input_dir`: Path to the directory containing input videos
- `--output_dir`: Path to save the fine-tuned model
- `--model_size`: Size of the DepthAnythingV2 model (small, base, large)
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--learning_rate`: Learning rate for training
- `-s`, `--use_scheduler`: Whether to use a learning rate scheduler (true/false)
- `-w`, `--warmup_steps`: Number of warmup steps for the scheduler
- `-ld`, `--logdir`: Directory for TensorBoard logs

To run multiple experiments with different configurations:
   ```
   bash run_experiments.sh
   ```

## Code Structure

- `train.py`: Main script for fine-tuning the DepthAnythingV2 model
- `training/training_utils.py`: Utility functions and classes for training
- `data_processing/dataloader.py`: Functions for creating data loaders
- `data_processing/dataset.py`: Custom dataset class
- `data_processing/convert_avi_to_mp4.py`: Script for converting AVI files to MP4
- `eval/evaluation.py`: Functions for computing evaluation metrics
- `utils/utils.py`: Utility functions for data processing and visualization
- `run_experiments.sh`: Bash script for running multiple training experiments

## Dataset

The challenge consisted of simulated colonoscopy data and images from real patients. This data release encompasses the synthetic portion of the challenge. The synthetic data includes three different anatomies derived from real human CT scans. Each anatomy provides several randomly generated trajectories with RGB renderings, camera intrinsics, ground truth depths, and ground truth poses. In total, this dataset includes more than 37,000 labelled images. 

The real colonoscopy data used in the SimCol3D challenge consists of images extracted from the EndoMapper dataset.
This dataset and accompanying files are licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0). https://creativecommons.org/licenses/by-nc-sa/4.0/

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

<!-- - The Hyper-Kvasir dataset: [Hyper-Kvasir](https://osf.io/mkzcq/) -->
- SimCol3D dataset: [SimCol3D](https://rdr.ucl.ac.uk/articles/dataset/Simcol3D_-_3D_Reconstruction_during_Colonoscopy_Challenge_Dataset/24077763?file=42248541)
- DepthAnythingV2 model: [DepthAnythingV2](https://huggingface.co/depth-anything)

## Contributing

Contributions to this project are welcome. Please feel free to submit a Pull Request.

## Contact

For any questions or concerns, please open an issue in this repository.
