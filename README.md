# 3D Depth Estimation for Colonoscopy Procedures

## Project Overview

This project focuses on applying depth estimation techniques to colonoscopy videos using synthetic and real patient data. It leverages the DepthAnythingV2 model to generate depth maps for each frame, providing valuable insights into the spatial structure of the colon during endoscopic procedures.


## Dataset

1. **Synthetic Colon Dataset (SimCol3D)**:
   - Simulated colonoscopy data derived from real human CT scans
   - Includes three different anatomies with randomly generated trajectories
   - Provides RGB renderings, camera intrinsics, ground truth depths, and ground truth poses
   - Contains over 37,000 labeled images

2. **C3VD Dataset**:
   - Real colonoscopy data from the EndoMapper dataset
   - Licensed under CC BY-NC-SA 4.0

## Features

- Video processing: Processes frames from synthetic and real colonoscopy videos
- Depth estimation: Applies the DepthAnythingV2 model for accurate depth map generation
- Visualization: Converts depth maps to heatmaps and creates overlay videos
- Dataset splitting: Automatically splits datasets into train, validation, and test sets
- Training: Supports fine-tuning of DepthAnythingV2 using PyTorch Lightning
- Evaluation: Includes comprehensive metrics for depth estimation performance
- Point Cloud Generation: Converts depth maps to 3D point clouds for visualization


## Installation

1. Clone this repository:
   ```
   git clone https://github.com/prototypeanugrah/Endoscopy-3D-Modeling.git
   cd Endoscopy-3D-Modeling
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage


### Dataset Structure

The training script expects the following dataset structure:
```
datasets/
├── SyntheticColon/
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
└── C3VD/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

Each .txt file should contain paths to the corresponding images and their depth maps in the same folder.

### Training

The project uses PyTorch Lightning for training. The main training script is `main_lightning.py`, which supports both SimCol3D and C3VD datasets, with options to train on either dataset individually or combined.

To fine-tune the model:
   ```
   python main_lightning.py \
    ++dataset.batch_size=<batch_size> \
    model=<small|base|large> \
    ++trainer.devices=[<gpu_ids>] \
    ++model.lr=<learning_rate>
   ```

#### Training Configuration

The training setup includes:
- Model: DepthAnythingV2 with configurable encoder sizes
- Loss: MSE Loss
- Optimizer: AdamW with different learning rates for encoder and decoder
- Learning Rate Scheduler: OneCycleLR with warm-up
- Mixed Precision Training: 16-bit mixed precision
- Early Stopping: Monitors validation loss with 5 epochs patience
- Model Checkpointing: Saves top 3 models based on validation loss

#### Model Sizes
- Small (vits): 48-384 channel features
- Base (vitb): 96-768 channel features
- Large (vitl): 256-1024 channel features
- Giant (vitg): 1536 channel features

#### Key Parameters
- `dataset.batch_size`: Number of images per batch
- `model`: Model size (small, base, large)
- `trainer.devices`: List of GPU devices to use
- `model.lr`: Base learning rate (decoder uses 10x this value)
- `trainer.max_epochs`: Maximum number of training epochs

#### Monitoring
Training progress can be monitored through:
- WandB logging (when enabled)
- TensorBoard logs
- Saved checkpoints in `checkpoints/<experiment_id>/`

#### Metrics
The model tracks several metrics during training and validation:
- Loss: MSE loss between predicted and ground truth depth
- D1: Delta 1 accuracy
- Abs Rel: Absolute relative error
- RMSE: Root mean square error
- L1: L1 error

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

<!-- - The Hyper-Kvasir dataset: [Hyper-Kvasir](https://osf.io/mkzcq/) -->
- SimCol3D dataset: [SimCol3D](https://rdr.ucl.ac.uk/articles/dataset/Simcol3D_-_3D_Reconstruction_during_Colonoscopy_Challenge_Dataset/24077763?file=42248541)
- DepthAnythingV2 model: [DepthAnythingV2](https://huggingface.co/depth-anything)
- C3VD dataset: [C3VD](https://durrlab.github.io/C3VD/)

## Contributing

Contributions to this project are welcome. Please feel free to submit a Pull Request.

## Contact

For any questions or concerns, please open an issue in this repository.
