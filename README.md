# 3D Depth Estimation and Pose Estimation for Colonoscopy Procedures

## Project Overview

This project focuses on applying depth estimation, pose estimation and 3D reconstruction techniques to colonoscopy videos using synthetic and real patient data. It leverages the DepthAnythingV2 model to generate depth maps for each frame, providing valuable insights into the spatial structure of the colon during endoscopic procedures. The system also includes pose estimation capabilities to enable 3D reconstruction of the colon from consecutive frames.

<img width="427" alt="image" src="https://github.com/user-attachments/assets/f2142902-e90e-4fac-9dd2-aa07dbebae51" />
<img width="848" alt="image" src="https://github.com/user-attachments/assets/184cc81f-fc57-421c-b92d-7dde5d91c108" />
<img width="426" alt="image" src="https://github.com/user-attachments/assets/1b5fd85c-ba05-4670-9284-59f0714d091f" />



## Dataset

**Synthetic Colon Dataset (SimCol3D)**:
   - Simulated colonoscopy data derived from real human CT scans
   - Includes three different anatomies with randomly generated trajectories
   - Provides RGB renderings, camera intrinsics, ground truth depths, and ground truth poses
   - Contains over 37,000 labeled images


## Features

- Video processing: Processes frames from synthetic and real colonoscopy videos
- Depth estimation: Applies the DepthAnythingV2 model for accurate depth map generation
- Pose estimation: Estimates camera pose between consecutive frames using a modified ResNet-18 architecture
- 3D Reconstruction: Combines depth maps and pose estimates to create 3D models of the colon
- Visualization: Converts depth maps to heatmaps and creates overlay videos
- Dataset splitting: Automatically splits datasets into train, validation, and test sets
- Training: Supports fine-tuning of DepthAnythingV2 using PyTorch Lightning
- Evaluation: Includes comprehensive metrics for depth estimation and pose estimation performance
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
│   ├── SyntheticColon_*/
│     ├── Frames_*/
│        ├── Depth_*.png/
│        └── FrameBuffer_*.png/
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
```

Each .txt file should contain paths to the corresponding images and their depth maps in the same folder.

### Training

The project uses PyTorch Lightning for training. The main training scripts are:

1. For depth estimation:
   ```
   python main_lightning.py \
    ++dataset.batch_size=<batch_size> \
    dataset=<simcol|c3vd> \
    model=<small|base|large> \
    ++trainer.devices=[<gpu_ids>] \
    ++model.encoder_lr=<encoder_learning_rate> \
    ++model.decoder_lr=<decoder_learning_rate> \
    ++trainer.max_epochs=<epochs>
   ```

2. For pose estimation:
   ```
   python pose_estimation_lightning.py \
    dataset.batch_size=32 \
    trainer.devices=[0] \
    model.lr=1e-4 \
    trainer.max_epochs=100
   ```

#### Training Configuration

The training setup includes:
- Model: DepthAnythingV2 with configurable encoder sizes
- Loss: MSE Loss or SiLog Loss for depth estimation
- Pose Loss: Combined translation and quaternion loss for pose estimation
- Optimizer: AdamW with different learning rates for encoder and decoder
- Learning Rate Scheduler: OneCycleLR with warm-up
- Mixed Precision Training: 16-bit mixed precision
- Early Stopping: Monitors validation loss with 5 epochs patience
- Model Checkpointing: Saves best model based on validation loss

#### Model Sizes
- Small (vits): 48-384 channel features
- Base (vitb): 96-768 channel features
- Large (vitl): 256-1024 channel features
- Giant (vitg): 1536 channel features

#### Key Parameters
- `dataset.batch_size`: Number of images per batch
- 'dataset': The dataset to be used. Choices: simcol | c3vd
- `model`: Model size (small, base, large)
- `trainer.devices`: List of GPU devices to use
- `model.encoder_lr`: Base encoder learning rate
- `model.decoder_lr`: Base learning rate (decoder uses 10x the encoder value as per the original [DepthAnythingV2](https://arxiv.org/pdf/2406.09414) paper)
- `trainer.max_epochs`: Maximum number of training epochs

### Testing

To evaluate a trained model:
   ```
   python test_lightning.py \
   dataset=<simcol|c3vd> \
   checkpoint_path=<path_to_checkpoint> \
   trainer.devices=[<gpu_ids>] \
   ++dataset.batch_size=<batch_size>
   ```

### Depth Map Generation:

To generate depth maps for images or videos:
   ```
   python run.py \
   --encoder <vitl|vitb|vits> \
   --load-from <checkpoint_path> \
   --max-depth <depth> \
   -i <input_path> \
   -d testing \
   --pred-only \
   --grayscale
   ```

### 3D Reconstruction

The system supports 3D reconstruction through:
1. Depth map generation for each frame
2. Pose estimation between consecutive frames
3. Point cloud generation from depth maps
4. Trajectory reconstruction using estimated poses

#### Monitoring
Training progress can be monitored through:
- WandB logging
- Saved checkpoints in `checkpoints/<ds_type>/<experiment_id>/`

#### Metrics
The model tracks several metrics during training and validation:
- Depth Estimation Metrics:
  - Loss: MSE loss between predicted and ground truth depth
  - D1: Delta 1 accuracy
  - Abs Rel: Absolute relative error
  - RMSE: Root mean square error
  - L1: L1 error

- Pose Estimation Metrics:
  - ATE: Absolute Translation Error
  - RTE: Relative Translation Error
  - ROTE: Rotation Error

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

- SimCol3D dataset: [SimCol3D](https://rdr.ucl.ac.uk/articles/dataset/Simcol3D_-_3D_Reconstruction_during_Colonoscopy_Challenge_Dataset/24077763?file=42248541)
- DepthAnythingV2 model: [DepthAnythingV2](https://huggingface.co/depth-anything)

## Contributing

Contributions to this project are welcome. Please feel free to submit a Pull Request.

## Contact

For any questions or concerns, please open an issue in this repository.
