# Train the model
# Dataset options: c3vd, simcol
python main_lightning.py ++dataset.batch_size=12 dataset=c3vd model=large ++trainer.devices=[1] ++model.lr=5e-2 ++trainer.max_epochs=1

# Test the model
# Dataset options: c3vd, simcol
python test_lightning.py dataset=simcol checkpoint_path="./checkpoints/mvitl_l1e-05_b20_e5_dc3vd/depth-any-endoscopy-epoch04-val_loss23.94.ckpt" trainer.devices=[0] ++dataset.batch_size=20

# Generate the predicted depth maps and npy files for a given model
# Encoder options: vitl, vitb, vits
# Max depth config: c3vd(10), simcol(20)
python run.py --encoder vitl --load-from "./checkpoints/simcol/mvitl_l5e-06_b20_e30_dsimcol/depth-any-endoscopy-epoch=28-val_loss=0.01.ckpt" --max-depth 20 --img_path datasets/SyntheticColon/

