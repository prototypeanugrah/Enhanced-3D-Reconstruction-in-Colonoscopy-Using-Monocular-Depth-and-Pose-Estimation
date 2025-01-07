# Train the model
# Dataset options: c3vd, simcol
python main_lightning.py ++dataset.batch_size=12 dataset=c3vd model=large ++trainer.devices=[1] ++model.encoder_lr=5e-6 ++model.decoder_lr=5e-5 ++trainer.max_epochs=20

# Test the model
# Dataset options: c3vd, simcol
python test_lightning.py dataset=simcol checkpoint_path="./checkpoints/mvitl_l1e-05_b20_e5_dc3vd/depth-any-endoscopy-epoch04-val_loss23.94.ckpt" trainer.devices=[0] ++dataset.batch_size=20

# Generate the predicted depth maps and npy files for a given model
# Encoder options: vitl, vitb, vits
# Max depth config: c3vd(10), simcol(20)
python run.py --encoder vitl --load-from "/home/public/avaishna/Endoscopy-3D-Modeling/checkpoints/simcol/mvitl_el5e-06_dl5e-05_b6_e30_dsimcol_p0.05/depth_any_endoscopy_epoch=29_val_loss=0.02.ckpt" --max-depth 20 -i datasets/SyntheticColon -d testing --pred-only --grayscale

