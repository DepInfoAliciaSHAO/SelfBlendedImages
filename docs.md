# Changes

## Inference
- Separation of cropping and actual inference in two scripts:  
    - `datasets.py` stores initialization of datasets, when adding a dataset, add it to `init_dataset` by creating a initialize function specific to the new dataset. This function returns three things: the list of the path of all images, the list of corresponding labels (0: pristine, 1: fake) and `video_root`, usually the name of the dataset to keep folders organized in further processes.  
    - `inference_cropping.py` crops and stores cropping data (faces as numpy arrays and indexes of image??) in a pickle file at `CROP_DIR/video_root` for a dataset defined in `dataset.py`.
    - `inference_datasets.py` does inferences on a dataset defined in `datasets.py`. Can plot ROC and APCER/BPCER curves. Can plot the confusion matrix.

> **NOTE:** Don't use the other inference scripts without checking if final_transforms are applied before passing the images to the model first. If needed, define final_transforms with `get_final_transforms` from `sbi.py`. All models from model called transforms use transforms, models trained before don't use transforms.

---

## Preprocess
Ask Alvaro for the script with joint dlib and yunet processes. **DO NOT** use `crop_yunet.py`.

---

## Training

### Config shenanigans
- Most things added are modular, configurable in `src/configs/sbi/base.json`
- Current options:
    - `epoch`: max epoch for training
    - `batch_size`: batch_size for training and validation datasets (training dataset actually uses `batch_size//2` (half real, half self-blended fake))
    - `image_size`: size of image (`image_size x image_size`) fed to the model during training and validation, change when using different EfficientNet backbones
    - `use_wandb`: 0 (False) or 1 (True), enables use of wandb
    - `lr`: base learning rate at the beginning, when using freezing, a new learning rate is hardcoded in `train_sbi.py` for now, beware.
    - `weighted_sampler`: 0/1, uses weighted sampling to balance datasets in training dataloader
    - `train_datasets`: list of datasets (in string) to be used in training data. Check available datasets in `sbi.py`, in `SBI_Custom_Dataset`'s `init_dataset` method.
    - `val_datasets`: same thing for validation
    - `test_datasets`: same thing for test. Current available datasets CDF (Celeb-DF-v2) and FF (FaceForensics++)
    - `crop_mode_ff`: FaceForensics++ train and val splits have been cropped once with retina (SBI) and once with yunet (ShareID's model), choose here which type of crops to use.
    - `crop_mode_test`: list of crop modes for test datasets. Keep the same order as `test_datasets`.
    - `test_every`: number of epochs between each automated test. Only tests if a new best model was produced.
    - `degradations`: 0/1, use degradation module during self-blending from the Practial Manipulation Model (PMM)
    - `poisson`: 0/1: enable 50% of blending to use poisson blending instead of regular alpha blending (from PMM).
    - `random_mask`: 0/1, enable new type of partial masks.
    - `freeze`: int, number of epochs at the beginning of training during which all layers but the last fully connected layer are frozen. Unfreezes afterwards. 0 means there is no freezing. Only works for EfficienNet backbones.
    - `lr_scheduler`: type of learning rate scheduler to use. New schedulers can be added in `train_sbi.py`
        - LAA-Net uses LAA-Net's LinearDecay
        - SBI uses SBI's LinearDecay
        - Cosine uses a FlatCosineAnnealingLR
        - PMM uses PMM paper's scheduler: ReduceLROnPlateau on validation loss with patience 10.
    - `adam`: 0/1: 1 uses AdamW as an optimizer or (0) SAM with SGD
    - `backbone`: backbone used during training. Should match the one in `weight_path` if training from checkpoint. format: efficientet-bx with x from 0 to 7.
    - `weight_path`: if left blank "", nothing happens, if not, loads the weights and continues training from aforementioned checkpoint.

---

### Important notes, I guess
- Added `hull_type` return during blending
- Training datasets' paths are hardcoded in `utils/initialize.py`. Follow template to load additional new data.
- Training uses `SBI_Custom_Dataset` which overwrites `SBI_Dataset`'s `__get_item__` function!
- Only difference between the two `__get_item__` is the check for yunet crops. 
- Degradations are done in the `__get_item__` and defined in `degradations.py`
- Custom schedulers are defined in `utils/scheduler.py`
- GradCam visualization can be obtaine with `gradcam.py`, final_transforms are applied.
- `gradcam_video.py` has not been fully implemented and is not functional
- `save_sample_images.py` can be used to visualize blending process. As it is, it saves 300 images from the validatio dataloader in a folder with name of original dataset and label.
- `txt_files_face_machine` are used on face_machine to retrieve the list of videos for training without actually storing any videos. Suboptimal solution, FF++ has been transfered, other datasets are yet to be transferred.
- Weights are either stored in `weights/` or in `output/`
- Change environment when using yunet!