# CSRNet - CNN-based method

This folder contains code for crowd counting using CSRNet dataset.

## Requirements
Library required for this experiment is:
- albumentations
- cv2
- matplotlib
- numpy
- pandas
- pillow
- pytorch lightning
- scipy
- torch
- torchvision



## steps for 

- download and extract JHU-Crowd++ dataset into `./jhu_crowd_v2.0` path

Training
```shell
python -m crowdnet.train
```

Test
```shell
python -m crowdnet.predict --model-path MODEL_PATH --img-path IMAGE_PATH
```

- Pre-trained model: [MODEL](https://drive.google.com/file/d/1YFxRZOiH3g5wOTj4vXCLxBSOqJknyuPk/view?usp=sharing)
