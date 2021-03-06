# Crowd-Counting

## Introduction

This folder contains the code of for crowd counting performed on JHU dataset, mall dataset and sample of other images with respect to Fast R-CNN. 



## Requirements

- tensorflow
- progressbar2
- numpy
- matplotlib
- cv2



## Note

- Model must be downloaded and added to the "/data/utils/" folder. Link of the model: [Model link](https://drive.google.com/drive/folders/14YzYPYbhV8tCwbYUa3LwTvGBtmq3Z_0d?usp=sharing)
- Datasets needs to be downloaded and added to their respective folders to "/data/images/" folder
  - [Mall datasat](https://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html) [1]
  - [JHU-Crowd++ dataset](http://www.crowd-counting.com/) [2]
- main_jhu.py and main_jhu.sh contains code to perform crowd counting on JHU dataset.
- main_mall.py and main_mall.sh contains code to perform crowd counting on Mall dataset.
- main_sample.py and main_sample.sh contains code to perform crowd counting on sample images.

- Results of JHU dataset is stored in jhu_output file, Mall dataset is stored in mall_output and sample images are stored in sample_results.

## Run
- Run main_jhu.py, main_mall.py and main_sample.py to run this project.
- Files of format .sh were used to run the code on server hence address URL links (file location links has to be changed accordingly).

## Dataset References
[1] Cumulative Attribute Space for Age and Crowd Density EstimationK. Chen, S. Gong, T. Xiang, and C. C. Loyin Proceedings of IEEE Conference on Computer Vision and Pattern Recognition, 2013.

[2] V. Sindagi, R. Yasarla and V. Patel, "Pushing the Frontiers of Unconstrained Crowd Counting: New Dataset and Benchmark Method," 2019 IEEE/CVF International Conference on Computer Vision (ICCV), 2019, pp. 1221-1231, doi: 10.1109/ICCV.2019.00131.
