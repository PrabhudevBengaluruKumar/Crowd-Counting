# Crowd Counting

Crowd counting is estimating the number of people present in a video or picture. It helps perform surveillance or to count people attending a social event. Manual crowd counting is a complex task that is also time-consuming. Computer vision techniques can solve this problem by automating this counting task. In this project, we aim to perform crowd computing by making use of deep learning approaches. We use Convolutional Neural Networks (CNN) to approach this problem. As the number of people in a picture increases, the accuracy of manual crowd counting decreases. So, in addition to the traditional CNN-based object detectors, some state-of-the-art techniques like density mapping and encoder-decoder models are used to increase the model performance. The performance metrics of the models will be compared, and inferences will be drawn.

## The different crowd counting techniques are:
1. Detection based methods.

	A shaped window-like detector identifies people in a photo or video using different classification algorithms, and the number of people is counted. In order to extract low-level features, classifiers must be well trained.
The algorithms used to detect faces in pictures and videos work well when there is a dense crowd present, but fail to give satisfactory results when there are dense crowds present. When working with dense crowds, it is difficult to distinguish and/or see the target features clearly.

2. Regression based methods

	The method of counting by detection cannot be used well when there is a dense crowd and there is a high level of background noise or clutter. In addition, regression-based methods can remove low-level features, which overcomes those challenges.
The image is cropped into patches, and then these patches are used to extract the low level features, such as edges, foreground pixels, etc. By using regression methods, the input images can be directly mapped to scalar values. However, these methods do not accurately reflect crowd distributions, which is overcome with density-based methods which perform pixel-wise regressions to improve model performance.

3. Density estimation based methods
  
	In order to be able to localize a crowd, density-based estimation methods are used. A density map is first created for each of the objects. By traversing the images while learning the mapping between local features and object density maps, the approach focuses on the density and localization of crowded spaces. By concatenating discrete object density patches, the full density map of the overall data can be obtained. Learning non-linear maps can be accomplished using a random forest regressor.

4. CNN based methods

	A CNN based method is the most reliable method to achieve better accuracy compared to the other conventional methods. CNNs are used specifically in computer vision to deal with crowd density problems.

	The CrowdNet CNN has the capability of capturing both low-level and high-level features of an image. In other words, it is a combination of deep and shallow Convolutional Network frameworks. To overcome the limitations of other counting methods, we augment the dataset with scale-invariant representations. Dense crowd counting is handled through this method, which returns density maps based on high-level semantics.

	The CSRNet technique, which is used in Deep Convolutional Networks and which we are about to implement here, is used the most often when counting problems arise. The model extracts high-level features and generates high-quality density maps without increasing the complexity of the network. The front end of CSRNet is implemented using the VGG-16 technique because of its better transfer learning rate. 1/pth size of the original input size is the output size of a VGG. CSRNet also uses CSRNet-like layers of dilated convolutions.
  
## Datasets
Two datasets are used in this experiment namely, Mall dataset and JHU-Crowd++ dataset.

1. Mall dataset
- 480x640 pixels
- 2000 images
- Min 15 persons - max 53 persons per image
 
Reference: Loy, Chen Change, et al. "Crowd counting and profiling: Methodology and evaluation." Modeling, simulation and visual analysis of crowds. Springer, New York, NY, 2013. 347-382.

2. JHU-CROWD++ dataset
- Diverse conditions
- 4,372 images
- Rich set of annotations
- Min 50 persons - max 25791 persons per image

Reference: V. A. Sindagi, R. Yasarla, en V. M. Patel, “JHU-CROWD++: Large-Scale Crowd Counting Dataset and A Benchmark Method”, Technical Report, 2020.

## Directory details
1. Faster R-CNN folder: contains the details and code for crowd counting using detection-based method (Faster R-CNN).
2. CSRNet folder: contains the details and code for crowd counting using CNN-based method (CSRNet).

## Refered Resources  
[1] V. A. Sindagi, R. Yasarla, and V. M. Patel, “Jhu-crowd++: Large-scale crowd counting dataset and a benchmark method,” Technical Report, 2020.  
[2] Building crowd counting model with python  
https://www.analyticsvidhya.com/blog/2019/02/building-crowd-counting-model-python/  
[3] Loy, Chen Change, et al. "Crowd counting and profiling: Methodology and evaluation." Modeling, simulation and visual analysis of crowds. Springer, New York, NY, 2013. 347-382.  
[4] N. Akbar and E. C. Djamal, "Crowd Counting Using Region Convolutional Neural Networks," 2021 8th International Conference on Electrical Engineering, Computer Science and Informatics (EECSI), 2021, pp. 359-364, doi: 10.23919/EECSI53397.2021.9624288.  
[5] V. A. Sindagi, R. Yasarla, en V. M. Patel, “Pushing the Frontiers of Unconstrained Crowd Counting: New Dataset and Benchmark Method”, arXiv [cs.CV]. 2019.  
[6] Y. Li, X. Zhang, en D. Chen, “CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes”, arXiv [cs.CV]. 2018.  
[7] Li, Y., Zhang, X., & Chen, D. (2018). CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes. 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, 1091-1100.  
[8] Crowd counting using deep learning methods    
https://www.analyticsvidhya.com/blog/2021/06/crowd-counting-using-deep-learning/  
[9] Awesome crowd counting  
 https://github.com/gjy3035/Awesome-Crowd-Counting.  
[10] Boominathan, Lokesh, Srinivas SS Kruthiventi, and R. Venkatesh Babu. "Crowdnet: A deep convolutional network for dense crowd counting." Proceedings of the 24th ACM international conference on Multimedia. 2016.  
[11] Wang, Qi, et al. "NWPU-crowd: A large-scale benchmark for crowd counting and localization." IEEE transactions on pattern analysis and machine intelligence 43.6 (2020): 2141-2149.  
[12] Crowd counting  
https://github.com/darpan-jain/crowd-counting-using-tensorflow  
[13] Gao, Guangshuai, et al. "Cnn-based density estimation and crowd counting: A survey." arXiv preprint arXiv:2003.12783 (2020).  
[14] Cao, X., Wang, Z., Zhao, Y., & Su, F. (2018). Scale aggregation network for accurate and efficient crowd counting. Computer Vision – ECCV 2018, 757-773. https://doi.org/10.1007/978-3-030-01228-1_45.  
[15] Ilyas, N.; Shahzad, A.; Kim, K. Convolutional-Neural Network-Based Image Crowd Counting: Review, Categorization, Analysis, and Performance Evaluation. Sensors 2020, 20, 43. https://doi.org/10.3390/s20010043.  
[16] Cumulative Attribute Space for Age and Crowd Density EstimationK. Chen, S. Gong, T. Xiang, and C. C. Loyin Proceedings of IEEE Conference on Computer Vision and Pattern Recognition, 2013.  
[17] V. Sindagi, R. Yasarla and V. Patel, "Pushing the Frontiers of Unconstrained Crowd Counting: New Dataset and Benchmark Method," 2019 IEEE/CVF International Conference on Computer Vision (ICCV), 2019, pp. 1221-1231, doi: 10.1109/ICCV.2019.00131. 
