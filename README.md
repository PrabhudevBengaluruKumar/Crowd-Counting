# Crowd Counting

Crowd counting is estimating the number of people present in a video or picture. It helps perform surveillance or to count people attending a social event. Manual crowd counting is a complex task that is also time-consuming. Computer vision techniques can solve this problem by automating this counting task. In this project, we aim to perform crowd computing by making use of deep learning approaches. We use Convolutional Neural Networks (CNN) to approach this problem. As the number of people in a picture increases, the accuracy of manual crowd counting decreases. So, in addition to the traditional CNN-based object detectors, some state-of-the-art techniques like density mapping and encoder-decoder models are used to increase the model performance. The performance metrics of the models will be compared, and inferences will be drawn.

The different crowd counting techniques are.
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


