## Sodeep's Final Project

### Introduction
----------------
Convolutional neural network is a key architure to recognize and classify the image to classes. However, variation of convolutional neural networks are very many and very vary their space/time usage for learning.  
In this project, I compared the state-of-art convolutional neural networks and compared them based on **Performance** and **Resources**(space/time). Through this project, it helps to decide the model to solve problem based on problem size and limitation of resources.  

### Goal
--------
Compare four **Models**(*MobileNet v1*, *VGGNet-16*, *ResNet-50*, *InceptionV3*) in four **Datasets**(*cifar3*, *cifar10*, *cifar100*, *Intel*)  
for **Performance**(*Training Accuracy*, *Validation Accuracy*) and **Resources**(*# of parameters*, *learning time*)

### Experiment
--------------

#### Models
* [MobileNet v1](https://arxiv.org/pdf/1704.04861.pdf)
  * [arXiv](https://arxiv.org/abs/1704.04861)
  * [Simple explanation](https://deepmi.me/deeplearning/74/)
* [VGGNet-16](https://arxiv.org/pdf/1409.1556.pdf)
  * [arXiv](https://arxiv.org/abs/1409.1556)
  * [Simple explanation](https://m.blog.naver.com/PostView.nhn?blogId=laonple&logNo=220738560542&proxyReferer=https%3A%2F%2Fwww.google.com%2F)
* [ResNet-50](https://arxiv.org/pdf/1512.03385.pdf)
  * [arXiv](https://arxiv.org/abs/1512.03385)
  * [Simple explanation](https://m.blog.naver.com/PostView.nhn?blogId=laonple&logNo=221259295035&proxyReferer=https%3A%2F%2Fwww.google.com%2F)
* [InceptionV3](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf)
  * [InceptionV1](https://arxiv.org/pdf/1409.4842.pdf)

#### Data
* [Cifar 3](https://coursys.sfu.ca/2019sp-cmpt-880-g1/pages/Homework2_data.zip)
  * 32 x 32 x 3 (Width, Height, RGB-Channel)
  * Output classes 3 (Cat, Dog, Frog)

* [Cifar 10](https://www.cs.toronto.edu/~kriz/cifar.html)
  * 32 x 32 x 3 (Width, Height, RGB-Channel)
  * Output classes 10 (Airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)

* [Cifar 100](https://www.cs.toronto.edu/~kriz/cifar.html)
  * 32 x 32 x 3 (Width, Height, RGB-Channel)
  * Output classes 100 (too many, check in the page)

* [Intel classification image](https://www.kaggle.com/puneet6060/intel-image-classification)
  * 150 x 150 x 3 (Width, Height, RGB-Channel), However, a few images(~10) are not 150 x 150 
  * Output classes 6 (buildings, forest, glacier, mountain, sea, street)

#### Hyper-parameters for each Model
* Epoch
* Batch-size

### References
--------------
[1] [MobileNet: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861.pdf)  
[2] [Very Deep Convolutional Networks For Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf)  
[3] [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)  
[4] [Rethinking the Inception Architecture for Computer Vision](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf)  
[5] [Going deeper with convolutions](https://arxiv.org/pdf/1409.4842.pdf)  
[6] [Cimon Farser University CMPT 880 G1](https://coursys.sfu.ca/2019sp-cmpt-880-g1/pages/)