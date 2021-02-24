# Deep-Learning-Practical1: Comparison of different CNN architectures, regularization and optimization strategies for the task of image classification.


## Project Description
In this practical we compare several convolutional neural network (CNN) architectures in regards to their classification performance on the CIFAR-10 image data set. More specifically, we compare a simplistic CNN as a baseline to the established AlexNet and VGGNet architectures. For all network architectures we also experiment with several different network attributes, such as the choice of regularization, activation function or optimization method and discuss their impact on classification performance.

## Network Architectures
- **AlexNet**:<br/>
<img src="images/Alexnet architecture.png" width="800"><br/>

- **Baseline Model**:<br/>
<img src="images/baselne cnn.png" width="400"><br/>

- **VGGNet**:<br/>
<img src="images/VGG net architecture.png" width="700"><br/>

## How to run the python files
- Install Python from [python.org](https://www.python.org/)
- `python -m pip install --upgrade pip`
- `pip install -r requirements.txt`


## Results
### Baseline Model
<img src="images/baseline cnn loss.png" width="300"> <img src="images/baseline cnn accuracy.png" width="300"><br/>

### VGG Net
<img src="images/VGGnet loss.png" width="300" > <img src="images/vggnet accuracy.png" width="300" ><br/>

- These are preliminary results to determine suitable number of epochs for the model. For a full description of the results of all architectures and hyperparameters tested in this project, refer to the report. 

## Authors
- [Brown Ogum](https://github.com/brown532)
- [Lars Cordes](https://github.com/L-Cordes)

## Reference
For this project, CIFAR-10 data set was used. We refer to the author's paper for more details: 
-Alex Krizhevsky.  Learning multiple layers of features from tiny images.University of Toronto, 052012