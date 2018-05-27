# **Traffic Sign Recognition**

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./my_examples/Arterial.jpg "Traffic Sign 1"
[image5]: ./my_examples/german-traffic-signs-picture-id459380917 "Traffic Sign 2"
[image6]: ./my_examples/Do-Not-Enter.jpg "Traffic Sign 3"
[image7]: ./my_examples/wegaanduiding-beurt-links-tegen-de-blauwe-hemel-stockfotografie_csp36280602.jpg "Traffic Sign 4"
[image8]: ./my_examples/Stop sign.jpg "Traffic Sign 5"
[image9]: ./writeup_pic/training_set.png "training set"
[image10]: ./writeup_pic/valid_set.png "valid set"
[image11]: ./writeup_pic/test_set.png "test set"
[image12]: ./writeup_pic/no_hist_eq.png "Before hist eq"
[image13]: ./writeup_pic/hist_eq.png "After hist eq"
[image14]: ./writeup_pic/random_grey_hist.png#1  "grey + hist"
[image15]: ./writeup_pic/faked.png#1  "faked"
[image16]: ./writeup_pic/turn_left_ahead.png#2  "turn_left_ahead"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* Number of training examples = 34799
* Number of testing examples = 12630
* Number of validation examples = 4410
* Image data shape = (32, 32, 3)
* Number of classes = 43
#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed over classes.

![training data][image9]
![validation data][image10]
![test data][image11]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because the colors on the pictures are not reliable due to weather, light conditions.

I also decided to apply the histogram equalization, since some dark or bright images are hard to interpret.
Here the example of dark image before and after histogram equalization.

![Before histogram equalization][image12]
![After histogram equalization][image13]

As a last step, I normalized the image data to have 0 mean and standard deviation 1. I also reshaped the data to fit it into input of Lenet CNN (n,32,32,1).
```python
X_norm = (X_greyscale -128.0) /255.0
```

Here is an example of a traffic sign image before and after grayscaling and after grayscaling with histogram equalization.


![greyscaling + hist_eq][image14]

I decided to generate additional data because I tried the lenet and it was underfitting, so I thought to add faked data.

To add more data to the the data set, I used the following techniques like rotation on +/- 15 degrees, zooming (0.9 -1.1) and adding normal noise.

Here is an example of an original image and an augmented images:

![alt text][image15]

So I randomly picked either I rotate/zoom/add noise 5 times on the training set + original training set, meaning my training set was 6x times bigger the initial one.



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 RGB image   							|
| Convolution layer 1 28x28     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|	relu of convolution layer 1 1											|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution layer 2 5x5x6x16      	| 1x1 stride, outputs  10x10x16				|
| RELU      	| relu of convolution layer 2 				|
| Pooling      	| 2x2 stride, valid padding			|
| Fully connected    	| 400x200	output 120	|
| RELU      	| relu of fully connected layer		|
| Dropout      	| 0.8 probability of keeping		|
| Fully connected	    | output 84      									|
| RELU	    | output 84      									|
| Dropout    | 0.8 probability of keeping     									|
| Fully connected		| output 43        									|
| Softmax				| softmax cross entropy with one hot encoding        									|

The model is based on Lenet model, I just added dropouts, hoping that the model will learn excessively to prevent underfitting.

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used a very small learning rate=0.0.1, the batch size of 128 and at least 50 epochs.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.997
* validation set accuracy of 0.962
* test set accuracy of 0.939

If an iterative approach was chosen:
The first architecture I've chosen was a Lenet architecture
But the problem with Lenet that it was designed for less number of classes and was underfitting.
To prevent underfitting I increased the training set and added dropouts with probability of keeping 0.8. This parameter I had to chose by try and error.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because they are not of size (32,32,1), so I had to rescale them first.
I also applied the preprocessing I used for training set.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| Probability
|:---------------------:|:---------------------------------------------:|
| No entry      		| No entry    									| 99.9 |
| Priority road     			| Priority road 										| 100 |
| Turn left ahead					| Right-of-way at the 											| 59.3 |
| Keep right	      		| Keep right					 				|99.9 |
| Stop			| Stop      							|99.6 |


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This is less than the accuracy on the test set of 0.939.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is:
```python
softmax = tf.nn.softmax(predicted_logits)
softmax = sess.run(softmax, feed_dict={logits: predicted_logits})
y_predict = np.argmax(softmax, axis=1)
```
For the first image "No entry", the model is relatively sure that this is a stop sign (probability of 0.999), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 0.999848         			| No entry   									|
| 0.000133     				| Turn left ahead 										|
| 0.000014					| Keep right											|
| 0.000003	      			| Priority road					 				|
| 0.000001			    | Go straight or right      							|


For the image "Turn left ahead" wrongly classified the top 5 look as following:

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 0.593561         			| No entry   									|
| 0.171426     				| Turn left ahead 										|
| 0.160520					| Keep right											|
| 0.055370	      			| Priority road					 				|
| 0.010672  |Go straight or right      							|

So the correct prediction is only on the second place of softmax probabilities. It can be either of the class bias, or of the complexity of the "Turn left ahead" arrow.

![alt text][image16]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
