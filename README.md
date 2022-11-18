# Deep Learning: Predicting Charity Funding
## Contributor: Sam Espe

### Overview
This project was created as a submission for Homework #21 for Data Visualization and Analysis Boot Camp.

The goal for this project was to build a TensorFlow deep neural network that could predict whether fictional charities would get funded or not. The data set of fictional charities was provided as a CSV file, located in the `Resources` folder. I created and developed the models using Google Colab, so I uploaded the data set to an AWS S3 bucket to access the data in a Google Colab notebook.

This project uses TensorFlow version 2.9.2, Pandas version 1.3.5, and SciKit-Learn version 1.0.2 packages. Make sure that your environment is compatible with these packages. If your environment is incompatible, the notebooks and output HDF5 files are available at this Google Drive address: https://drive.google.com/drive/folders/11RlxbjVi_kyS8NPmCnPHfQIhZClXpJ9h?usp=sharing.

### Results

#### Data Pre-Processing
![image](https://user-images.githubusercontent.com/106678018/202801573-43ab7313-0db8-40d2-b046-2f0177070241.png)

The target of the model is the "IS_SUCCESSFUL" column of the dataframe. Because I am doing supervised learning, I had to remove this column from the data set and save it separately so the model doesn't accidentally get contaminated.

The features of the model are:
* "APPLICATION_TYPE"
* "AFFILIATION"
* "CLASSIFICATION"
* "USE_CASE"
* "ORGANIZATION"
* "STATUS"
* "INCOME_AMT"
* "SPECIAL_CONSIDERATIONS"
* "ASK_AMT"

The variables that were removed because they were neither targets nor features are:
* "EIN"
* "NAME"
I removed these variables because they are unique to each application, so they will not have any value in trying to predict whether a given application will be successful or not.

After separating the feature matrix from the target array and removing the irrelevant columns, I examined the number of unique values for each feature. For the categorical variables that had more than 10 unique values, I binned together the low-frequency values into a category called "Other". 

I used the Pandas method "get_dummies" to one-hot encode the categorical data. This process increased the number of columns from 9 to 46. I eliminated the redundant column caused by one-hot encoding the "SPECIAL_CONSIDERATIONS" column into two: one where Y was encoded as 1, and one where N was encoded as 1. I removed the column where N was encoded as 1 for clarity.

I scaled the data using the Standard Scaler provided by SciKit-Learn, and then split the scaled data into a training set and a testing set at the default proportion (70/30).

#### The Original Neural Network Model

The original neural network model I created is located in the `Original_Neural_Network_Model` folder of this repo. For this neural network, I used TensorFlow to create a neural network. The input layer had 46 input dimensions, 150 nodes, and used a ReLU activation function. The hidden layer had 150 nodes using a ReLU activation function. The output layer had 1 node with a sigmoid activation function.

This model was trained on the training set for 100 epochs. This model did not achieve the goal of 75% accuracy. This model produced an accuracy of 72.79% on the testing set, with a loss of 58.79%.

#### First Optimization Attempt

This model is located in the `Optimization_Attempts` folder of this repo. For my first attempt at improving my model, I tried increasing the size of the "Other" categories when I binned the columns with a large number of unique values. I reduced the number of input parameters to 44. 

This model was trained in the same way as the original. It did not achieve the goal of 75% accuracy. This model produced an accuracy of 72.73% on the testing set, with a loss of 57.93%. This model performed about the same as the original. 

#### Second Optimization Attempt

This model is located in the `Optimization_Attempts` folder of this repo. For my second attempt at improving the model, I tried adding another hidden layer. This second hidden layer had 100 nodes with a ReLU activation function. 

This model was trained in the same way as the original. It did not achieve the goal of 75% accuracy. This model produced an accuracy of 72.76%, but with a loss of 61.38%. It is possible that this model was a little overtrained. The accuracy on the training set was slightly better than on the testing set, but the loss on the testing set was much larger than the loss on the training set.

#### Third Optimization Attempt

This model is located in the `Optimization_Attempts` folder of this repo. For this attempt, I tried increasing the number of nodes in the first layer and decreasing the number of nodes in the hidden layer. I also implemented the binning changes from attempt 1. 

This model was trained in the same way as the original. It did not achieve the goal of 75% accuracy. This model produced an accuracy of 72.68% with a loss of 58.70%. The outcomes from this model were about the same as the outcomes from the original model.

### Summary

Overall the neural network models did a fair job predicting whether charities would receive funding. I was not able to reach the goal of 75% accuracy but I was close to it. The modifications I did to my models did not seem to improve the performance.

I wonder if the neural network is the best model to use in this case. I would be interested to see if another method of supervised learning, such as a Random Forest Classifier, could produce similar or better results. 
