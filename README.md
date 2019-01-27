
# Detecting Malaria infected cells with Machine Learning and TensorFlow.
HealthCare and Medical Science are the constantly developing fields which require human expertise
and high-end technology. With Machine Learning and [TensorFlow](https://www.tensorflow.org/), we could fight one of the deadly disesases
like Malaria. 

# About the Project
The following files/directories are included in this GitHub project :
1. `data28` and `data32` are the directories which include 4 files each namely :
    1. `x.npy` : Holds the training data with images
    2. `y.npy` : Holds the training data with labels
    3. `test_x.npy` : Holds the testing/validation data with images
    4. `test_y.npy` : Holds the testing/validation data with labels
    
    The numbers `28` and `32` at the end of the directory's name indicate the dimension of the
    images present in it. For example, `data32` would have images having dimensions `32 * 32`.
    The data is collected from [ Arunava's Malaria Cell Images Dataset on Kaggle ](https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria)

2. `models` : The directory where the [Keras](https://www.tensorflow.org/guide/keras) models will be saved. A sample model named
    0001.h5 is placed in it. It can act as a preliminary model. See [MainFile.py](https://github.com/shubham0204/Malaria_Cell_Classification/blob/master/MainFile.py)
    
3. `MainFile.py` : The Python script which loads the data and feed it to the model for training and evaluation.

4. `Model.py` : The Python script which defines a class for building the Keras model.

5. `DataConverter.py` **( only study purpose )** The Python script which can load images from a directory. 
It is only for study purposes and we recommend to use the data provided in the `data32` and `data28` directories.



