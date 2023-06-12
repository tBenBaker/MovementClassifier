# MovementClassifier

This project develops a classifer for dance data, which should generalize to other forms of movement data

The main elements are

(a) A python class (DanceObj) which takes in the X, Y, and Z position of a body's joints at each frame of a sequence, 
    and the framerate in 1/seconds.
    It has functions within it for deriving features of movement sequences.
    get_features(self) runs the input data through all the other functions
    and outputs a dictionary of feature-names and their values.

(b) Code for loading and organizing data (see data_proc).
    It was built to be run first on the AIST++ dataset, containing 1408 clips of 10 hip-hip dance genres
    At this point the dataloading is specific to this dataset, and would need to be adapted for others
    Data is fed into the python class above, so the output can be used to train a classifier

(c) Training a machine learning genre-classifers based on the features, and evaluating the resulting model. Here We have focused on XGBoost, which had the best performance of those tested. Evaluation is still in progress.

Notebook 01 gives as simplified look at the features. Notebook 02 goes through classification. Notebook 03 inspects the classifier.

