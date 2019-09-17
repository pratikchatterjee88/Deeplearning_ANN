# Artificial Neural Network

# Install Theano

# Installing Tensorflow
    # Numerical library

# Above are used to do research work

# Installing Keras
# it is based on theno and tensorflow
    
# Part 1- Data Preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
    
# importing data set
dataset=pd.read_csv("Churn_Modelling.csv")
X=dataset.iloc[:,3:13].values
y=dataset.iloc[:,13].values

# Encodong categorical data
# encoding categorical variables of independent variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1=LabelEncoder()
X[:,1]=labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2=LabelEncoder()
X[:,2]=labelencoder_X_2.fit_transform(X[:,2])
# Doing one hot encording for the location column
onehotencoder = OneHotEncoder(categorical_features=[1])
X=onehotencoder.fit_transform(X).toarray()

# Removing one column from the dataset of the location column 
#to remove chances of falling into dummy variable trap
X=X[:,1:]

# Splitting the dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=   train_test_split(X, y, test_size=0.2, random_state=0) 

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

# PART 2
# Building Artificial Neural Network
import keras
# import sequential module required to initialise or NN
# dense module to build the layers of our ANN

from keras.models import Sequential
from keras.layers import Dense

# to implement dropouts as discussed in line 209
from keras.layers import Dropout

# Initializing the ANN as defining the sequense of layers

classifier = Sequential()

# Adding the input layer and the first hidden layer--------with dropouts
'''output dim=no. of nodes we need to add to the hidden layer
# why 6? There is not rule of thumb for the no. of nodes but as a better way to take we may choose to
#rules
#1. if the data is linearly seperable we do not need a hidden layer or neural network
#2. Chhose the no. of nodes in the hiden layer as the average of the number of nodes of the hidden layer
# and the no. of nodes in the output layer, else we can do parameter tuning if we want to be artist
#hence, as out total variables are 11 so, the number of nodes input layer is 11 which is nothing but the no. of independent variables
# hence, 11+1/2=6 nodes 

# what is init: we need to randomly initialise weight using uniform
# input_dim: no of dependent variables'''

classifier.add(Dense(output_dim=6,init='uniform', activation='relu', input_dim=11))
classifier.add(Dropout())
# Adding the second hidden layer
classifier.add(Dense(output_dim=6,init='uniform', activation='relu'))

# Adding the output layer

# why outpu dim is 1, as our output has a coategotical varible so 1.
classifier.add(Dense(output_dim=1,init='uniform', activation='sigmoid'))

# Applying scholastic gradient descent to ANN

#Compile the ANN
''' compiling means that we need to add sholastic gradient descent to the ANN

#optimizer: the algo that we want to apply to find the weights. There are several scholastic gradient descent
# algo but 'adam' is very useful
#loss finction: if out output is a binary output then the loss function will be _binary_crossentropy and if
# the output has more that 2 binary output then the loss function should be categorical_cross_entropy'''

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the traning set

#batch size= it is after which we want out weights to be updated
#epoc: the no. of round the ANN runs

classifier.fit(X_train, y_train,  batch_size =10, nb_epoch = 100 )

#PART 3: Making the preductions and evaluating the model
y_pred=classifier.predict(X_test)
# the above will tell the probability of the customer leaving the bank.

# Although we want our output to be in TRUE/FALSE hence the below can be done to get. A nor threshold
# of 50% is good but if we want higher precision in canse of cancer the we can make it 80%.
y_pred=(y_pred>0.5)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)

# accuracy=(1531+160)/2000
#0.8455

#2nd Instance: accuracy(1540+140)/2000
# 0.84

#--------------------------
# HOMEWORK

'''Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure:: 3 years
Balance: $60000
No. of products: 2
Does this cust. have a credit card: yes
Is this customer an Active member: Yes
Estimated Salary: $50000'''

# we will make a single new prediction

''' The trick here is, since the data is in horizontal vector we need to add [[]]. 
#our dataset is distributed horizontallly so new data should also be same way added'''
# single bracket[] will add data in column where as [[]] make 2 dimentional array 

#new_prediction= classifier.predict(np.array([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]))
# we need to make add standard scaller so convert the above

new_prediction= classifier.predict(sc.transform(np.array([[0., 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction=(new_prediction > 0.5)

#--------------------------------------------------------------------#########

# PART 4: Evaluating, Improving and Tuning the ANN

'''So, as we run different times our model the accuracy changes hence in order to get a fixed accuracy
we need to fix it. This can be done using

------K_ fold Cross validation------ it works by train the train set into 10 folds, and we train our model on 9 folds
and we test it on the last remaining.

It also introduces the Bias- Variance Trade off. As in we need to train our model so that we get 
"low bias and low varience"

So in short, judging the accuracy with one combination is not the best method so we should always do
cross validation with 10 iteration and then use the mean of the 10 iterations as the final accuracy.
'''


# Evaluating the ANN
''' we need to restart our kernal as few of the steps above are not required coz they are used inside the
k fold cross validation although we need the data preprocessing'''
# run part 1 of data preprocessing first----

# we need to combine and run keras and scikit leran together using a keras wrapper called keras classifier
import keras
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
# we need to create a function here to build the architecture of the ANN as done above to build the ANN

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

# now create a classifier which is a global variable because the classifier is a local variable created inside the function
# and also it will not be build on the Train but on k fold
    
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, nb_epoch = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv= 10)
# n_jobs=-1 is not working
#the accuracy will hold the 10 accuracies
#n-jobs is very important in deep learning as -1 means use all the cpu's to run parallel ccomputing to make our work faster

''' So why do we use cross validation
    1. which one accuracy should we take and 
    2. we will be able to understand where our model is falling in the bias- variance trade off'''

mean=accuracies.mean()
varience=accuracies.std()
'''Now since the accuracy became 79% reduced from 84% we will try to increase the accuracy by using
parameter tuning;'''


# -----------------PARAMETER TUNING-----------------
# Tuning the ANN
''' One very important thing in deep learning is
Dropout Regularization
which removes the overfitting of the model

How dropout works

At each iteration of the training, some neurons of the ANN are randomly disabled to prevent them from
being too dependent on each other when they learn to co relation and therefore by overwriting these neurons
the ANN learns several independent corellations in the data because each time there is not same configuration
of the neurons.
And the fact that we get these independent corelation of the datas thanks to the fact that the neurons work
more independently that prevents the neurons from learning too much therefore that prevents OVERFITTING

IMPLEMENTING THE DROPOUTS...these will be done before we create the layers. Although our data has no overfitting but in case...so 
go back to the layer area..





