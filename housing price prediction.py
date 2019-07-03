# -*- coding: utf-8 -*-
"""
Created on Sun May 05 17:41:23 2019

@author: John
"""

import pandas as pd
import matplotlib.pyplot as plt

directory_as_string = "C:/Users/Sriram/Desktop/Semester 2/Capita Selecta/"

data = pd.read_csv(directory_as_string + "housing.csv") 

#print(data.head())
#print(data.info())

#data.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, s=data["population"]/100,
#        label="population", figsize=(15,10), c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)

#data.hist(bins=50, figsize=(20,15))
#plt.show()
#print(data["ocean_proximity"]).value_counts()

#remove all observations on the island
#print(data[data["ocean_proximity"]=="ISLAND"])



data2=data[data["ocean_proximity"]!="ISLAND"]
#print(data2["ocean_proximity"]).value_counts()

data3 = data2[data2["median_house_value"] < 500000]
#print(sum(data3["median_house_value"]>=500000))

data4 = data3[data3["housing_median_age"] < 50]
#print(sum(data4["housing_median_age"] >= 50))
#print(data4.shape)
#test set randomly generated
from sklearn.model_selection import train_test_split
predictors = data4.drop("ocean_proximity", axis=1)
labels = data4["ocean_proximity"].copy()
#print(labels.value_counts())
train_set, test_set = train_test_split(data4, test_size = 0.25, 
                                       random_state = 2019)
train_set_predictors = train_set.drop("ocean_proximity", axis=1)
train_set_labels = train_set["ocean_proximity"].copy()
#print(train_set_predictors.shape)
#print(train_set_labels)
test_set_predictors = test_set.drop("ocean_proximity", axis =1)
test_set_labels = test_set["ocean_proximity"].copy()
print(train_set_labels.value_counts())

import numpy as np
#print(sum(np.isnan(train_set_predictors["total_bedrooms"])))
#150 NaN
from sklearn.preprocessing import Imputer
imputer = Imputer(strategy = "median")

#fit the imputer
imputer.fit(train_set_predictors)
imputer.fit(predictors)
I = imputer.transform(train_set_predictors)
data_training = pd.DataFrame(I, columns = train_set_predictors.columns)
I2 = imputer.transform(predictors)
data = pd.DataFrame(I2, columns = predictors.columns)
#print(sum(np.isnan(data_training["total_bedrooms"])))
#0 NaN

#one hot encode the label
print(train_set_labels)
train_labels_encoded, train_labels_categories = train_set_labels.factorize()

from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
labels_1hot_encoded = encoder.fit_transform(train_labels_encoded.reshape(-1,1))


#data_training.hist(bins=50, figsize=(20,15))
#=============================================
#transformation pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

transformation_pipeline = Pipeline([
        ("imputer", Imputer(strategy = "median")),
        ("min_max_scaler", MinMaxScaler()),
        ])

transformed_training_predictors = pd.DataFrame(transformation_pipeline.fit_transform(train_set_predictors))
transformed_test_predictors = pd.DataFrame(transformation_pipeline.fit_transform(test_set_predictors))
print(transformed_training_predictors)
transformed_predictors = pd.DataFrame(transformation_pipeline.fit_transform(predictors))
#transformed_training_predictors.hist(bins=50, figsize=(20,15))

target_names = ["<1H OCEAN","INLAND","NEAR OCEAN","NEAR BAY"]


# =======================Random forest============================================
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(max_depth = 8, min_samples_split = 10, 
                                min_samples_leaf = 10,
                                random_state = 2019, oob_score = True)
forest.fit(transformed_training_predictors, train_labels_encoded)
predictions_on_test = forest.predict(transformed_test_predictors)
# =============================================================================

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


#======================test set performance===================================
test_labels_encoded, test_labels_categories = test_set_labels.factorize()
cm = confusion_matrix(test_labels_encoded, predictions_on_test)

print(cm)
print(classification_report(test_labels_encoded, predictions_on_test, target_names=target_names))

#======================test for overfitting===================================
predictions_on_training = forest.predict(transformed_training_predictors)
cm_overfit_train = confusion_matrix(train_labels_encoded, predictions_on_training)

print(cm_overfit_train)
print(classification_report(train_labels_encoded, predictions_on_training, target_names=target_names))


#==================cross validation==========================================
forest = RandomForestClassifier(max_depth = 8, min_samples_split = 10, 
                                min_samples_leaf = 10, random_state = 2019)
forest.fit(transformed_training_predictors, train_labels_encoded)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(forest, transformed_training_predictors, train_labels_encoded,
                         scoring = "f1_micro", cv=10)
print(scores.mean(), scores.std())


#=================grid search for best hyperparameters========================
# =============================================================================
# from sklearn.model_selection import GridSearchCV
# 
# param_grid = [
#         {"max_depth":[3, 5, 8], "min_samples_split" : [3, 5, 8], "min_samples_leaf": [2, 5]}]
# 
# forest_reg = RandomForestClassifier()
# grid_search = GridSearchCV(forest_reg, param_grid, cv=10, scoring = "f1_micro")
# grid_search.fit(transformed_training_predictors, train_labels_encoded)
# print(grid_search.best_params_)
# 
# =============================================================================







#======================= Neural Network ======================================

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder


#reproduce the results
seed = 2019
np.random.seed(2019)

#one hot encode the response variable
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(train_set_labels)
train_labels_1hot_encoded = encoder.transform(train_set_labels)
test_labels_1hot_encoded = encoder.transform(test_set_labels)
labels_1hot_encoded = encoder.transform(labels)

# convert integers to dummy variables (i.e. one hot encoded)
dummy_train_labels = np_utils.to_categorical(train_labels_1hot_encoded)
dummy_test_labels = np_utils.to_categorical(test_labels_1hot_encoded)
dummy_labels =  np_utils.to_categorical(labels_1hot_encoded)
#===================== Shallow network (1 layer) ======================================
n_cols = transformed_training_predictors.shape[1]
input_shape = (n_cols,)


from keras.callbacks import EarlyStopping

early_stopping_monitor = EarlyStopping(patience=2)


from keras.layers.normalization import BatchNormalization
# split into input (X) and output (Y) variables
X = transformed_predictors
Y = dummy_labels


# create model
neural_network = Sequential()
neural_network.add(Dense(300, kernel_initializer='uniform', activation='relu', 
                input_shape = input_shape))
#model.add(Dropout(0.5))
neural_network.add(BatchNormalization())
neural_network.add(Dense(200, kernel_initializer='uniform', activation='relu'))
#model.add(Dropout(0.5))
neural_network.add(BatchNormalization())
neural_network.add(Dense(4, kernel_initializer='uniform', activation='softmax'))
# Compile model
neural_network.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['accuracy'])
# Fit the model
history = neural_network.fit(X, Y, validation_split=0.33, epochs=150,
                    callbacks=[early_stopping_monitor], verbose=1)
# list all data in history
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# summarize history for loss
# =============================================================================
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# 
# =============================================================================
#predictions

predictions_on_training_set = neural_network.predict(transformed_training_predictors)
predictions_on_test_set = neural_network.predict(transformed_test_predictors)
#=========================== performance metrics ============================

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

cm_overfit_train = confusion_matrix(dummy_train_labels.argmax(axis=1),
                                    predictions_on_training_set.argmax(axis=1))
cm_overfit_test = confusion_matrix(dummy_test_labels.argmax(axis=1), 
                                   predictions_on_test_set.argmax(axis = 1))
print(cm_overfit_train)
print(cm_overfit_test)

#precision and recall of neural network
training_report = classification_report(dummy_train_labels.argmax(axis=1), 
                            predictions_on_training_set.argmax(axis=1), 
                            target_names=target_names)

test_report = classification_report(dummy_test_labels.argmax(axis=1), 
                            predictions_on_test_set.argmax(axis=1), 
                            target_names=target_names)
print(test_report)
print(training_report)









#==============================Support vector machine========================

import pandas as pd

#using radial basis function kernel
transformed_training_predictors = pd.DataFrame(transformation_pipeline.fit_transform(train_set_predictors))
transformed_test_predictors = pd.DataFrame(transformation_pipeline.fit_transform(test_set_predictors))
train_set_labels = train_set["ocean_proximity"].copy()
test_set_labels = test_set["ocean_proximity"].copy()


from sklearn.svm import SVC
#==============================polynomial kernel=============================
poly_kernel_svc = SVC(kernel = "poly",  class_weight='balanced',degree = 20, 
                      coef0=1, C=5)
poly_kernel_svc.fit(transformed_training_predictors, train_set_labels)

#==============================rbf kernel====================================
rbf_kernel_svc = SVC(kernel = "rbf",class_weight='balanced', gamma = 5, C = 0.001, verbose=True)
rbf_kernel_svc.fit(transformed_training_predictors, train_set_labels)

#==============================predictions===================================
predictions_rbf = rbf_kernel_svc.predict(transformed_test_predictors)
predictions_poly = poly_kernel_svc.predict(transformed_test_predictors)

predictions_train_rbf = rbf_kernel_svc.predict(transformed_training_predictors)
predictions_train_poly = poly_kernel_svc.predict(transformed_training_predictors)
#=============================performance====================================
from sklearn import metrics
print("RBF Accuracy:",metrics.accuracy_score(test_set_labels, predictions_rbf))
print("Poly Accuracy:",metrics.accuracy_score(test_set_labels, predictions_poly))

#============================polynomial kernel===============================
test_poly_report = classification_report(test_set_labels, 
                            predictions_poly, 
                            target_names=target_names)

train_poly_report = classification_report(train_set_labels, 
                            predictions_train_poly, 
                            target_names=target_names)
print(test_poly_report)
print(train_poly_report)


#================================rbf=========================================
test_rbf_report = classification_report(test_set_labels, 
                            predictions_rbf, 
                            target_names=target_names)

train_rbf_report = classification_report(train_set_labels, 
                           predictions_train_rbf, 
                           target_names=target_names)
print(test_rbf_report)
print(train_rbf_report)


rbf_kernel_svc = SVC(kernel = "rbf",class_weight='balanced', gamma = 12, C = 5, 
                     verbose=True)
rbf_kernel_svc.fit(transformed_training_predictors, train_set_labels)
predictions_rbf = rbf_kernel_svc.predict(transformed_test_predictors)
print("RBF Accuracy:",metrics.accuracy_score(test_set_labels, predictions_rbf))
test_rbf_report = classification_report(test_set_labels, 
                            predictions_rbf, 
                            target_names=target_names)

#train_rbf_report = classification_report(train_set_labels, 
#                            predictions_train_rbf, 
#                            target_names=target_names)
print(test_rbf_report)





#==============================naive bayes ==================================
from sklearn.naive_bayes import GaussianNB


transformed_training_predictors = pd.DataFrame(transformation_pipeline.fit_transform(train_set_predictors))
transformed_test_predictors = pd.DataFrame(transformation_pipeline.fit_transform(test_set_predictors))
train_set_labels = train_set["ocean_proximity"].copy()
test_set_labels = test_set["ocean_proximity"].copy()

obs_proportions = []
total = sum(train_set_labels.value_counts())

for freq in train_set_labels.value_counts():
    obs_proportions.append(freq/total)


naive_bayes_classifier = GaussianNB(priors = obs_proportions)
naive_bayes_classifier.fit(transformed_training_predictors, train_set_labels)

nb_predictions = naive_bayes_classifier.predict(transformed_test_predictors)

#============================performance metrics=============================
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(test_set_labels, nb_predictions))

# ==========================overfitting test==================================
# on the training set
predictions_on_training = naive_bayes_classifier.predict(transformed_training_predictors)
cm_overfit_train = confusion_matrix(train_set_labels, predictions_on_training)

print(cm_overfit_train)
print(classification_report(train_set_labels, predictions_on_training, 
                            target_names=target_names))
# now on test set
predictions_on_test = naive_bayes_classifier.predict(transformed_test_predictors)
cm_overfit_test = confusion_matrix(test_set_labels, predictions_on_test)

print(cm_overfit_test)
print(classification_report(test_set_labels, predictions_on_test, 
                            target_names=target_names))
# =============================================================================







#===========================the ensemble ====================================
from sklearn.ensemble import VotingClassifier

#names of the classifiers:
#forest
#neural_network
#rbf_kernel_svc
#naive_bayes_classifier 

#have to wrap the neural network since it's done in Keras
def create_model(optimizer='adam',
                 kernel_initializer = 'uniform'):
    neural_network = Sequential()
    neural_network.add(Dense(300, kernel_initializer='uniform', activation='relu', 
                input_shape = input_shape))
    #model.add(Dropout(0.5))
    neural_network.add(BatchNormalization())
    neural_network.add(Dense(200, kernel_initializer='uniform', activation='relu'))
    neural_network.add(Dropout(0.5))
    neural_network.add(BatchNormalization())
    neural_network.add(Dense(4, kernel_initializer='uniform', activation='softmax'))
    # Compile model
    neural_network.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['accuracy'])
    return neural_network

#wrap the network
from keras.wrappers.scikit_learn import KerasClassifier
network = KerasClassifier(build_fn=create_model, verbose=0)


#state weights for each classifier
from sklearn.metrics import accuracy_score
weights = [0.25, 0.25, 0.25, 0.25]
weights2 = [0.016, 0.984, 0.984, 0.984]

#create voting classifier
voting_clf = VotingClassifier(
        estimators = [('random forest', forest), ("neural network", network),
                      ("support vector", rbf_kernel_svc), ("naive bayes", naive_bayes_classifier)],
                      weights = weights, voting = "hard")
voting_clf.fit(transformed_training_predictors, train_set_labels)

#================================ performance metrics =======================
predictions_on_training = voting_clf.predict(transformed_training_predictors)
cm_overfit_train = confusion_matrix(train_set_labels, predictions_on_training)

print(cm_overfit_train)
print(classification_report(train_set_labels, predictions_on_training, 
                            target_names=target_names))
# now on test set
predictions_on_test = voting_clf.predict(transformed_test_predictors)
cm_overfit_test = confusion_matrix(test_set_labels, predictions_on_test)

print(cm_overfit_test)
print(classification_report(test_set_labels, predictions_on_test, 
                            target_names=target_names))

#=============================accuracy of all classifiers ====================
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

for classifier, label in zip([forest, network, rbf_kernel_svc, naive_bayes_classifier, voting_clf], ["Random Forest", "Neural Network", "Support Vector Machine", "Naive Bayes", "Voting Classifier"]):
    classifier.fit(transformed_training_predictors, train_set_labels)
    predictions = classifier.predict(transformed_test_predictors)
    scores = cross_val_score(classifier, transformed_training_predictors,
                                              train_set_labels, cv=5, scoring="accuracy")
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))



#======================== find optimal weights for ensemble ==================

