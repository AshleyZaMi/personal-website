# Ashley Zapata Minero
# Project07: Diabetes Classification Revisited
# Description: in this project I made a decisionTree model for both Giniindex and for Entropy
#in order to compare which one is better. We were also asked to use this classification data in order to perform the DecisonTree model
# I first wrangled the data by gathering the data, dropping duplicates and null values. I then went to splitting up the X features vector(all except Outcome) and 
#the y-target(Outcome). Then I went on to partitioning X feature vector and y-target into training and testing sets. After that I proceeded to plot them using their
# respecive types by changing the criterion to either "gini" or "entropy" and made an empty model before proceeding to train each individual one using .fit() with the
# respective paritioned training data: X_train, y_train. Then I proceed to make the plotting of them within a one row and 2 columns figure. I also added the prediction
# and testing function in order to determine which criterion is the best and proceeded to make a y_prediction by using the trained Decision tree respecitive to either gini/entropy
# and applyiing .predict() to the X_test to feed it new data. Then I used the y_prection and used it on accuracy_score function from sklearn and fed it the y_test and y_prediciton 
# respective to gini/entropy and was able to print out the accuracy scores of these two classifications and determine the answer for questions 1 and 2. For the plotting I added labels 
# corresponding to the features by making the X's dataframe columns into list to make it accesible, and I also applied "Yes,No" to represent 0 and 1 from Outcome(y_target) for the class
#name labeling.
import pandas as pd
import sklearn as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def wrangling():
    #collect the data
    file_path = 'diabetes.csv'
    df_diabetes= pd.read_csv(file_path)
    #clean the data- by droping duplicates
    df_diabetes = df_diabetes.drop_duplicates()
    #removing any nulls
    df_diabetes = df_diabetes.dropna()
    #seperating X feature vector and y target- we'rre droping outcome because we only want those features that are not outcome, and because outcome is the y-target
    X = df_diabetes.drop(columns=['Outcome']) #X is a dataframe
    # print("X:", type(X))
    y = df_diabetes['Outcome'] #y is a series
    # print("y:", type(y)) 

    #returning X feature vector and y target
    return df_diabetes, X, y

def partitioning(X,y):
    #paritioning the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=777, stratify=y)

    return X_train, X_test, y_train, y_test
    
#training two classifiers: one for entropy and another for gini index
def trainingEntropy_classifier(X_train,y_train):
    #creating an empty model
    model_DecisionTreeEntropy = tree.DecisionTreeClassifier(criterion='entropy', splitter = 'best', random_state= 777, max_depth=3)
    #training the model
    trained_EntropyDTree = model_DecisionTreeEntropy.fit(X_train,y_train)
    
    return trained_EntropyDTree


def gini_classifier(X_train,y_train):
    #creating an empty model
    model_DecisionTreeGini = tree.DecisionTreeClassifier(criterion ='gini', splitter='best', random_state = 777, max_depth=3)
    #training the model
    trained_GiniDTree = model_DecisionTreeGini.fit(X_train,y_train)
    return trained_GiniDTree

    
#using X_test and y_test paritions to test accuracy
def prediction_and_testingAccuracy(trained_GiniDTree, trained_EntropyDTree, X_test, y_test):
    Gini_y_prediction = trained_GiniDTree.predict(X_test)

    Entropy_y_prediction = trained_EntropyDTree.predict(X_test)
    #accuracy_score helps us to compare both these models in order to see which one performed better
    accuracyGini = accuracy_score(y_test, Gini_y_prediction, normalize=True)
    accuracyEntropy = accuracy_score(y_test, Entropy_y_prediction, normalize=True)

    return accuracyGini, accuracyEntropy

    

def main():
    # write your code here
    #returning the wrangled data through the variable 
    df_diabetes, X, y = wrangling()

    #spliting the data into training and testing sets
    X_train, X_test, y_train, y_test = partitioning(X, y)

    #Making an empty Decision tree model and training it using the training parition criteron entropy, and storing the trained model in a variable
    trained_EntropyDTree = trainingEntropy_classifier(X_train, y_train)
    # print(trained_EntropyDTree)

    #Making an empty Decision tree model and training it using training partition using gini index, and storing the trained model in a variable 
    trained_GiniDTree = gini_classifier(X_train, y_train)

    #making a y_prediction based on X_test and comparing that y_prediction to y_test -> using accuracy_score helps us compare both the Gini and entropy model accuracy on new data
    accuracyGini, accuracyEntropy = prediction_and_testingAccuracy(trained_GiniDTree, trained_EntropyDTree, X_test, y_test)
    
    print("accuracyGini: ", accuracyGini)
    print("accuracyEntropy:", accuracyEntropy)



    #getting feature names and class class names
    featureNames = X.columns.tolist() #because it is a dataframe then we have to convert the column names to a list
    classNames = ['Yes','No']


    #making a ploting area
    fig, ax = plt.subplots(1,2,figsize=(30,15))
    tree.plot_tree(trained_EntropyDTree, ax=ax[0], filled=True, feature_names= featureNames, class_names= classNames)
    tree.plot_tree(trained_GiniDTree, ax=ax[1], filled=True, feature_names= featureNames, class_names= classNames)
    fig.suptitle('Diabetes Decision Tree Classification Results')
    fig.tight_layout()
    fig.savefig('decisiontrees.png')

    #question portion:
    print("\nQuestion 1: No the resulting models are not the same, as there is a 0.0043 difference in their accuracy.Entropy is shown to have a little bit of a better performance than Gini index, because Entropy had an accuracy score of 0.7273 while Gini had 0.7229.")

    print("\nQuestion 2: Based on the accuracy results the Entropy model is the better model, however since the difference is negligible both models roughly perform similarily in this case.")



if __name__ == '__main__':
    main()
