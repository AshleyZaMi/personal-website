# Ashley Zapata Minero
# Project 6 -Dendroguins
# Description: In this Project I wranggled the data, applied a cross validator using the Decision Tree Model in order to get the most efficeint hyperparameters. We gave it a set of parameters
# in dictionary form, and then I proceeded to do training with X_train, y_train and as a result I got a dictionary with the most efficeint hyperparameters. I then proceeded to use those hyperparameters
# and put it into a new DecisionTree learning model using those hyperparameters. I plotted it in using a confusion matrix that shows the classes labels and also plotted a decision tree to the left of it.
# I also made a classification report, and answered questions. 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA

# def conducting_PCA(df_penguins):
#     #decomposing the feature vector
#     comp=2
#     pca = PCA(n_components=comp)

#     # Select only numeric columns for PCA (features)
#     X = df_penguins.select_dtypes(include=[float, int])  # Select numeric features only
#     X_pca = pd.DataFrame(pca.fit(X).transform(X), index=X.index)

#     return X_pca

def wrangling():
    #gathering data
    file_path='penguins.csv'
    df_penguins = pd.read_csv(file_path)

    #filtering
    df_penguins = df_penguins.drop_duplicates().dropna()

    # #PCA
    # X_pca= conducting_PCA(df_penguins)

    #encoding
    # df_penguins = pd.get_dummies(df_penguins, columns=['island','sex'])

    #X(dropping non-numeric features) and y
    X = df_penguins.drop(columns=['species', 'sex','island'])
    y = df_penguins['species']

    #training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=567)

    return df_penguins, X, y, X_train, X_test, y_train, y_test

def cross_validation(X_train, y_train):
 #cross validating a Decision Tree Classifier
    # parameters dictionary holding key and value of parameters
    grid_of_params = {
        'criterion': ['entropy', 'gini'],
        'max_depth': [2, 4, 6],
        'min_samples_leaf': [1, 2, 3]
    }

    #setting the model
    model_DecisionTree = tree.DecisionTreeClassifier()
    #Doing cross validation using GridSearchCV based on hyperparameters stored in grid_of_params dictionary
    gacv = GridSearchCV(estimator=model_DecisionTree, 
    param_grid=grid_of_params, scoring='accuracy')

    #training GridSearch with hyperparameters
    gacv.fit(X_train, y_train)


    print('best params:\n', gacv.best_params_)
    print('scores:\n',gacv.cv_results_['mean_test_score'])
    
    return model_DecisionTree, gacv, gacv.best_params_, gacv.cv_results_

def reinitalized_Best_Decision_Tree(best_params_, model_DecisionTree, X_train, y_train):
    #training decsion tree based on cross validation parameters
    model_DecisionTree = tree.DecisionTreeClassifier(**best_params_) # ** unwraps the dictionary
    #training
    trained_DecisionTree = model_DecisionTree.fit(X_train,y_train)

    return trained_DecisionTree

def setting_confusion_matrix(X_test, y_test, trained_DecisionTree):
    #predicting on the test set
    y_pred = trained_DecisionTree.predict(X_test)

    #computing a confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    #Creating a confusion matrix display object
    cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, 
    display_labels=trained_DecisionTree.classes_)
    
    return cm_display, y_pred


def visualization(X, cm_display, trained_DecisionTree):
    featureNames = X.columns.tolist()

    fig,ax = plt.subplots(1,2, figsize=(17,5))

    #plot the confusion matrix display object
    cm_display.plot(ax=ax[0])
    ax[0].set(title='Optimized Confusion Matrix')
    #add title

    #plotting the decision tree
    tree.plot_tree(trained_DecisionTree, ax=ax[1], filled=True, 
    feature_names = featureNames, class_names=trained_DecisionTree.classes_)
    ax[1].set(title='Optimized Decision Tree')

    #saving image
    fig.suptitle('Penguins Decision Tree, optimized via Cross Validation')
    fig.tight_layout()
    fig.savefig('Penguins Decision Tree.png')

def generating_classification_report(y_pred, y_test):
    classification_rep = classification_report(y_test, y_pred)
    print("classification report:\n", classification_rep)

def questions(best_params_):
    print("Q1| Opimized hyperparams: ", best_params_)
    print("Q2| Predicted instance of a label I am most confident about being predicted correctly is \'Gentoo\'")
    print("Q3| True instance of a label I am most confident about being predicted correctly is \'Adelie\'")
    print("Q4| Discrepancy of how each of the labels are being predicted might be due to factors such as the structure of a decision tree, which may affect where the tree is partitioned, since it tends to favor the classes that have highly distinct features distributions when making the split. Perhaps there may also be some overlaps in some species where they have the same attributes which may cause discrepancy in how lables are being predicted. Also perhaps some classes in the penguins.csv may have more samples than others, which can also cause discrepancy.")


def main():

    #Wrangling: gathering data, cleaning data, training and testing sets
    df_penguins, X, y, X_train, X_test, y_train, y_test = wrangling()

    #cross validating: using a Decision Tree Classifier to use the parameters indicated and to check all possibilites in order to recieve the best results to use for our model's optimal hyperparameters
    model_DecisionTree, gacv, best_params_, cv_results_ = cross_validation(X_train, y_train)
   
    #reinitalized_Best_Decision_Tree: training decision tree based on cross validation parameters 
    #end product -> trained decision tree model
    trained_DecisionTree = reinitalized_Best_Decision_Tree(best_params_, model_DecisionTree, X_train, y_train)

    #confusion_matrix: predicting on a test set from trained_DecisionTree model, Computing a confusion matrix
    #creating a confusion matrix display object
    cm_display, y_pred = setting_confusion_matrix(X_test, y_test, trained_DecisionTree)

    #visualizing: visualizing the confusion matrix on the left and the trained_DecisionTree on the right in one plot
    visualization(X, cm_display, trained_DecisionTree)

    #Producing a classification report
    generating_classification_report(y_pred, y_test)

    questions(best_params_)

    



if __name__ == '__main__':
    main()
