# Ashley Zapata Minero
# Project06: Diabetes Classification
# Description: This project involves doing wrangling steps 1-4(gather data, unusable data, quantative(correlation matrix),
# spliting into feature vector and target) and 7(partitioning the data into training and testing sets) and for logisticRegression 
# do empty model set up model training, and prediction and testing acurracy for the model. In which the end result is the accuracy score.
# Also added a set seed to get the same testing and training data, in order to not get randomized results each time.
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#This class handles all the steps of the wrangling process
class wrangleData:
    #constructor-> initializing attributes file_path and dataframe
    def __init__(self,file_path):
        self.file_path = file_path #Attribute -> stores the file path
        self.dataframe = self.gather_data() #Attribute dataframe -> stores the gathered dataframe| .gather_data() calls the method gather_data() and initalizes itself with self

    #gather_data -> is a helper method used to populate self.dataframe
    def gather_data(self):
        #wrangling 1: gather data
        pd.set_option('display.width', None)

        #modifying an self's attribute
        self.dataframe = pd.read_csv(self.file_path, header=1) #skipping the first row

        # print('diabetes data:\n', self.dataframe)
        # print('\ndiabetes data info:')
        # self.dataframe.info()
        
        return self.dataframe
    
    def wrangle_unusable_data(self):
        #wrangling #2: unusable data

        #remove duplicates- better practice for ML if wanting to train to find new insights also cause diabetes data is crossectional | use self.attribute_name when modifying or accessing data
        self.dataframe = self.dataframe.drop_duplicates()

        #dropping null values
        self.dataframe = self.dataframe.dropna()

        # print("in wrangle #2 method")


    def dfColumnCorrelationMatrix(self):
        #wrangling #3: quantative(correlation Matrix)
        #correlation matrix: in a dataframe shows how related each of the numeric attributes
        #is to each other numeric attribute

        #no need for a for loop just apply .corr() to get the correlation matrix, take the column you want and then sort them,then extract the top one that's not y
        
        #create a correlation matrix
        correlation_matrix = self.dataframe.corr()
        # print('correlation_matrix:\n', correlation_matrix)

        #dropping the row of 'Outcome', associated the column 'Outcome' that makes a new dataframe and prevents that being one of the highest correlations found
        correlation_matrix= correlation_matrix['Outcome'].drop(index='Outcome')
        # print('\ncorrelation_matrix \'Outcome\' index/row droped from \'Outcome\' column:\n', correlation_matrix)
    
        #sorting out the values through descending order to see which have more correlation
        correlation_matrix_sorted = correlation_matrix.sort_values(ascending=False)
        # print('\ncorrelation_matrix_sort:\n', correlation_matrix_sorted)
        
        # print(type(correlation_matrix_sorted.values))

        #creating a list to hold the attributes
        four_highest_correlation_attribute = []

        #Getting the top 4 highest correlation by accessing integer locations from the series holding the top 4,
        #and doign so it made a new dataframe with only those values
        df_of_top_4_correlations = correlation_matrix_sorted.iloc[0:4]
        # print('\ndf top 4 corr:', top_4_correlations)

        #appending the indexes of the data frame of top_4_correlations to the list for easier access| using self. to be able to access it in other methods
        self.four_highest_correlation_attribute = df_of_top_4_correlations.index

        #returning the list as self. to be able to access it in other methods
        return self.four_highest_correlation_attribute

    def seperate_Xfeature_yTarget(self):
        #Wrangling 4: seperate In and Out: spliting into feature vector and target

        #splitting into feature vector
        self.y = self.dataframe['Outcome']

        # print('\ny-target:\n', self.y)

        #splitting into y-targets from four_highest_correlation_attribute
        self.X = self.dataframe[self.four_highest_correlation_attribute]

        # print('\nX feature matrix:\n', self.X)

        return self.X, self.y

    # def encoding(self): #categrorical variables are replaced with new columns with their categories/label and each row
    #     # has a 1 if that row corresponds to the category/label/unique category and 0 if it does not
    #     print('Before encoding, X column types:\n', self.X.dtypes)

    #     #wrangling step 5: encoding -> converting categorical data into numerical values for machine learning algorithms to process
    #     self.X = pd.get_dummies(self.X)

    #     print('encoding X:\n', self.X)

    def partitioning(self):
        #wrangling step 7: partioning the data into training and testing sets
        #train_test_split -> is a tool well function that will partition a dataset for you
        #random_state = a fixed seed of time that ensures everytime you run the code train_test_split
        #selects the same random subset of data for training and testing
        #stratify -> in train_test_split means that the proportions of a specific category(usually the y target variable)
        # are maintained in both the training and testing sets| ensure they have similar distributions of categories
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, train_size=0.7, random_state=111, stratify=self.y)

        # print("X_train:\n", self.X_train)
        # print("X_test:\n", self.X_test)
        # print("y_train:\n", self.y_train)
        # print("y_test:\n", self.y_test)

        return self.X_train, self.X_test, self.y_train, self.y_test

#This class handles the logistic regression components of training the model and making predictions and testing accuracy
class logisticRegression:
    #constructor -> initializing attributes X_train, X_test, y_train, y_test, and model
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = LogisticRegression() #initializing the model it's empty when initizalizing
    
    def modelTraining(self):
        #training the logistic regression model
        self.model.fit(self.X_train, self.y_train)

    def predictions_and_testingAccurary(self):
        #making a set of predictions
        y_predictions = self.model.predict(self.X_test)

        # print('y_predictions:')
        # print(y_predictions[:5])

        # print('y_test:\n', self.y_test[:5])

        #testing portion-> accuracy checking
        #accuracy_score is a sklearn method computes subset accuracy for y_true, y_pred
        accuracy = accuracy_score(self.y_test, y_predictions, normalize=True)

        #printing the accurary score
        # print('Accuracy:', accuracy)

        #returning the accuracy score
        return accuracy

    # Adding this after training the model to save the model
    def save_model(self, filename='logistic_regression_model.pkl'):
        # Saving the trained model to a file
        joblib.dump(self.model, filename)


   
def main():
    # write your code here
    file_path = 'diabetes_class.csv'
    
    #creating an instance(object) of the wrangleData class
    df_diabetes_class = wrangleData(file_path)

    #wrangling #2: wrangling unusable data
    df_diabetes_class.wrangle_unusable_data()

    #wrangling #3: quantative(correlation Matrix)
    df_diabetes_class.dfColumnCorrelationMatrix()

    #wrangling #4: seperate In and Out: spliting into feature vector and target
    df_diabetes_class.seperate_Xfeature_yTarget()

    #wrangling #7: partioning the data into training and testing sets
    df_diabetes_class.partitioning()

    #------------------wrangling was finished------------------------------

    #Creating an instance of LogisticRegression and passing partitioned data from wrangleData class
    df_diabetes_class_LogRegmodel =logisticRegression(df_diabetes_class.X_train, df_diabetes_class.X_test, df_diabetes_class.y_train, df_diabetes_class.y_test) 

    #training the model
    df_diabetes_class_LogRegmodel.modelTraining()

    #getting predictions and testing the acurrracy by evaluating it| 
    #getting the accuracy
    accuracy = df_diabetes_class_LogRegmodel.predictions_and_testingAccurary()

    #printing the accuracy
    print("Accuracy: ", accuracy)

    # saving the trained model
    df_diabetes_class_LogRegmodel.save_model()

    # Save accuracy to a text file
    with open("accuracy.txt", "w") as f:
    f.write(f"Model Accuracy: {accuracy:.2f}")



    #goal: train and test a Logistic Regression model based on the diabetes classifaction
    #data

    #performing classification on diabetes_class.csv dataset-> based on the four
    #most statistically-correlated attributes.

    #wrangle the data

    #print the accuracy score of the testing data when run through the model


if __name__ == '__main__':
    main()
