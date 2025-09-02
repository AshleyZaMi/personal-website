""" Ashley Zapata Minero
    Project 7 - Titanic Classification
    Description: in this project I used KNN algorithm. First I wrangled the data, making
    sure to keep only variables relevant to the end goal which is making a prediction for a person
    to predict if they were alive or not. In the wrangling of the data I split X into age, pclass and sex. I also dropped duplicates and nulls and did one hot handling
    by using get_dummies on pclass and sex.For y I only put survive since that is the target for this supervised model. I made sure to apply 
    the standard_scaler only on the X_train and X_test.For X_train I used fit_transform and for X_test I used transform, the reason being that for
    no data leakage to occur and so that the model wouldn't break transform should only be applied to X_train. Also since StandardScaler
    transforms the data to np data type then I converted back into a pandas dataframe to be able to use it in further functions.  I set up the KNN model, and created an k_upperlimit
    to have an upperlimit based on the rows and using the pandas math library by a range of k neighbors 1.5. I iterated through a for loop from 1 to the k_upperlimit + 1 and applied cross validation
    . I choose cross validation to add more variance, and so that the splits wouldn't be biased, and I used scoring as accuracy and used the mean of all the splits. Essentially the cross validation also
    does the training and predicting but during the amounts of splits, and it gives a better view in a more holisitic manner by removing as much bias as possible. Through using this I stored the k values, and 
    also the accuracy represented by the mean in each split and appended it all one by one in the k_accuraccies_list . After I created a final_knn_model by using that list and finding the best_k_val by iterating through the list
    in a pair manner and storing the most highest one, so then I can use that k for the neighbor in my the set up of the KNN model. I then did the training(using x and y training data) and predicting(using testing data),  and also evaluated the 
    final_knn_model's accuracy. I then plotted the graph of accuracy_vs_num_neighbors_plot and spliting the k values and accuracies form the list I was able to plot them in a graph to represent the predicton accuracy in relation to the number of neighbors
    I then made a data_prediction to see if a 31-year old female passenger in 2nd class had survived by creating a dataframe for her. I then transformed it using the previous variable I had saved for StandardScaler called tfr_standard putting the .transform to the
    prediction_df, and since doing standardization converts it to nd array then I converted it back to a dataframe with column names used in X_train.columns to be able to match it to previous data it used to train. And was able to make a prediction. I printed out the 
    prediction output.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

def correlation_matrix():
    file_path='titanic.csv'
    df_titanic = pd.read_csv(file_path)

    #selecting only numeric columns
    df_titanic_numeric = df_titanic.select_dtypes(include=[np.number])


    correlation_matrix = df_titanic_numeric.corr()
    correlation_matrix = correlation_matrix['survived'].drop(index='survived')
    correlation_matrix_sorted = correlation_matrix.sort_values(ascending=False)
    print(correlation_matrix_sorted)



def wrangling():
    #gathering the data
    file_path='titanic.csv'
    df_titanic = pd.read_csv(file_path)

    #dropping nulls and duplicates
    df_titanic = df_titanic.drop_duplicates()

    # print("after dropping duplicates:\n", df_titanic)

    #dropping non-usuable features
    #drop columns except and only keep pclass, age, sex
    df_titanic = df_titanic[['pclass','age','sex','survived']]
    #encoding(to sex and embark since they are non-numerical)
    #using get_dummies because we are using it for euclidian distance so it is important for them
    #to retain their original meaning|drop_first -> removes multicullinarity 
    df_titanic = pd.get_dummies(df_titanic, columns=['pclass','sex'], dtype=float)
    
    # print('before dropping:\n ', df_titanic)
    df_titanic.dropna(inplace=True)
    # print('\nafter dropping:\n ', df_titanic)

    #X feature and Y feature
    X = df_titanic.drop(columns=['survived'])
    # print(X.dtypes)
    y = df_titanic['survived']
    # print("\n X data tyes: ", X.shape)
    
    #transforming data-
    #since we're working with a supervised learning model we will use 
    #standardization so that no attribute will dominate the input vector
    # will have a mean of 0 an a standard deviation of 1

    tfr_standard = StandardScaler()

     #split and training set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, stratify=y)

   
    #fiting and transforming on train
    X_train_transformed = tfr_standard.fit_transform(X_train)
    #just transforming on test, if not it might break the model and cause data leakage
    X_test_transformed = tfr_standard.transform(X_test)

    #converting transformed ndarrays to dataframes
    X_train = pd.DataFrame(X_train_transformed, columns=X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(X_test_transformed, columns=X_test.columns, index=X_test.index)
    
    # X_transformed = tfr_standard.fit_transform(X)
    # print("X_transformed: ", X_transformed)

    # #X_transformed is an ndarray so converting into dataframe
    # X = pd.DataFrame(X_transformed, columns=X.columns, index=X.index)
    # print('X_transformed_df:\n', X)

    return X, y, X_train, X_test, y_train, y_test, tfr_standard

def KNN(X_train, X_test, y_train, y_test):
    #setting up k's maximum number/upper limit| which is 1.5 times the sqrt of the number 
    # of records
    k_upperlimit=int(1.5*math.sqrt(X_train.shape[0]))

    #storing all the accuracies in k in a list
    k_accuraccies_list = []

    #iterating through 1 to k_upperlimit
    for k in range (1, k_upperlimit + 1):
        #initlizaing the model which will hold each individual k
        model_knn = KNeighborsClassifier(n_neighbors=k)

        # #normal approach
        #train(learn mean and SD fit -> training data, fit and transform training and testing data)
        # model_knn= model_knn.fit(X_train, y_train)

        # #predict
        # y_predictions = model_knn.predict(X_test)

        # #accuracy
        # accuracy = accuracy_score(y_test, y_predictions, normalize=True)

        # #appending all the accuracies for each k
        # k_accuraccies_list.append((k,accuracy))

        #--------------------------------------
        #cross validation approach
        #doing cross validation across many splits to have more real world data especially for the accuracies of k
        #this also does the training and predicting but within 5 folds and makes it have less variance
        cross_val_scores= cross_val_score(model_knn, X_train, y_train, scoring='accuracy', cv=5)

        # print("cross_vale_scores:", cross_val_scores)
        #since cross val has folds, it is better to gather the mean of all the scores in that fold and assign it to 
        #the one iteration it is on to store latter in the array
        fold_mean_accuracy = cross_val_scores.mean()

        #appending all the accuracies for each k
        k_accuraccies_list.append((k,fold_mean_accuracy))
     
    
    # print('accuraccies_list: ',k_accuraccies_list)
    return k_accuraccies_list

def final_knn_model(X_train, X_test, y_train, y_test, k_accuraccies_list):
    #this will allow us to have a value to help us start to determine which is the best k value
    most_accurate= -1
    #unpacking the tuples found in the k_accuraccies_list
    for k,accuracy in k_accuraccies_list:
        if accuracy > most_accurate:
            most_accurate = accuracy
            # print("accuracy:", (k,accuracy))
            best_k_val = k

    print('best K value: ', best_k_val)

    #setting up model
    final_model_knn = KNeighborsClassifier(n_neighbors=best_k_val)
    #training the model 
    final_model_knn = final_model_knn.fit(X_train, y_train)

    #y_predictions
    y_predictions = final_model_knn.predict(X_test)
    
    #accuracy
    knn_accuracy = accuracy_score(y_test, y_predictions, normalize=True)
    print("final knn accuracy: ",knn_accuracy)
    #printing classification report just to see the performance
    # print(classification_report(y_test, y_predictions))

    return knn_accuracy, final_model_knn

def accuracy_vs_num_neighbors_plot(k_accuraccies_list):
    fig, ax = plt.subplots(1,1, figsize=(16,9))
    #intializing lists to store k values and accuracy values seperately
    k_vals=[]
    accuracy_vals = []
    #iterating through knn_accuracies_list that contains (k,accuracy) pairs
    for k , accuracy in k_accuraccies_list:
        k_vals.append(k)
        accuracy_vals.append(accuracy)
        # print(k, accuracy)



    ax.plot(k_vals, accuracy_vals)
    ax.set(xlabel='Number of Neighbors(k)', ylabel='Accuracy Score')
    fig.suptitle('Accuracy score vs number of neighbors, Titanic dataset')
    fig.tight_layout()
    plt.savefig('KNN\'s accuracy score vs num neighbors.png')


def data_prediction(final_model_knn, tfr_standard, X_train):
    #Make a prediction using this data 
    #creating a new dataframe that has the same column names for the 31 year old female passenger
    #in 2nd class
    # print('X_train:\n', X_train)
    prediction_df = pd.DataFrame([{
            'age':31, #she is 31
            'pclass_1.0':0,
            'pclass_2.0':1, #she is in second class
            'pclass_3.0':0, #she is not in third class
            'sex_female':1, #she is female
            'sex_male':0 #she is not male
          }
        ])

    #transforming - using tfr_standard to use StandardScaler to the information to standardize it
    #just like the model itself was so it is standardizing making it to np
    transformed_prediction = tfr_standard.transform(prediction_df)

    #because standardization was done need to convert it back to a dataframe with column names
    transformed_prediction = pd.DataFrame(transformed_prediction, columns=X_train.columns)
    
    #using the transformed data(stnadardized) to make a prediction using the final_model_knn
    prediction = final_model_knn.predict(transformed_prediction)

    print("31-year-old female passenger in 2nd Class: ", prediction[0])
  


def main():
    # write your code here
    # correlation_matrix()
    #this is just doing wrangling such as drop_duplicates, dropna, transforming(by using StandardScaler), cutting into training and testing data into partitions
    X, y, X_train, X_test, y_train, y_test, tfr_standard = wrangling()

    #KNN function| this one makes a model and gets all the possibilites from 1.5 *math.sqrt of k, and run through 
    # a loop from 1 to the total amount of possibilies and collect all the accuracies for k in a list
    #end result: is a k_accuracies_list
    k_accuraccies_list = KNN(X_train, X_test, y_train, y_test)

    #retrained the model and used k_accuraccies_list to train it and do a prediction
    knn_accuracy, final_model_knn = final_knn_model(X_train, X_test, y_train, y_test, k_accuraccies_list)

    #plotting accuracy vs number of neighbors
    accuracy_vs_num_neighbors_plot(k_accuraccies_list)

    #make a prediction using the final_model_knn about a women who is 31-year-old female passenger in 2nd class
    prediction = data_prediction(final_model_knn, tfr_standard, X_train)

   
if __name__ == '__main__':
    main()
