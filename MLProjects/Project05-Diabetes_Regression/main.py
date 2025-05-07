""" Ashley Zapata Minero
    Project05 - Diabetes Regression
    Description: This project is about making a linear Regression model and finding the line of best fit. First we start off by qualitative wrangling to see a visualization of the data,
    then we create a correlation matrix such as in the dfColumnCorrelationMatrix function, in order to find the highest correlated attribute by dropping the similar one and sorting the new dataframe
    then using idxmax to find the highest correlated attribute, to later use as a value that will be present in our graph, and training and model. We proceed to wrangling the data to have the data 
    clean, then we display the x(highest_correlation_attribute) and y-target data in the graph, then we create a linear regression model , and use .fit() to train it
    However to train it we need x(highest_correlation_attribute) to become X so we have to reshape it into a format -> 2D in which ski-kitlearn will accept it., then we
    we just use X and model_linreg(trained model from the function before -> LinearRegression) and use [min(X),max(X)](contain X values for best line of fit) for X line of best fit, and use .predict(X_lobf) to predict
    line of best fit for y(y_lobf). Then plot the data and add the labels.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

def qualitativeWrangling(df_diabetesReg):
    #This function is a representative of qualitative data where we iterate through the data and plot it to see a visual representaion
    #by using a for loop
    #dim-> dimensions made by number of columns
    #getting number of features(columns)
    dim = len(df_diabetesReg.columns)
    fig,ax = plt.subplots(dim,dim, figsize=(dim*2,dim*2)) #adjusting figure size based on dim

    #creating subplots (dim x dim)
    for i in range(dim):
        for j in range(dim):
            if i!=j:
                ax[i,j].scatter(df_diabetesReg.iloc[:,i], df_diabetesReg.iloc[:,j])
                ax[i,j].set_xlabel(df_diabetesReg.columns[i]) #label x-axis
                ax[i,j].set_ylabel(df_diabetesReg.columns[j]) #label y-axis
            else:
                ax[i,j].hist(df_diabetesReg.iloc[:,i], bins=10)
                ax[i,j].set_title(df_diabetesReg.columns[i]) #label histogram
    
    plt.tight_layout() #adjusting layout to prevent overlapping
    plt.savefig('qualitativeWrangling.png')


def dfColumnCorrelationMatrix(df_diabetesReg):
    #This function allows us to choose the most correlated attribute after creating a correlation matrix, we drop the column that could
    #affect the outcome, and by doing that we get either a new dataframe or series depending what type of data it is. In this case we are given
    #a new dataframe, and We sort out the values derives from previously dropped value, and look for the highest correlated Attribute
    #by using .idxmax

    #using correlation matrix-> how to use in slides to find the best fit coefficient 
    #take all attributes and make correlation values between each of those
    #need to extract it from correlation matrix

    #correlation matrix: in a dataframe shows how related each of the numeric attributes
    #is to each other numeric attribute

    #no need for a for loop just apply .corr() to get the correlation matrix, take the column you want and then sort them,then extract the top one that's not y
    #create a correlation matrix
    correlation_matrix = df_diabetesReg.corr()
    # print('correlation_matrix:\n', correlation_matrix)

    #dropping the row of 'Y', associated the column 'Y' that makes a new dataframe
    correlation_matrix_RC_Y= correlation_matrix['Y'].drop(index='Y')
    # print('correlation_matrix y index drop from y column:\n', correlation_matrix_RC_Y)
   
    #sorting out the values through descending order to see which have more correlation
    correlation_matrix_sorted = correlation_matrix_RC_Y.sort_values(ascending=False)
    # print('correlation_matrix_sort:\n', correlation_matrix_sorted)

    #Returning the most correlated attribute| idxmax() returns the highest value
    highest_correlation_attribute = correlation_matrix_sorted.idxmax()
    # print('highest correlation attribute:', highest_correlation_attribute)

    return highest_correlation_attribute

def wrangleData(df_diabetesReg, highest_correlation_attribute): #cleaning data/structuring
    #only want to keep feature X and target Y| 1 column

    #returning a new dataset with removed duplicates(should do 1st)
    df_diabetesReg = df_diabetesReg.drop_duplicates()
    

    # do not hardcode BMI
    #Only selecting attributes I want: BMI(highest_correlation_attribute), Y
    df_diabetesReg = df_diabetesReg[[highest_correlation_attribute,'Y']] #it might
    #professor probably dropped nulls


    #removing nulls 
    df_diabetesReg = df_diabetesReg.dropna()

    #professor removed the outliers
    
    return df_diabetesReg

def XAndYRepresentation(ax, df_diabetesReg, df_column):
    #Description: creating a scatterplot of the progression vs the single most statstically-correlated attribute

    #independent variable/x feature
    x = df_diabetesReg[df_column[0]].values
  
    #dependent variable/ y value
    y = df_diabetesReg[df_column[1]].values

    #adding the scatterplot for x and y values
    ax.scatter(x,y, label = "Diabetes Data")  

    return x, y


def linearRegression(x, y):
    #Description: doing reshaping of x to X, making an empty regression model, training the model with .fit()
    #getting b, m, and getting y_hat by by predicting X then returning model_linreg(linearRegression model)

    #need to reshape x to 2D because sklearn expects a 2D array
    X= x.reshape(-1,1)

    #Linear regression model - empty model, use an optimization algoritm that will minimize residuals
    model_linreg = LinearRegression()

    #model is being trained with .fit()
    model_linreg.fit(X,y)

    #end-result something can derive from the model| getting the intercept(b) and slope(m)
    b = model_linreg.intercept_ #b/b0 is intercept
    m = model_linreg.coef_[0] #m/b1 is the slope 
    
   #we don't need noise since we have real data

    #returning X feature (2D) and trained model model_linreg to make line of best fit
    return X, model_linreg

    
def lineOfBestFit(ax, X, model_linreg):
    #Description: creating a line of best fit with the model_linreg(linear Regression model) that was returned from the LinearRegression function 

    #contains X values of the line of best fit
    X_lobf = [min(X),max(X)]
    #asking for X feature vector to predict y
    y_lobf = model_linreg.predict(X_lobf)
    # print(y_lobf)
    
    #plotting the line of best fit for X feature BMI and y-target line of best fit
    ax.plot(X_lobf, y_lobf, label="Line of best fit", color="orange")
   

def main():
    # write your code here
    #goal: create a line of best fit of quantative diabetes progession
    #1st wranggle the data-> produce a single visualization: (1)scatterplot of the progression vs the single most stastically-correlated attribute
    #(2)A line of best fit based on the Linear Regression results.

    #collect the data
    file_path = "diabetes_regression.csv"
    #have to skip 1 row because doesn't really pertain, to access the whole data
    df_diabetesReg = pd.read_csv(file_path, skiprows=1)
    # print(df_diabetesReg.info())

    #columns 
    df_column = df_diabetesReg.columns
    # print('columns: ', df_column)

    #qualitative Wrangling
    # qualitativeWrangling(df_diabetesReg)

    #correlation matrix -> where columns are compared to one another to determine the highest_correlation_attribute based on the highest_correlation achieved
    #choose the most correlated attribute after creating a correlation matrix, we drop the column that could affect the outcome, and by doing that we get either 
    #a new dataframe or series depending what type of data it is. In this case we are given a new dataframe, and We sort out the values derives from previously 
    #dropped value, and look for the highest correlated Attribute by using .idxmax
    highest_correlation_attribute = dfColumnCorrelationMatrix(df_diabetesReg)

    #wrangling the data| #1 feature X 1 target Y | feature-> highest_correlation_attribute==BMI | target-> Y
    df_diabetesReg = wrangleData(df_diabetesReg, highest_correlation_attribute)
    # print(df_diabetesReg)
    
    #updated columns after wrangling the data
    df_column = df_diabetesReg.columns
    # print('columns: ', df_column)

    #plotting y vs x in a scatterplot
    fig, ax =plt.subplots(1,1, figsize=(16,9))

    #representing x and y in a plot| returning x and y for further use in linearRegression function
    x, y = XAndYRepresentation(ax, df_diabetesReg, df_column)

    #Running a linear regression between the target and the attribute most statistically highly correlated to the target
    X, model_linreg = linearRegression(x, y)

    #Creating a line of best fit model with the model_linreg(linear Regression model) that was returned from the LinearRegression function 
    lineOfBestFit(ax, X, model_linreg)
    
    
    ax.set(xlabel=highest_correlation_attribute, ylabel="Progression")
    fig.suptitle('Diabetes Data: Progression vs ' + highest_correlation_attribute + '(Linear Regression)')
   
    ax.legend() 
    fig.tight_layout()
    plt.savefig('linearRegression.png')


if __name__ == '__main__':
    main()
