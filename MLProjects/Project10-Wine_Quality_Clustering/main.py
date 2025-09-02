""" Ashley Zapata Minero
    Project 8 - Wine Quality Clustering
    Description: This project is about training and testing a K-Means clustering model based on red wine quality dataset. 
    First, wrangling was done by dropping duplicates, and nulls, and dropping the wine index since we don't need it during 
    unsupervised machine learning. We also seperate the quality of the wine in a series, and then drop it from the data frame.
    Storing it in a series because it will be used later on. We then normalize the data since we're working with a unsupervised 
    machine learning model. We set up an inertia function that uses KMean algorithm, and attains inertia by doing .inertia_.
    Inertia is where we evaluate the  measuring how tightly clustesterd the datapoints are within each cluster. In the function, we
    look for the best_k_value by setting up lists, to retain k and inertia value. We had also set up 1-10 bound for k. After best_k_value
    was found we proceeded to plot the graph which is k number of clusters vs inertia, essentially using the previous k_and_inertias_list 
    that contains (k, inertias) to plot it. Then we incorporate best_k_value into a silhouette_score function. Silhoute score evaluates how well 
    seperated data points are within their clusters compared to other clusters. Knowing this and using the best_k_value which is the optimal k from
    1-10 best_k_value becomes the number of clusters for KMeans algorithm in the silhouette_score function. By training K Mean algorithm we can extract
    labels related to sihoute by using cluster.labels_(holding the cluster assignment for each data point in training set), which we use this to now 
    put it in the silhoute_score and assigned it to score_sil to  provide a more comprehensive understanding of the clustering algorithms. We also printed
    a contingency table in order to compare the attributes from the X that were clustered into KMean to just quality data that has no relation to the X since
    it didn't train on that data and has no relation to the quality. Then questions were answered.
"""
import pandas as pd
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

def wrangling():
    #making the file path
    file_path = "wineQualityReds.csv"
    df_redWineQuality = pd.read_csv(file_path)
    #dropping duplicates
    df_redWineQuality = df_redWineQuality.drop_duplicates()
    #dropping nulls
    df_redWineQuality.dropna(inplace=True)

    #dropping the wine column since it's just an index
    df_redWineQuality = df_redWineQuality.drop(columns=['Wine'])

    #storing quality in a seperate Series(for use later)
    quality_Series = df_redWineQuality['quality']

    #dropping quality from main DataFrame -> df_redWineQuality
    df_redWineQuality = df_redWineQuality.drop(columns=['quality'])
    
    #the dataframe is essentially X because we're working with unsupervised machine learning
    X = df_redWineQuality

    #transforming all columns
    transform_norm = Normalizer()

    X_transformed = transform_norm.fit_transform(X)

    #X_transformed is an ndarray

    X = pd.DataFrame(X_transformed, columns=X.columns, index=X.index)

    # X_train, X_test = train_test_split(X, test_size=0.25, random_state=33)

    return X, quality_Series #, X_train, X_test

def inertia(X):
    k_and_inertias_list = []
    #goes from 1 to 10 because 11 is exclusive
    k_range = range(1,11)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=33)
        kmeans.fit(X)
        k_and_inertias_list.append((k, kmeans.inertia_))
    
    #finding optimal k for inertia because we have a list with k and inertias
    #most optimal would be the lowest

    #using the intial values in list to reference
    best_k_value,optimal = k_and_inertias_list[0]

    #then comparing to get the most lowest in the list in the for loop
    for k,inertia in k_and_inertias_list:
        if inertia < optimal:
            optimal = inertia
            best_k_value = k

    # print(best_k_value)

    # print(k_and_inertias_list)
    return best_k_value, k_and_inertias_list

def inertia_vs_number_of_clusters_k_plot(k_and_inertias_list):
    fig, ax = plt.subplots(1,1, figsize=(13,9))
    k_vals = []
    inertia_vals = []

    for k,inertia in k_and_inertias_list:
        k_vals.append(k)
        inertia_vals.append(inertia)

    ax.plot(k_vals, inertia_vals, marker='o')
    ax.set(xlabel = 'Number of clusters (k)', ylabel='Inertia')
    fig.suptitle('Red Wines: Inertia vs Number of Clusters')
    fig.tight_layout()
    plt.savefig('Red Wines: Inertia vs Num of Clusters.png')


def optimal_k_in_silhoutte_score(X,best_k_value, quality_Series):
    #optimal_k value gotten from inertia model
    optimal_k = best_k_value
    #setting up the cluster model to this time use the optimal_k
    kmeans_model = KMeans(n_clusters=optimal_k, random_state=33)
    kmeans_model.fit(X)
    #cluster labels from KMeans model
    cluster_labels = kmeans_model.labels_
    #calculating the silhouette_score using cluster labels
    score_sil = silhouette_score(X, cluster_labels)
    # print(cluster_labels)
    # print(quality_Series)


    #creating a crosstab(contingency table) cluster_number/cluster_labels 
    #vs original quality Series to compare them
    cont_table = pd.crosstab(cluster_labels, quality_Series)
    print("Contingency table:")
    print(cont_table)
  
def questions(best_k_value):
    print('\nQuestion [1] - \'optimal\' k: ', best_k_value)
    print('Question [2] -  no, the clusters don\'t directly represent the quality of wine')
    print('Question [3] - The reason is that because we\'re comparing the patterns of features that the wines were clustered into using KMeans, which has no relation to the quality of wine, since the clustering is not based on the quality of wine. So we\'re essentially comparing the original quality of wine with the clusters generated by the KMeans algorithm, which won\'t align with one another because the clustering is based on different features than quality of wine.')

    



def main():
    # write your code here
    #in this step we are doing all the necessary steps to wrangle the data for an 
    #unsupervised machine learning algorithm
    X, quality_Series = wrangling()
    
    #inertia is measuring how tightly clustesterd the datapoints are within each cluster, so in this function, 
    #find the best k value from a range of 1-10 using inertia as the quantifier and we get best_k_value
    #and a k_and_inertias_list that contains (k,inertia)
    best_k_value, k_and_inertias_list = inertia(X)

    #this function just graphs the inertia values and the number of clusters
    inertia_vs_number_of_clusters_k_plot(k_and_inertias_list)

    #this function uses silhoute_score but also using the best_k_value, in order to be able to 
    #raise the question in the next function. Silhoute score evaluates how well seperated data points are 
    #within their clusters compared to other clusters. Typically inertia and silhoute_score can be used 
    #independenly but they can also be used together to provide a more comprehensive understanding of the clustering algorithms
    optimal_k_in_silhoutte_score(X,best_k_value, quality_Series)

    #here the answers are just answered based on the contingency table and other questions.
    questions(best_k_value)




if __name__ == '__main__':
    main()
