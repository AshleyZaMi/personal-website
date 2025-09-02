""" Ashley Zapata Minero
    Project 9: NASA Kepler Exoplanet Candidate Classification
    Description: This project was about wrangling the data, but the most essential was the cross-validating tasks, there were two
    RandomizedSearchCV tasks one of them was for non-PCA X vector and another for PCA X vector. Within in that the goal was to find the best
    model from both of those, by comparing them both, and seeing their accuracies/performace. After that We used that optimized_model in order to use
    it for GridSearchCV in which the goal was to find the optimal hyperparameters for that given optimized_model and also we had to provide for GridSearchCv 
    whether PCA X vector or non-PCA X vector was better, which I had made into a function that was resuable called get_train_test_split. Furthermore, 
    RandomizedSearchCV and GridSearchCV were all made using the estimator_list criterion and saving resuauble code into lists. After the optimal_hyperparams 
    were found then an acutal final_optimized_model could be performed by fitting the data, and with that a confusion matrix was made and plotted. I also made 
    a classification report based on the final_optimized_model. Finally I answered the 5 sets of questions, and for question 4 i had also made a function that
    had permutation_feature_importances which I think was crucial to answering number 4. I was able to print it within reflection_questions function. All the functions were 
    put into a class in order to facilliate the process of reusing code and to make it more dynamic.
"""
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from math import sqrt
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt


class NASA_Exo_Cand_Classification:
    def __init__(self, file_path):
        self.file_path = file_path #storing the file_path from main
        self.dataframe = self.gather_data() #initializes the dataframe with gathering of the data having been done
        self.wrangling()
        self.X = self.dataframe.drop(columns=['koi_disposition','koi_pdisposition']).select_dtypes(include=[int,float]) #using wrangled dataset for PCA
        self.y = self.dataframe['koi_pdisposition']
        # self.X_pca_result = self.apply_pca()
    
    #gathering the data from the given file_path in the constructor
    def gather_data(self):
        return pd.read_csv(self.file_path, delimiter=",",header=41)


    #wrangling the dataset
    def wrangling(self):
        self.dataframe = self.dataframe.drop_duplicates()
        self.dataframe = self.dataframe.drop(columns=['koi_period_err1', 'koi_period_err2', 
                                                    'koi_eccen_err1', 'koi_eccen_err2', 'koi_duration_err1', 
                                                    'koi_duration_err2','koi_prad_err1', 'koi_prad_err2', 
                                                    'koi_sma_err1', 'koi_sma_err2', 'koi_incl_err1', 
                                                    'koi_incl_err2','koi_teq_err1', 'koi_teq_err2', 
                                                    'koi_dor_err1', 'koi_dor_err2', 'koi_steff_err1', 
                                                    'koi_steff_err2', 'koi_srad_err1', 'koi_srad_err2', 
                                                    'koi_smass_err1', 'koi_smass_err2'])
        self.dataframe = self.dataframe.dropna()
        

     #to print the dataset itself
    def print_dataframe(self):
        print(self.dataframe.head(10))

    def train_test_split_func(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        return X_train, X_test, y_train, y_test
    

    #cross validating to find 1) optimal model and 2) its hyperparameters
    #creating pca_result which contained X vector that is Pcaed
    def apply_pca(self, X_train, X_test):
        #Standardizing the data first--it's important for PCA
        scaler=StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train) #fitting and transforming the training set
        X_test_scaled = scaler.transform(X_test) # only transforming the test set

        #Applying PCA to standardized X feature vector
        #decomposing the feature vector to 2D
        components= 3 #keeping 95% variance, because it is important to retain most information 95% variance minimizes information lost and preserves most of the data
        pca = PCA(n_components=components)

        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        
        #storing the pca columns
        pca_columns = []

        for i in range(components):
            col_name = 'PCA' + str(i+1)
            pca_columns.append(col_name)
        
        df_X_train_pca = pd.DataFrame(X_train_pca, columns=pca_columns, index=X_train.index)
        df_X_test_pca = pd.DataFrame(X_test_pca, columns=pca_columns, index=X_test.index)

        #converting to dataframe for easier use later and returning ot with pca columns
        return df_X_train_pca, df_X_test_pca

    def get_train_test_split(self, X, y, use_pca = False):
        #splitting
        X_train, X_test, y_train, y_test = self.train_test_split_func(X, y)

         #choosing the appropriate dataset (X_pca_result or X(non-PCA))
        if use_pca:
            df_X_train_pca, df_X_test_pca = self.apply_pca(X_train, X_test)
            return df_X_train_pca, df_X_test_pca, y_train, y_test
        else:
            #Applying Standard Scaling
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            #converting scaled outputs to DataFrames
            df_X_train = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
            df_X_test = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)
            return df_X_train, df_X_test, y_train, y_test
        
    
    #finding optimal hyperparameters, and model by cross validaing the non PCA data with the 
    #estimators of LogisticRegression, KNeighborsClassifier, DecisionTreeClassifier, and SVC
    #just looking for the accuracy from this and the estimator comparison
    def non_PCA_cross_validate(self):
        #splitting into train test split here
        X_train, X_test, y_train, y_test= self.get_train_test_split(self.X, self.y, use_pca=False)
        #storing results
        results=[]
        #calculating the upperlimit for knn
        knn_upperlimit = int(1.5*sqrt(len(X_train)))

        pipe = Pipeline([
            # ('scaler',StandardScaler()),
            ('estimator', None) #placeholder for the estimators
        ])
       
        #creating a list of estimators and their hyperparameters
        estimator_list =[
        {
            'estimator':[LogisticRegression()]
        },
        {
            'estimator':[KNeighborsClassifier()],
            'estimator__n_neighbors':range(1, knn_upperlimit + 1)
        },
        {
            'estimator':[DecisionTreeClassifier()],
            'estimator__criterion':['entropy','gini'],
            'estimator__max_depth':range(3,19)
        },
        {
            'estimator':[SVC()],
            'estimator__kernel':['rbf'],
            'estimator__C':[0.01, 0.1, 1, 10],
            'estimator__gamma':[1, 10, 100]
        }
        ]

        #saving this list for later to use in gscv
        self.non_PCA_estimators_list = estimator_list

        #calculating the hyperparameter space size
        hyperparam_space_size = 0
        for estimator_dict in estimator_list:
            size=1 #this is resetting for each estimator
            for value in estimator_dict.values():
                if isinstance(value, (list,range)):
                   size *= len(value)
             #summing the size across all of them
            hyperparam_space_size += size

        #calculating the number of iterations to search at least 10% of the total hyperparameter space
        num_iterations = max(1, int(0.1*hyperparam_space_size))

        #creating a cross_validator by creating a RandomizedSearchCV object:
        rscv = RandomizedSearchCV(
            pipe, #pipeline object
            param_distributions= estimator_list,
            scoring = 'accuracy', #using accuracy for evaultion
            n_iter = num_iterations, #this is the number of times it will iterate
            # random_state = 41
        )

        #fitting the model to the training data
        rscv.fit(X_train, y_train)

        #outputting the best parameters and scores from rscv
        ## print('best params non-PCA:\n ', rscv.best_params_)
        ## print('scores non-PCA:\n ', rscv.cv_results_['mean_test_score'])
        ## print('Best RandomSearchCV non-PCA cross-validation score:\n ', rscv.best_score_)
       
       #storing the params in a list containing a dictionary  which contain values obtained
        #from RandomizedSearchCV. .append is adding these to the end of the list
        results.append({
            'best_estimator': rscv.best_estimator_,
            'best_score': rscv.best_score_
        })

        non_PCA_results_list_dict = results
       
        return non_PCA_results_list_dict

     #just looking for the accuracy from this and the estimator comparison
    def PCA_cross_validate(self):
        #splitting into train test split here
        X_train, X_test, y_train, y_test=self.get_train_test_split(self.X, self.y, use_pca=True)
        #storing results
        results=[]
        #calculating the upperlimit for knn
        knn_upperlimit = int(1.5*sqrt(len(X_train)))

        pipe = Pipeline([
            ('estimator', None) #placeholder for the estimators| not applying StandardScaler because it will use the X_pca_result that scaled X
        ])
       
        #creating a list of estimators and their hyperparameters
        estimator_list =[
        {
            'estimator':[LogisticRegression()]
        },
        {
            'estimator':[KNeighborsClassifier()],
            'estimator__n_neighbors':range(1, knn_upperlimit + 1)
        },
        {
            'estimator':[DecisionTreeClassifier()],
            'estimator__criterion':['entropy','gini'],
            'estimator__max_depth':range(3,19)
        },
        {
            'estimator':[SVC()],
            'estimator__kernel':['rbf'],
            'estimator__C':[0.01, 0.1, 1, 10],
            'estimator__gamma':[1, 10, 100]
        }
        ]
        #saving this estimator list for later to use in gscv
        self.PCA_estimators_list = estimator_list

        #calculating the hyperparameter space size
        hyperparam_space_size = 0
        for estimator_dict in estimator_list:
            size=1 #this is resetting for each estimator
            for value in estimator_dict.values():
                if isinstance(value, (list,range)):
                   size *= len(value)
             #summing the size across all of them
            hyperparam_space_size += size

        #calculating the number of iterations to search at least 10% of the total hyperparameter space
        num_iterations = max(1, int(0.1*hyperparam_space_size))

        #creating a cross_validator by creating a RandomizedSearchCV object:
        rscv = RandomizedSearchCV(
            pipe, #pipeline object
            param_distributions= estimator_list,
            scoring = 'accuracy', #using accuracy for evaultion
            n_iter = num_iterations, #this is the number of times it will iterate
            # random_state = 41
        )

        #fitting the model to the training data
        rscv.fit(X_train, y_train)

        #outputting the best parameters and scores from rscv
        ## print('best params PCA:\n ', rscv.best_params_)
        ## print('scores PCA:\n ', rscv.cv_results_['mean_test_score'])
        ## print('Best RandomSearchCV PCA cross-validation score:\n ', rscv.best_score_)
        
        #storing the params in a list containing a dictionary  which contain values obtained
        #from RandomizedSearchCV. .append is adding these to the end of the list
        results.append({
            'best_estimator': rscv.best_estimator_,
            'best_score': rscv.best_score_
        })

        PCA_results_list_dict = results
        return PCA_results_list_dict


    def optimized_model_selection(self, non_PCA_results_list_dict, PCA_results_list_dict):
        non_PCA_best_score =non_PCA_results_list_dict[0]['best_score']
        PCA_best_score = PCA_results_list_dict[0]['best_score']
        
        if non_PCA_best_score > PCA_best_score:
            optimized_model= non_PCA_results_list_dict[0]['best_estimator']
            optimized_score = non_PCA_results_list_dict[0]['best_score']
            optimized_Xdataset = False #don't use PCA dataset
        else:
            optimized_model = PCA_results_list_dict[0]['best_estimator']
            optimized_score = PCA_results_list_dict[0]['best_score']
            optimized_Xdataset = True #use PCA data
        
        ## print('optimized_model: ', optimized_model)
        ## print('optimized_score: ', optimized_score)
        ## print('optimized_Xdataset', optimized_Xdataset) 

        #intializing the list
        optimized_model_list_dict = []

        #adding to the list
        optimized_model_list_dict.append({
            'optimized_model':optimized_model,
            'optimized_Xdataset':optimized_Xdataset
        })
        
        return optimized_model_list_dict
    
    def optimal_hyperparams_cross_validation(self, optimized_model_list_dict):
        #using X or X_pca_result depending on optimized_model_selection
        use_pca = optimized_model_list_dict[0]['optimized_Xdataset']
        X_train, X_test, y_train, y_test = self.get_train_test_split(self.X, self.y, use_pca= use_pca)

        #using the correct estimator list
        if use_pca == True:
            estimator_list = self.PCA_estimators_list
        else:
            estimator_list = self.non_PCA_estimators_list

        #getting the optimal model's class pipeline
        best_model_pipeline = optimized_model_list_dict[0]['optimized_model']
        #getting the estimator from the pipeline
        best_estimator_from_pipeline = best_model_pipeline.named_steps['estimator']

        #finding the correct estimator list to assign it the correct param_grid for gscv
        param_grid = {}
        for estimator_dict in estimator_list:
            estimator= estimator_dict['estimator'][0]
            #if the estimator that was from the estimator_list is the same as the best_estimator_from_pipeline from optimzed_model which it the best one
            if type(estimator) == type(best_estimator_from_pipeline):
                #then if it it is the same then we assign that one as the param_grid dict
                param_grid = estimator_dict

        
        #making a new Pipeline
        if use_pca == True:
            pipe = Pipeline([
                ('estimator', best_estimator_from_pipeline)
            ])
        else:
            pipe = Pipeline([
                # ('scaler', StandardScaler()),
                ('estimator', best_estimator_from_pipeline)
            ])

        #conducting GridSearchCV
        gscv = GridSearchCV(pipe,param_grid=param_grid, scoring='accuracy', cv=5)

        gscv.fit(X_train, y_train)

        ## print('Best hyperparameters from Grid Search:\n ', gscv.best_params_)
        ## print('Best GridSearchCV cross-validation score:\n ', gscv.best_score_)

        #initializing gscv list/dict
        gscv_hyperparams_list_dict = []

        #appending the values to it
        gscv_hyperparams_list_dict.append({
            'optimal_hyperparams': gscv.best_params_,
            'best_score': gscv.best_score_
        })
        return gscv_hyperparams_list_dict
    
    def final_optimized_model(self, optimized_model_list_dict, gscv_hyperparams_list_dict):
        use_pca = optimized_model_list_dict[0]['optimized_Xdataset']
        X_train, X_test, y_train, y_test = self.get_train_test_split(self.X, self.y, use_pca=use_pca)
        #getting the optimal hyperparameters in order to get the estimator
        best_params = gscv_hyperparams_list_dict[0]['optimal_hyperparams']
        best_estimator = best_params['estimator']

        #getting the actual hyperparameters
        hyperparams={}

        #removing 'estimator' in order to only get the hyperparameters
        for key, value in best_params.items():
            if key != 'estimator':
                new_key = key.replace('estimator__','')
                hyperparams[new_key] = value

        #rebuilding and training the final model
        #setting up the model
        model = best_estimator.set_params(**hyperparams)
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        
           
         
        #confusion matrix
        con_matrix = confusion_matrix(y_test, y_pred, labels =model.classes_) #model.classes_
        #creating a confusion matrix display object
        con_matrix_display = ConfusionMatrixDisplay(confusion_matrix=con_matrix, display_labels=model.classes_)

        #plotting the confusion matrix display object
        fig, axes = plt.subplots(1,1, figsize=(12,10))
        con_matrix_display.plot(ax=axes)

        #saving the figure
        axes.set(title ='Kepler Exoplanet Candidate Classification')
        fig.savefig('NASA_kepler_exoplanet_candidate_classification.png')

        #classification report
        print('Classification Report:\n ')
        print(classification_report(y_test, y_pred))

        return model

    def permutation_feature_importance(self, model,optimized_model_list_dict):
        use_pca = optimized_model_list_dict[0]['optimized_Xdataset']
        X_train, X_test, y_train, y_test = self.get_train_test_split(self.X, self.y, use_pca=use_pca)
        #using hassattr in order to check if the model contains feature_importances_ those such as decision trees do
        #but other models might not

        #applying permutation_importance 
        permutation_import_result = permutation_importance(model, X_test, y_test, n_repeats=50, random_state=0)

        #this if else is accounting if X_test after PCA is still a numpy array
        if isinstance(X_test, pd.DataFrame):
            feature_names = X_test.columns #this would only work if it is a pandas dataframe
        else: #do this if it is a numpy array, will convert it to an a regular list of features, generating names such as feature_1, feature_2
            feature_names=[]
            for i in range(X_test.shape[1]):
                feature_names.append('feature_'+str(i))

        #creating a dataframe that holds the permutation_importances of feature_names
        df_perm_importances = pd.DataFrame({
            'feature': feature_names,
            'importance': permutation_import_result.importances_mean
        }).sort_values(by='importance', ascending=False)
        
        # print("\n Top 10 important features: ")
        # print(df_perm_importances.head(10).to_string(index=False))
        print(" Top important feature: ")
        print(df_perm_importances.head(1).to_string(index=False))

    
    def reflection_questions(self, model,optimized_model_list_dict):
        #Question 1
        print('1) No, PCA did not improve upon the results from when I did not use it. If I were to compare it to the results I got from both PCA and non-PCA RandomizedSearchCV, showed a clear difference in its performance. In PCA the best_score was around 0.7231, while non-PCA best_score was around 0.8908. Through this it can be seen that non-PCA was actually improved more than that of PCA.')
        #Question 2
        print('2) the key characteristic of PCA is its ability of helping reduce the curse of dimensionality while still maintaining variance, by combining features into principal components. So because it focuses on variance and not on target labels, it can perhaps dillute the data by removing important data for classification--perhaps some features held significant power or weight. Another thing is that PCA assummes linear relationships which may be difficult when working with complex datasets such as this one of NASA.')
        #Question 3
        print('3) Yes the model was able to classify the objects equally across labels. This can be seen in the classification report, where the F1-score for CANDIDATE is around 0.80, and for FALSE POSITIVE it is around 0.79, which are very close. Similarily, the scores for precision and recall seem to be balanced. We can tell this especially because f1-score is a metric that combines precision and recall into one number, so if the number is particularly high it suggest that there is a class balance. In this case since the F1-Score for CANDIDATE and FALSE POSITIVE are nearly equal it suggests that the model is performing consistently across the two classes. Therefore, there is no strong indicator of imbalance happening across labels.')
        #Question 4
        print('4) The attribute that significantly influences whether or not an object is an exoplanet is: ')
        self.permutation_feature_importance(model,optimized_model_list_dict)
        #Question 5
        print('5) The most important attribute appears to be koi_prad, which represents planetary radius. This makes sense as a strong predictor because the size of the object is directly related to if it resembles known exoplanets. If the radius is too small, it might be noise or a starspot on the star\'s surface. If the radius is too large, it could be a binary star--two stars that orbit around a common center, or another type of celestial object such as stars(like the sun), planets, moons, asteroids, black holes etc. I think the model particularly finds the koi_prad feature really informative because known exoplanets tend to fall with certain size ranges, making koi_prad a reliable feature to use for the model. This feature in particular suggests that planetary size plays a key role in distinguishing real exoplanets from false positives in the kepler exoplanet dataset.')
        

def main():
    # write your code here
    file_path = "cumulative_2023.11.07_13.44.30.csv"
    #making a NASA object
    Nasa_obj = NASA_Exo_Cand_Classification(file_path)
    
    ## Nasa_obj.print_dataframe()

    #storing the returned values from non-PCA cross val and PCA cross validation
    #the return values are list containing dictionaries with values such as best_estimator_ and
    #best_score_
     #printing out the non PCA cross validating
    non_PCA_results_list_dict = Nasa_obj.non_PCA_cross_validate()
    #printing out the PCA cross validating
    PCA_results_list_dict = Nasa_obj.PCA_cross_validate()

    #this function compares the PCA and nonPCA cross validattion of RandomizedSearchCV, and returns the best model in a list
    optimized_model_list_dict = Nasa_obj.optimized_model_selection(non_PCA_results_list_dict, PCA_results_list_dict)

    #this function uses the optimized_model_list to be able to do GridSearchCV to find the best hyperparameters, and returns a list with them
    gscv_hyperparams_list_dict = Nasa_obj.optimal_hyperparams_cross_validation(optimized_model_list_dict)

    #creating the final optimized model using both the optimal model and the optimal hyperparams done from the previous processes
    model = Nasa_obj.final_optimized_model(optimized_model_list_dict, gscv_hyperparams_list_dict)

    #answering the reflection questions
    Nasa_obj.reflection_questions(model,optimized_model_list_dict)


if __name__ == '__main__':
    main()
