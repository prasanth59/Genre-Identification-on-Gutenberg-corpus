#Model selection , training and prediction of genre of books.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame, Series
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import f1_score, make_scorer,accuracy_score

# Evaluation
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

# Hyper param tuning
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# Classification Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import class_weight
from sklearn.svm import SVC

#File conists of the extracted features for each book and it's genre.
filePath = r'featurefile.csv'

book_details = pd.read_csv(filePath, engine='python')
book_details.set_index('book_id', drop=True, inplace=True)

# REMOVING BOOKS WITH NO GENRES
book_details = book_details[book_details['genre']!='NONE']

features = book_details.values[:, 1:-1]
genre = book_details.values[:,-1]

#Perform train,test split on the data.
train_features, test_features, train_genre,test_genre = train_test_split(features, genre, test_size=0.33, stratify = genre)

sampler = RandomOverSampler()
train_features, train_genre = sampler.fit_sample(train_features,train_genre)

#Using RandomForesclassifier to train on extracted features and predict the genre of books
classifier = RandomForestClassifier()
grid_params = { 
    'n_estimators': [200, 700],
    'max_features': ['auto', 'sqrt', 'log2']
}


#Using grid search cv to check for the set of parameters

f1_scorer = make_scorer(fbeta_score, beta=1, average='micro')
grid = GridSearchCV(classifier, grid_params, cv=10, scoring=f1_scorer, return_train_score=True)
grid.fit(train_features, train_genre)

grid_results = pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']]

#Predict the genre of books in test using the best model suggested by GridSearchCV based on the given parameters.
predicted_genre = grid.predict(test_features)


labels=['Allegories', 'Christmas Stories', 'Detective and Mystery',
       'Ghost and Horror', 'Humorous and Wit and Satire', 'Literary',
       'Love and Romance', 'Sea and Adventure', 'Western Stories']
  
#Writing the confusion matrix for the predcitions made on test set to a csv file     
performance_report = ((pd.DataFrame(confusion_matrix(test_genre,predicted_genre,labels=labels),index=[format(label) for label in labels], 
    columns=[format(label) for label in labels])))
    
performance_report.to_csv('test_data_results.csv')
print(accuracy_score(y_test,grid.predict(X_test))*100)