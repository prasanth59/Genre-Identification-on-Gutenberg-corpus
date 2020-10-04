# Genre Identification on Project Gutenberg corpus
## Introduction

Classification of literary genre of books can hugely benefited by the usage of semantic features rather than
simple bag of words representation. Implementation was carried under the  assumption that
model trained on semantic features performs better than a model trained on bag of words as features.

## Workflow

### Pre-processing

To better classify the book, it can be divided into multiple sub texts of equal length. Corpus will be
divided only  into train and test sets as we have less data. Train set was used to train
models along with their hyper parameter tuning. Finally, test set to predict/identify the genre of a unknown instances. 

### Feature Extraction

Features like Sentiment, Plot, Setting and writing style that can help in distinguishing literary genre
of a book will be identified, under the assumption that every author has an individual style of writing
and usually writes books belonging to only one genre.

### Feature Selection

Feature selection approaches will be used to check which features contribute more for the classification
task. Depending on the model performance, addition or discarding of features was  done.

### Classification

Random Forest classifier was used to identify the genre of a book using the extracted features.

### Evaluation

Performance of the model was compared against the baseline model. Baseline model was built
on simple bag of words representation.

## Technologies and libraries
Language used : Python
Libraries     : Spacy, Pandas, and Sklearn
