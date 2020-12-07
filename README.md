# Genre Identification on Project Gutenberg corpus
## Introduction

Classification of literary genre of books can hugely benefited by the usage of semantic features rather than
simple bag of words representation. Implementation was carried under the  assumption that
model trained on semantic features performs better than a model trained on bag of words as features.

## Toolkit
Language   : Python

Libraries  : BeatuifulSoup, Spacy, Pandas and Sklearn

## Workflow

### Pre-processing
The content of the books was procvided in form of html files. To parse the html files and extract the textual content 
BeautifulSoup library was used. Spacy nlp pipeline was used later to perform the basic pre-processing steps on 
the textual content of the book.

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

A baseline model to classify the genre of a book was implemented using simple bag of words approach (TF-IDF). 
We used baseline model as a way to check whether a model using handcrafted features to classify genre of book 
performed better when compared with TF-IDF approach.


