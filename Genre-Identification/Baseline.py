
#Creating a baseline model using TF-IDF, to use it during evaluation of model trained on handcrafted features 
import pandas as pd 
from os import listdir
from os.path import join
import codecs
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('punkt')
nltk.download('stopwords') 
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
stoplist = set(stopwords.words('english')) 


# Helper functions to perform basic pre-processing tasks
def tokenize_word_text(text): 
    tokens = nltk.word_tokenize(text) 
    tokens = [token.strip() for token in tokens if token.isalpha()] 
    return tokens

# Remove stop words from the book content
def remove_stop_words(tokens_list):
  tokens_without_sw = [word for word in tokens_list if not word in stoplist]
  return tokens_without_sw

# Applying lemmatization on book content
def get_lemmatized_words(tokens_list):
  lemmatizer = WordNetLemmatizer()
  lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens_list]
  return lemmatized_tokens


directory = "D:\\Edu\\dump"
#Get the meta-data about the books present in the master csv file
master_frame = pd.read_csv(directory + '\\master996.csv', sep=";", encoding='unicode_escape')

#Fetch the html files of books in the directory
path=directory+'\\dump\\' 

htmlFiles = [join(path, f) for f in listdir(path) ]

textList = []
genereList = []

for index in range (len(htmlFiles)):
  bookId = htmlFiles[index].strip().split("/")[-1][:-13]
  try:
    bookGenre = master_frame.loc[master_frame['book_id'] == bookId]['guten_genre'].values[0] 
  except:
    bookGenre = "None"
  genereList.append(bookGenre)
  file = codecs.open(htmlFiles[index], "r", "utf-8")
  soup = BeautifulSoup(file.read(), 'html.parser')
  text = ''.join([item for item in soup.find_all(text=True) if item.parent.name == 'p'])
  text = tokenize_word_text(text)
  text = remove_stop_words(text)
  text = get_lemmatized_words(text)
  text = " ".join(text)
  textList.append(text)

df = pd.DataFrame(textList) 
df = df[df.columns[0]]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df)

newdf = pd.DataFrame(X.toarray(), columns = vectorizer.get_feature_names()) 
newdf['Genere'] = genereList 
newdf.to_csv(directory+"\\tfIdf_csvFile.csv")
