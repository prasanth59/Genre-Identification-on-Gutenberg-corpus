from collections import Counter
import numpy as np
from textblob import TextBlob
import random
from fuzzywuzzy import fuzz
from enum import Enum

class Constants(Enum):
    PREPOSITION_LIST = ["in", "at", "on", "beside", "near", "towards", "above", "under", "below"]
    COMMA = ","        
    PERIOD = "."
    COLON = ":"
    SEMICOLON = ";"
    HYPHEN = "-"
    EXCLAM = "!"
    CONJ = "CCONJ"
    INTJ = "INTJ"
    MALE_PR_1 = "he"
    MALE_PR_2 = "him"
    FEMALE_PR_1 = "she"
    FEMALE_PR_2 = "her"

class WritingStyle:

 
    def __int__(self):

        self.male_pn_count = 0
        self.female_pn_count = 0
        self.locative_pn_count = 0
        self.interjection_count = 0
        self.conjunction_count = 0
        self.comma_count = 0
        self.period_count = 0
        self.colon_count = 0
        self.hyphen_count = 0
        self.exclam_count = 0
        self.semi_colon_count = 0
        self.plotComplexity = 0


    def summarize_text(self,proper_nouns, top_num):
        counts = dict(Counter(proper_nouns).most_common(top_num))
        charList = []
        sorted_charDict = sorted(counts, key=counts.get, reverse=True)
        for item in sorted_charDict:
            charList.append(item)
        return  charList

    def get_Perfect_Charachter_counts(self,charList):
        totalMatchings = []
        for name in charList:
            for next_name in charList:
                if (name != next_name) and (fuzz.token_set_ratio(name, next_name) == 100):
                    totalMatchings.append([name, next_name])
        for char_names in totalMatchings:
            name_1 = char_names[0]
            name_2 = char_names[1]
            if len(name2) > len(name1):
                charList = [name2 if name == name1 else name for name in charList]
            else:
                charList = [name1 if name == name2 else name for name in charList]
        uniqueList = np.unique(charList)
        return uniqueList


    # Mehod to reset variables after feature extraction for each book
    def reset_variables(self):
        self.male_pn_count = 0
        self.female_pn_count = 0
        self.locative_pn_count = 0
        self.interjection_count = 0
        self.conjunction_count = 0
        self.comma_count = 0
        self.period_count = 0
        self.colon_count = 0
        self.hyphen_count = 0
        self.exclam_count = 0
        self.semi_colon_count = 0
        self.plotComplexity = 0
        
    #Normalizing the values of extracted features 
    def normalize_by_size(self ,feature_list,size ):
        if(size==0):
            size=1

        feature_list=[number / size for number in feature_list]
        feature_list = ['%.3f' % elem for elem in feature_list]
        return feature_list


    #Extract the features related to writing style of book
    def extract_writing_style(self,book_tokens,sentences,doc_dict):
        feature_dict = {}

        preposition_list = ["in", "at", "on", "beside", "near", "towards", "above", "under", "below"]
        for book_id,tokens in book_tokens.items():

            text_list = [token.text.lower() for token in tokens]

            pos_list = [token.pos_ for token in tokens]

            self.comma_count = text_list.count(Constants.COMMA.value)
            self.period_count = text_list.count(Constants.PERIOD.value)
            self.colon_count = text_list.count(Constants.COLON.value)
            self.semi_colon_count = text_list.count(Constants.SEMICOLON.value)
            self.hyphen_count = text_list.count(Constants.HYPHEN.value)
            self.exclam_count = text_list.count(Constants.EXCLAM.value)
            self.conjunction_count = pos_list.count(Constants.CONJ.value)
            self.interjection_count = pos_list.count(Constants.INTJ.value)
            self.locative_pn_count = sum([text_list.count(item) for item in Constants.PREPOSITION_LIST.value])
            self.male_pn_count = text_list.count(Constants.MALE_PR_1.value) + text_list.count(Constants.MALE_PR_2.value)
            self.female_pn_count = text_list.count(Constants.FEMALE_PR_1.value) + text_list.count(Constants.FEMALE_PR_2.value)


            personList = [str(ee) for ee in doc_dict[book_id].ents if ee.label_ == 'PERSON']
            charList = self.summarize_text(personList,200  )
            finalCharactertList = self.get_Perfect_Charachter_counts(charList)
            self.plotComplexity = len(finalCharactertList)
          
            feature_dict[book_id] = [self.male_pn_count,self.female_pn_count,self.locative_pn_count,self.comma_count,
                                     self.period_count,self.colon_count,self.semi_colon_count,self.hyphen_count,
                                     self.exclam_count,self.conjunction_count,self.interjection_count,self.plotComplexity]
            feature_dict[book_id]= self.normalize_by_size(feature_dict[book_id],len(text_list))
            self.reset_variables()

        return feature_dict


    # Method to fetch random sentences from books
    def get_sentence_list(self,sentences):
        percent = 0.1
        num_sentences = int(percent * len(sentences))
        random_list = random.sample(range(0, len(sentences)), num_sentences)
        random_senteces = [sentences[num] for num in random_list]
        return  random_senteces

    # Method to calculate sentiment for random sentences
    def get_senti_count(self,random_sentences):
        pos_count = 0;
        neg_count = 0;
        neu_count = 0;

        for item in random_sentences:
            blob_obj = TextBlob(item)
            if blob_obj.sentiment.polarity > 0:
                pos_count += 1
            elif blob_obj.sentiment.polarity < 0:
                neg_count += 1
            else:
                neu_count += 1
        return pos_count,neg_count,neu_count

    # Method to retrieve count of sentences belonging to certain sentiment 
    def calculate_sentiment(self,sentences_dict):
        senti_dict = {}
        for book_id , sentences in sentences_dict.items():
            random_sentences = self.get_sentence_list(sentences)

            pos_count,neg_count,neu_count = self.get_senti_count(random_sentences)
            senti_dict[book_id] = [pos_count,neg_count,neu_count]
            senti_dict[book_id]=self.normalize_by_size(senti_dict[book_id],len(random_sentences))
        return senti_dict

