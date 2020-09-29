from PreprocessPipeline import Pipeline
from ExtractFeature import WritingStyle
import pandas as pd
from csv import writer
import os
from os import path




def execute_preprocess_pipeline(input_path):
    """This method uses spacy for preprocessing"""
    pipeline = Pipeline()
    processed_tokens,sentences,doc_dict = pipeline.preprocess_text(html_path=input_path)
    return processed_tokens,sentences,doc_dict


def fetch_ws_features(processed_tokens,sentences,doc_dict):
 """This method fetches writing style features"""
    ws_obj = WritingStyle()
    writing_style_dict = ws_obj.extract_writing_style(processed_tokens,sentences,doc_dict)
    return  writing_style_dict


def fetch_sentiment_dict(sentences_dict):
 """This method gets the count of sentences belonging to certain sentiment"""
    ws_obj = WritingStyle();
    sentiment_dict = ws_obj.calculate_sentiment(sentences_dict)
    return sentiment_dict

def write_to_csv(feature_dict,senti_dict,file_path):
"""Write the hand crafted features of books to a csv file with corresponding genre"""

    csv_file = file_path + "\\ext_features.csv"
    
    header = ['book_id','male_pn_count','female_pn_count','locative_pn_count','comma_count',
              'period_count','colon_count','semi_colon_count','hyphen_count',
              'exclam_count','conjunction_count','interjection_count','pos_sent','neg_sent','neut_sent',
              'plot_complexity', 'genre']

    master_frame = pd.read_csv(file_path+'\\master996.csv", sep=";", encoding='unicode_escape')

    if path.exists(csv_file):
         os.remove(csv_file)
    try:
        with open(csv_file, 'w+') as write_obj:
            # Create a writer object from csv module
            csv_writer = writer(write_obj, lineterminator='\r')
            csv_writer.writerow(header)
            # Add contents of list as last row in the csv file

            for (book_id, list_val), (b_id, sent_list) in zip(feature_dict.items(), senti_dict.items()):
                # rows = []
                temp = [book_id]
                temp.extend(list_val)
                temp.extend(sent_list)
                try:
                    temp.append(master_frame.loc[master_frame['book_id'] == book_id]['guten_genre'].values[0])
                except:
                    temp.append("NONE")
                # rows.append(temp)
                csv_writer.writerow(temp)

    except IOError:
        print("I/O error in writing the feature csv file")


def main():
    file_path = "D:\\Edu\\dump\\"

    #Perform preprocessing tasks on the content of the books 
    processed_tokens, sentences,doc_dict = execute_preprocess_pipeline(file_path)
    
    #Extract features that are pertainging to writing style
    feature_dict = fetch_ws_features(processed_tokens,sentences,doc_dict)
    
    #Extract features related to  sentiment of sentences
    sentiment_dict = fetch_sentiment_dict(sentences)
    
    #Writing the extracted features of each book to a csv file 
    write_to_csv(feature_dict, sentiment_dict,file_path)
   


if __name__ == "__main__":
    main()
