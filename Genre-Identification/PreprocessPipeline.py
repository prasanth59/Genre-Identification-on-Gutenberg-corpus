import spacy
import os
from bs4 import BeautifulSoup


class Pipeline:
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 2000000
    
    def fetch_file_list(self, html_path: str):
        """Returns list of html files in given directory"""
        files_path_list = [html_path + file for file in os.listdir(html_path) if file.endswith('-content.html')]
        return files_path_list

        # Fetch the book content from html file

    def get_file_content(self, files_path_list):
        """Returns book content in <p> tags in html file """
        file_text = {}
        for path in files_path_list:
            text = ""
            book_id = path.strip().split("\\")[-1][:-13]
            try:
                with open(path, "r", encoding="utf8") as f:
                    contents = f.read()
                    soup = BeautifulSoup(contents, 'html5lib')
                    for tag in soup.find_all("p"):
                        text = text + tag.text.replace("\\", "").replace("Â", "").strip("\n")
            except IOError:
                print("Error: Input HTML files not found ")
            file_text[book_id] = text
        return file_text

    def execute_spacy_pipeline(self, book_text):
        """Returns list of tokens and sentences after nlp spacy pipeline"""
        tagged_tokens = {}
        sentences_dict = {}
        doc_dict = {}

        for book_id, text in book_text.items():
            # Create an nlp object
            doc = self.nlp(text)
            
            tagged_tokens[book_id] = doc
            
            sentence_list = [str(item).strip() for item in list(doc.sents)]
            
            sentences_dict[book_id] = sentence_list
            
            doc_dict[book_id] = doc

        return tagged_tokens,sentences_dict,doc_dict

    
    def preprocess_text(self, html_path):
    #returns list of tokens ,list of sentences
        file_list = self.fetch_file_list(html_path)
        book_text = self.get_file_content(file_list)
        processed_tokens,sentences,doc_dict = self.execute_spacy_pipeline(book_text)
        return processed_tokens,sentences,doc_dict

