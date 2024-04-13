from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import pickle

# nltk.download('stopwords')

ps = PorterStemmer()

with open('.\\rfmodel\\vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('.\\rfmodel\classifier.pkl', 'rb') as f:
    model = pickle.load(f)


def preprocessing(content):
    preprocessed_content = re.sub('[^a-zA-Z]'," ", content)
    preprocessed_content = preprocessed_content.lower()
    preprocessed_content = preprocessed_content.split()
    # preprocessed_content = [ps.stem(word) for word in preprocessed_content if word not in stopwords.words('english')]
    preprocessed_content = " ".join(preprocessed_content)
    return preprocessed_content

def prediction(title:str):
    content = title 
    preprocessed_content = [preprocessing(content)]
    vectorized_content = vectorizer.transform(preprocessed_content)
    return model.predict(vectorized_content[0])
