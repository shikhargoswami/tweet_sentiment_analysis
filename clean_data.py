import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

STOPWORDS = set(stopwords.words('english'))

def remove_urls(text):
    url_remove = re.compile(r'https?://\S+|www\.\S+')
    return url_remove.sub(r'', text)


def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)


# Lower casing
def lower(text):
    low_text= text.lower()
    return low_text



# Number removal
def remove_num(text):
    remove= re.sub(r'\d+', '', text)
    return remove




", ".join(stopwords.words('english'))
STOPWORDS = set(stopwords.words('english'))

def punct_remove(text):
    punct = re.sub(r"[^\w\s\d]","", text)
    return punct




def remove_stopwords(text):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])



#Remove mentions and hashtags
def remove_mention(x):
    text=re.sub(r'@\w+','',x)
    return text

def remove_hash(x):
    text=re.sub(r'#\w+','',x)
    return text


def remove_space(text):
    space_remove = re.sub(r"\s+"," ",text).strip()
    return space_remove



def clean(text):

    text = remove_urls(text)
    text = remove_html(text)
    text = remove_hash(text)
    text = remove_num(text)
    text = remove_mention(text)
    text = remove_stopwords(text)
    text = remove_space(text)

    return text 