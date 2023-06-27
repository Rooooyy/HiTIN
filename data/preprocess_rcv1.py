import pandas as pd
import re
import xml.etree.ElementTree as ET
import json

english_stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll",
                     "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's",
                     'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs',
                     'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am',
                     'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                     'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
                     'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during',
                     'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
                     'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
                     'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
                     'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't",
                     'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't",
                     'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
                     'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn',
                     "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won',
                     "won't", 'wouldn', "wouldn't"]

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = string.strip().strip('"')
    string = re.sub(r"[^A-Za-z0-9(),!?\.\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " ", string)
    string = re.sub(r"\.", " ", string)
    string = re.sub(r"\"", " ", string)
    string = re.sub(r"!", " ", string)
    string = re.sub(r"\(", " ", string)
    string = re.sub(r"\)", " ", string)
    string = re.sub(r"\?", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()                    

def read_rcv1_ids(filepath):
    ids = set()
    with open(filepath) as f:
        new_doc = True
        for line in f:
            line_split = line.strip().split()
            if new_doc and len(line_split) == 2:
                tmp, did = line_split
                did = int(did)
                if tmp == '.I':
                    ids.add(did)
                    new_doc = False
                else:
                    print(line_split)
                    print('maybe error')
            elif len(line_split) == 0:
                new_doc = True
    print('{} samples in {}'.format(len(ids), filepath))
    return ids

def parse_xml(xml):
    content = ''
    root = ET.XML(xml)
    for p in root.find('text').findall('p'):
        content += p.text
    return content

def df_to_dict(df):
    data = []
    for _, row in df.iterrows():
        doc = clean_str(row['text'])
        token = [word.lower() for word in doc.split() if word not in english_stopwords and len(word) > 1]

        label = row['topics'].lstrip('[').rstrip(']')
        label = [l.strip().strip("'") for l in label.split(',')]
        data.append(json.dumps({'token': token, 'label': label, 'topic': [], 'keyword': []}) + '\n')
    return data


if __name__ == "__main__":
    df = pd.read_csv('rcv1_v2.csv')

    train_id_set = read_rcv1_ids('./lyrl2004_tokens_train.dat')

    mask = df['id'].isin(train_id_set)
    train_df = df.loc[mask]
    test_df = df.loc[~mask]
    val_df =  train_df.sample(frac=0.1, random_state=7, axis=0)
    train_df = train_df.loc[~train_df['id'].isin(val_df['id'])]

    train_df.loc[:,['text']] = train_df['text'].apply(parse_xml)
    val_df.loc[:, ['text']] = val_df['text'].apply(parse_xml)
    test_df.loc[:, ['text']] = test_df['text'].apply(parse_xml)

    train_dict = df_to_dict(train_df)
    val_dict = df_to_dict(val_df)
    test_dict = df_to_dict(test_df)

    with open('rcv1_train.json', 'w') as f:
        f.writelines(train_dict)
    with open('rcv1_val.json', 'w') as f:
        f.writelines(val_dict)
    with open('rcv1_test.json', 'w') as f:
        f.writelines(test_dict)

