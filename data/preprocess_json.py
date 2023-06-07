import json


def modify_keyname(file_path):
    corpus = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            data = json.loads(line)
            data["label"] = data.pop("doc_label")
            data["token"] = data.pop("doc_token")
            data["topic"] = data.pop("doc_topic")
            data["keyword"] = data.pop("doc_keyword")
            corpus.append(data)

    with open(file_path, 'w', encoding='utf-8') as f:
        for data in corpus:
            line = json.dumps(data)
            f.write(line + '\n')


if __name__ == "__main__":
    modify_keyname('rcv1_train.json')
    modify_keyname('rcv1_val.json')
    modify_keyname('rcv1_test.json')



