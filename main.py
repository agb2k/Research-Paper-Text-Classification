import os
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import pandas as pd

path = "Q3/"
fileList = os.listdir(path)

word_list = []

for x in fileList:
    if x.endswith(".txt"):
        with open(path + x, "r") as f:
            for line in f:
                words = word_tokenize(line)

                if not words[0] == '#':
                    word_list.append([words[0], " ".join(words[1:-1])])

f.close()

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
df = pd.DataFrame(word_list, columns=['Label', 'Description'])
print(df)

# Removes english stop words
vectorizer = CountVectorizer(stop_words='english')
all_features = vectorizer.fit_transform(df.Description)

print(vectorizer.vocabulary_)

x_train, x_test, y_train, y_test = train_test_split(all_features, df.Label, test_size=0.2, random_state=88)

classifier = MultinomialNB()
classifier.fit(x_train, y_train)
print(classifier.score(x_test, y_test))
