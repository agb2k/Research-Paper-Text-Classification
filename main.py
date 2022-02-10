import os
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import pandas as pd

# Setting path for files
path = "Q3/"
fileList = os.listdir(path)

word_list = []

# Checks each text file in the given folder path, tokenizes each word and adds words to a list
for x in fileList:
    if x.endswith(".txt"):
        with open(path + x, "r") as f:
            for line in f:
                words = word_tokenize(line)
                if not words[0] == '#':
                    word_list.append([words[0], " ".join(words[1:-1])])

f.close()

# Converts list to dataframe
df = pd.DataFrame(word_list, columns=['Label', 'Description'])

# Removes english stop words and prepares for naive bayes model by creating a tally of tokens
vectorizer = CountVectorizer(stop_words='english')

# Fits and transforms to learn the vocabulary set up by the vectorizer and return a document-term matrix
all_features = vectorizer.fit_transform(df.Description)

# Splits data into training and testing data sets
x_train, x_test, y_train, y_test = train_test_split(all_features, df.Label, test_size=0.3)

# Sets up classifier and prints score
classifier = MultinomialNB()
classifier.fit(x_train, y_train)
score = classifier.score(x_test, y_test)
print(score)
