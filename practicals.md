Convert the text into tokens. Find the word frequency. 

# Import Libraries
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter


# Download NLTK Data
nltk.download('punkt')
nltk.download('punkt_tab')


# Input Text
text = """
Natural Language Processing is interesting.
Natural Language Processing is useful.
NLP is a part of Artificial Intelligence.
"""


# Convert Text into Lowercase
text = text.lower()

print(text)


# Sentence Tokenization
sentences = sent_tokenize(text)

print("Sentence Tokens:")
for i, sentence in enumerate(sentences, 1):
    print(i, ":", sentence)


# Word Tokenization
tokens = word_tokenize(text)

print("Word Tokens:")
print(tokens)


# Remove Punctuation
words = [word for word in tokens if word.isalpha()]

print("Words without punctuation:")
print(words)


# Calculate Word Frequency
word_frequency = Counter(words)

print("Word Frequency:")
for word, frequency in word_frequency.items():
    print(word, ":", frequency)


Practical 2: Find Synonyms and Antonyms of a Word Using WordNet

# Import Libraries
import nltk
from nltk.corpus import wordnet

# Download WordNet
nltk.download('wordnet')
nltk.download('omw-1.4')

# Accept a word from user
word = input("Enter a word: ").lower()

# Create empty sets
synonyms = set()
antonyms = set()

# Find synonyms and antonyms
for synset in wordnet.synsets(word):
    for lemma in synset.lemmas():

        synonyms.add(lemma.name())

        for antonym in lemma.antonyms():
            antonyms.add(antonym.name())

# Display Result
print("Word:", word)

print("\nSynonyms:")
for synonym in sorted(synonyms):
    print(synonym)

print("\nAntonyms:")
for antonym in sorted(antonyms):
    print(antonym)


Practical 3: Demonstrate a Bigram / Trigram Language Model and Generate Regular Expressions for a Given Text

# Import Libraries
import nltk
from nltk.tokenize import word_tokenize
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline

# Download required resources
nltk.download('punkt')
nltk.download('punkt_tab')

# Accept text from user
text = input("Enter the text: ")

# Tokenize the text
tokens = word_tokenize(text.lower())

# Select Bigram or Trigram
n = int(input("Enter 2 for Bigram or 3 for Trigram: "))

if n == 2 or n == 3:

    # Prepare training data
    train_data, vocabulary = padded_everygram_pipeline(n, [tokens])

    # Create and train model
    model = MLE(n)
    model.fit(train_data, vocabulary)

    print("\nTokens:")
    print(tokens)

    print("\nLanguage Model created successfully!")

    print("\nVocabulary:")
    print(list(model.vocab))

    # Calculate probability
    if n == 2 and len(tokens) >= 2:
        word1 = tokens[0]
        word2 = tokens[1]

        probability = model.score(word2, [word1])

        print("\nBigram:")
        print(word1, word2)

        print("Probability:")
        print(f"P({word2} | {word1}) =", probability)

    elif n == 3 and len(tokens) >= 3:
        word1 = tokens[0]
        word2 = tokens[1]
        word3 = tokens[2]

        probability = model.score(word3, [word1, word2])

        print("\nTrigram:")
        print(word1, word2, word3)

        print("Probability:")
        print(f"P({word3} | {word1} {word2}) =", probability)

else:
    print("Invalid choice. Enter 2 or 3.")

Practical 4: Perform Lemmatization and Stemming. Identify Parts of Speech Using Penn Treebank Tag Set. 

# Import Libraries
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download Required Data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

# Accept Sentence
text = input("Enter a sentence: ")

# Tokenization
tokens = word_tokenize(text)

print("\nTokens:")
print(tokens)

# Stemming
stemmer = PorterStemmer()

print("\nStemming:")
for word in tokens:
    if word.isalpha():
        print(word, "->", stemmer.stem(word))

# Lemmatization
lemmatizer = WordNetLemmatizer()

print("\nLemmatization:")
for word in tokens:
    if word.isalpha():
        print(word, "->", lemmatizer.lemmatize(word))

# POS Tagging
pos_tags = nltk.pos_tag(tokens)

print("\nPOS Tags:")
for word, tag in pos_tags:
    print(word, "->", tag)

# Penn Treebank POS Tag Meaning
tag_meanings = {
    "DT": "Determiner",
    "NNS": "Noun, plural",
    "VBP": "Verb, non-3rd person singular present",
    "VBG": "Verb, gerund/present participle",
    "RB": "Adverb",
    "IN": "Preposition/Subordinating conjunction",
    "JJ": "Adjective",
    "NN": "Noun, singular or mass"
}

print("\nPenn Treebank POS Tag Meaning:")
for word, tag in pos_tags:
    meaning = tag_meanings.get(tag, "Other/Punctuation")
    print(word, "->", tag, "->", meaning)

Practical 5: Implement HMM for POS Tagging and Build a Chunker

# Import Libraries
import nltk
from nltk import RegexpParser
from nltk.probability import ConditionalFreqDist
from nltk.tag import HiddenMarkovModelTrainer

# Training Data
data = [
    [("The", "DT"), ("dog", "NN"), ("barks", "VBZ")],
    [("A", "DT"), ("cat", "NN"), ("runs", "VBZ")],
    [("The", "DT"), ("boy", "NN"), ("plays", "VBZ")],
    [("She", "PRP"), ("reads", "VBZ"), ("books", "NNS")]
]

# Train HMM Model
trainer = HiddenMarkovModelTrainer()
model = trainer.train_supervised(data)

# Test Sentence
sentence = ["The", "dog", "runs"]

# POS Tagging
tagged_words = model.tag(sentence)

print("HMM POS Tagging:")
print(tagged_words)

# Chunker
grammar = r"""
NP: {<DT>?<JJ>*<NN.*>}
VP: {<VB.*><NP>?}
"""

chunker = RegexpParser(grammar)

# Build Chunk Tree
tree = chunker.parse(tagged_words)

print("\nChunker Output:")
print(tree)


Practical 6: Implement Named Entity Recognition (NER)

# Import Library
import spacy

# Load NLP Model
nlp = spacy.load("en_core_web_sm")

# Input Text
text = input("Enter a sentence: ")

# Perform Named Entity Recognition
doc = nlp(text)

# Display Entities
print("\nNamed Entities:")

for ent in doc.ents:
    print(ent.text, ":", ent.label_)

Practical 7: Implement Semantic Role Labelling (SRL) to Identify Named Entities

# Import Library
import spacy

# Load NLP Model
nlp = spacy.load("en_core_web_sm")

# Input Sentence
text = input("Enter a sentence: ")

# Process Text
doc = nlp(text)

# Identify Named Entities
print("\nNamed Entities:")
for ent in doc.ents:
    print(ent.text, ":", ent.label_)

# Semantic Role Labelling
print("\nSemantic Roles:")

for token in doc:
    if token.pos_ == "VERB":
        print("\nVerb:", token.text)

        for child in token.children:

            if child.dep_ == "nsubj":
                print("Agent:", child.text)

            elif child.dep_ in ("dobj", "obj"):
                print("Patient/Theme:", child.text)

            elif child.dep_ == "iobj":
                print("Recipient:", child.text)

            elif child.dep_ == "prep":
                for obj in child.children:
                    if obj.dep_ == "pobj":
                        print(child.text, ":", obj.text)

Practical 8: Implement Text Classifier Using Logistic Regression Model

# Import Libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample Dataset
texts = [
    "computer software and programming",
    "artificial intelligence and technology",
    "computer hardware and coding",
    "software development and algorithms",

    "football and cricket match",
    "basketball sports tournament",
    "team won the championship",
    "players practice for the game",

    "doctor treats the patient",
    "hospital provides medical treatment",
    "healthcare and medicine",
    "nurse provides patient care"
]

labels = [
    "Technology", "Technology", "Technology", "Technology",
    "Sports", "Sports", "Sports", "Sports",
    "Healthcare", "Healthcare", "Healthcare", "Healthcare"
]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.25, random_state=42
)

# Convert Text into Numerical Data
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict Results
y_pred = model.predict(X_test)

# Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Actual Labels:", y_test)
print("Predicted Labels:", y_pred)
print("Accuracy:", accuracy * 100, "%")

# Test New Text
new_text = ["The doctor treats patients in the hospital"]

new_data = vectorizer.transform(new_text)
prediction = model.predict(new_data)

print("\nNew Text:", new_text[0])
print("Predicted Category:", prediction[0])

Practical 9: Implement a Movie Reviews Sentiment Classifier


import nltk
from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Download dataset
nltk.download('movie_reviews')

# Load movie reviews
texts = []
labels = []

for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        texts.append(movie_reviews.raw(fileid))
        labels.append(category)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.25,
    random_state=42, stratify=labels
)

# Convert text into numerical features
vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2),
    max_features=3500
)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression(max_iter=400)
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred) * 100, "%")

# Predict new movie review
review = input("\nEnter a movie review: ")
review_vector = vectorizer.transform([review])
prediction = model.predict(review_vector)

print("Sentiment:", prediction[0].capitalize())

Practical 10: Implement RNN for Sequence Labelling


import torch
import torch.nn as nn
import torch.optim as optim

# Training dataset
data = [
    [("Alan", "B-PER"), ("Turing", "I-PER"), ("lived", "O"),
     ("in", "O"), ("London", "B-LOC")],

    [("Google", "B-ORG"), ("was", "O"), ("founded", "O"),
     ("in", "O"), ("California", "B-LOC")],

    [("Ada", "B-PER"), ("Lovelace", "I-PER"), ("worked", "O"),
     ("with", "O"), ("Charles", "B-PER"), ("Babbage", "I-PER")],

    [("Microsoft", "B-ORG"), ("operates", "O"),
     ("in", "O"), ("Seattle", "B-LOC")]
]

# Create vocabulary
words = list(set(w.lower() for sentence in data for w, tag in sentence))
word2idx = {word: i + 1 for i, word in enumerate(words)}
word2idx["<PAD>"] = 0

tags = ["<PAD>", "O", "B-PER", "I-PER", "B-LOC", "B-ORG"]
tag2idx = {tag: i for i, tag in enumerate(tags)}
idx2tag = {i: tag for tag, i in tag2idx.items()}

# Prepare input and output
max_len = max(len(s) for s in data)

X, Y = [], []

for sentence in data:
    x = [word2idx[w.lower()] for w, t in sentence]
    y = [tag2idx[t] for w, t in sentence]

    x += [0] * (max_len - len(x))
    y += [0] * (max_len - len(y))

    X.append(x)
    Y.append(y)

X = torch.tensor(X)
Y = torch.tensor(Y)

# Define RNN model
class RNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(len(word2idx), 32, padding_idx=0)
        self.rnn = nn.RNN(32, 48, batch_first=True)
        self.fc = nn.Linear(48, len(tags))

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        return self.fc(x)

# Create model
model = RNNModel()

# Train model
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=0.02)

for epoch in range(35):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output.reshape(-1, len(tags)), Y.reshape(-1))
    loss.backward()
    optimizer.step()

print("Model Training Completed!")
print("Final Loss:", loss.item())

# Test the model
sentence = ["Alan", "Turing", "lived", "in", "London"]
input_data = torch.tensor([
    [word2idx[w.lower()] for w in sentence]
])

with torch.no_grad():
    output = model(input_data)
    predictions = torch.argmax(output, dim=2)[0]

print("\nSequence Labelling:")
for word, tag in zip(sentence, predictions):
    print(word, ":", idx2tag[tag.item()])


