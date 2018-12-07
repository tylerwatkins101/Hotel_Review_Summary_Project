import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('vader_lexicon')

import string
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import gensim
from gensim import corpora
import re
import pyLDAvis.gensim


def rating_reviews(hotel):
    n_reviews = hotel.reviewsrating.count()
    avg_rating = hotel.reviewsrating.mean()
    msg = "\nThis hotel has been reviewed %i times with an average rating of %.2f out of 5. \n" % (n_reviews,avg_rating)
    print(msg)


def generate_wordcloud(hotel):

    # Add all text from selected hotel
    text = " ".join(review for review in hotel.cleanrev)

    from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

    # Create stopword list:
    stopwords = set(STOPWORDS)
    stopwords.update(["hotel", "room", "stay", "place", "one", "de", "que", "la", "en","el", "con"])

    # Generate a word cloud image
    wordcloud = WordCloud(stopwords=stopwords, background_color="white", contour_width=3, contour_color='firebrick').generate(text)

    # Display the generated image:
    print("These are the words that hotel guests frequently use when reviewing this hotel:")
    plt.figure(figsize=[12,6])
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


def get_sentiment(hotel):

    # Make review text into a single doc
    doc_complete = list(hotel.cleanrev)

    # Using NLTK built in function (could train a Naive Bayes Classifier)
    import nltk
    def nltk_sentiment(sentence):
        from nltk.sentiment.vader import SentimentIntensityAnalyzer

        nltk_sentiment = SentimentIntensityAnalyzer()
        score = nltk_sentiment.polarity_scores(sentence)
        return score

    nltk_results = [nltk_sentiment(row) for row in doc_complete]
    results_df = pd.DataFrame(nltk_results)
    text_df = pd.DataFrame(doc_complete, columns = ['text'])
    nltk_df = text_df.join(results_df)

    # Melt the dataframe to get boxplots
    df = nltk_df.melt(id_vars=['text', 'compound'], var_name='Sentiments', value_name='Percentages')

    # Make plot
    f, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=df, x = 'Sentiments', y='Percentages')
    plt.title('Sentiment Analysis of Reviews', fontsize=15)
    plt.ylabel('Percentages', fontsize=15)
    plt.xlabel(' ')
    plt.xticks(np.arange(3), labels = ['Negative', 'Neutral', 'Positive'], fontsize=15)

    print("\nThis is the analyzed sentiment of guests reviewing this hotel:")

    # Show the figure
    plt.show()


def get_topics(hotel):

    # Make review text into a single doc
    doc_complete = list(hotel.cleanrev)

    # Clean by removing the punctuations, stopwords and normalize the corpus.
    from nltk.corpus import stopwords
    from nltk.stem.wordnet import WordNetLemmatizer
    import string

    stop = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()
    def clean(doc):
        stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
        punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
        normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
        return normalized

    doc_clean = [clean(doc).split() for doc in doc_complete]

    import gensim
    from gensim import corpora

    # Create the term dictionary of our corpus, where every unique term is assigned an index.
    dictionary = corpora.Dictionary(doc_clean)

    # Convert list of documents (corpus) into Document Term Matrix using dictionary prepared above.
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

    # Create the object for LDA model using gensim library
    Lda = gensim.models.ldamodel.LdaModel

    # Run and Train LDA model on the document term matrix.
    ldamodel = Lda(doc_term_matrix, num_topics=3, id2word = dictionary, passes=50)

    # Extract the 3 main topics (10 words each) from the LDA model and display them
    x = ldamodel.show_topics(num_topics=3, num_words=10)
    import re
    topic1 = " ".join(re.findall("[a-zA-Z]+", x[0][1]))
    topic2 = " ".join(re.findall("[a-zA-Z]+", x[1][1]))
    topic3 = " ".join(re.findall("[a-zA-Z]+", x[2][1]))

    print("The words associated with the 3 most important topics of reviews for this hotel are: \n")
    print("Topic 1 top words: %s"%(topic1))
    print("Topic 2 top words: %s"%(topic2))
    print("Topic 3 top words: %s"%(topic3))
    print("\n")


    # Display LDA viz with pyLDAvis library
    import pyLDAvis.gensim
    lda_display = pyLDAvis.gensim.prepare(ldamodel, doc_term_matrix, dictionary, sort_topics=False)
    pyLDAvis.enable_notebook(lda_display)
