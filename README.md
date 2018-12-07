# CS 410 Final Project (Hotel Reviews Summarizer) 
- UIUC MCS-DS (Fall 2018)
- Tyler Watkins

## An overview of the function of the code (i.e., what it does and what it can be used for). 

The code is used to help potential hotel guests get a better overall view of a hotel's reviews.

The potential guest inputs a selected Hotel name and several useful summaries of the past reviews are displayed including:

- number of ratings, mean rating
- common words used by reviewers
- reviewer sentiment
- the most important topics discussed by reviewers


## Documentation of how the software is implemented with sufficient detail so that others can have a basic understanding of your code for future extension or any further improvement. 

The code is implemented in a jupyter notebook as follows: 

1. It takes as input the name of a user selected hotel from the hotels database found in cleaned_reviews.csv.
2. The hotel name is passed to the function full_hotel_review which extracts all numerical and written reviews about the hotel and passes that information to 4 additional functions held within the function full_hotel_review. 

The 4 additional functions are:
- rating_reviews - a function that returns the total number of reviews and mean review rating out of 5.
- generate_wordcloud - a function that returns a wordcloud showing the most frequently used words used by reviewers.
- get_sentiment - a function that returns a graphical overview of the negative, neutral and positive sentiment of reviewers.
- get_topics - a function that returns a list of the 10 most important words in each of the 3 main topics reviewers focused on.

The output is displayed on the screen in a jupyter notebook.


## Documentation of the usage of the software including either documentation of usages of APIs or detailed instructions on how to install and run a software, whichever is applicable. 

The following packages were used in this project:
- sklearn
- numpy
- pandas
- matplotlib
- seaborn
- wordcloud
- gensim
- re
- pyLDAvis
- nltk

Including:
- nltk.download('stopwords')
- nltk.download('punkt')
- nltk.download('averaged_perceptron_tagger')
- nltk.download('wordnet')
- nltk.download('vader_lexicon')
