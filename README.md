# Hotel_Review_Summary_Project
## CS 410 Final Project (Hotel Reviews Summarizer) 

- Tyler Watkins

### An overview of the function of the code (i.e., what it does and what it can be used for). 

The code is used to help potential hotel guests get a better overall view of a hotel's reviews.

The potential guest inputs a selected Hotel name and several useful summaries of the past reviews are displayed including:

a. number of ratings, mean rating
b. common words used by reviewers
c. reviewer sentiment
d. the most important topics discussed by reviewers


### Documentation of how the software is implemented with sufficient detail so that others can have a basic understanding of your code for future extension or any further improvement. 

The code is implemented as follows: 

1. It takes as input the name of a user selected hotel from the hotels database found in cleaned_reviews.csv.
2. The hotel name is passed to the function full_hotel_review which extracts all numerical and written reviews about the hotel and passes that information to 4 additional functions held within the function full_hotel_review. 

The 4 additional functions are:
a. rating_reviews - a function that returns the total number of reviews and mean review rating out of 5.
b. generate_wordcloud - a function that returns a wordcloud showing the most frequently used words used by reviewers.
c. get_sentiment - a function that returns a graphical overview of the negative, neutral and positive sentiment of reviewers.
d. get_topics - a function that returns a..***need to finish what it returns***

### Documentation of the usage of the software including either documentation of usages of APIs or detailed instructions on how to install and run a software, whichever is applicable. 


This project was done by myself. There were no other group members.
