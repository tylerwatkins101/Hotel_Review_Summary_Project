# CS 410 Final Project (Hotel Rating Summary Tool) 
- Tyler Watkins
- Fall 2018
- University of Illinois at Urbana-Champaign (MCS-DS)

## An overview of the function of the code. 

The code is used to help potential hotel guests get a better overall view of a hotel's reviews.

The potential guest inputs a selected hotel name and several useful summaries of the past reviews are displayed including:

- total number of reviews and average review rating
- common words used by reviewers
- a summary of reviewer sentiment (broken down proportionally into: positive, neutral, and negative)
- the 3 most important topics discussed by reviewers


## Documentation of how the software is implemented

The code is implemented as follows.

There are 3 files.

1. cleaned_reviews.csv 
  - A csv file containing over 25,000 hotel reviews. Originally downloaded from https://www.kaggle.com/datafiniti/hotel-reviews which I cleaned in a separate jupyter notebook for preparation to use with this project. 

2. Hotel Rating Summary Tool - User Interface.ipynb
  - A jupyter notebook which gives the user the opportunity to choose a hotel name selected from cleaned_reviews.csv, and see the output of the hotel rating summary tool. 
  - The hotel name is entered in the jupyter notebook.
  - The hotel name is passed to the function full_hotel_review which extracts all numerical and written reviews about the hotel and passes that information to 4 additional functions stored in the file hotel_review_summarizer.py. 

3. hotel_review_summarizer.py
  - A python script containing the main 4 functions used by the hotel rating summary tool.
  - The 4 functions are:
    - rating_reviews - a function that returns the total number of reviews and mean review rating out of 5.
    - generate_wordcloud - a function that returns a wordcloud showing the most frequently used words used by reviewers.
    - get_sentiment - a function that returns a graphical overview of the negative, neutral and positive sentiment of reviewers.
    - get_topics - a function that returns a list of the 10 most important words in each of the 3 main topics reviewers focused on.
  
After the code has run. The output of the hotel rating summary tool is displayed on the screen in the jupyter notebook.

## Documentation of the usage of the software

The project was coded with python 3.7. If you don't have python you can install it here:
https://www.python.org/downloads/

The following packages are used for this project and may need to be installed prior to running the tool. The recommended method of installation is to use pip install from the command line.
 
 Packages:
  - jupyter notebook
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - wordcloud
  - gensim
  - re
  - string
  - nltk

Including:
  - nltk.download('stopwords')
  - nltk.download('punkt')
  - nltk.download('averaged_perceptron_tagger')
  - nltk.download('wordnet')
  - nltk.download('vader_lexicon')

After installing the above. To use the tool:

1. Download these 3 files from https://github.com/tylerwatkins101/Hotel_Review_Summary_Project and store them in a folder.
- cleaned_reviews.csv
- hotel_review_summarizer.py
- Hotel Rating Summary Tool - User Interface.ipynb

2. From the folder you've stored the 3 files open a command line and type in 'jupyter notebook'.

3. Open 'Hotel Rating Summary Tool - User Interface.ipynb' in your browser.

4. Follow the instructions within the notebook.
- Hit enter on the first cell to load the libraries needed.
- Select a hotel name
- Hit enter on the second cell to run the program
