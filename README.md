Run the app.py for starting the server
data folder contains the entire database used in the app.py
data/problemdata contains the problem title and problem statement, this data was used to calculating the tfidf.txt, magnitude.txt, keyword.txt, idf.txt
data/problemtitles is the collection of problem titles in the entire dataset
data/problemurls is the collection of problem urls in the entire dataset
data/problems is the collection of all he problem text files, it doesnt inlcude the title of the file
LeetcodeScraped includes tfidfgen which is used to generate tfidf.txt, magnitude.txt, keyword.txt, idf.txt using the problemdata folder and the problemtitles
It also contains a test.py file which tests the algorithm of tfidf using  a samll part of the dataset (5 files) and a sample query string.
