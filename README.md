<h1>Prediction of the time of Donald Trump's tweets</h1>
<h2>Given a Donald Trump's tweet decide if it has been posted before or after Trump's announcement for running for presidency</h3>

<h3>How to use</h3>
Place all the files and folders in this repo in the same directory.<br>
Run the main.py file to train the models and save them locally. <br>
You can see the console output on the test dataset. Your output will be different from the one I provided below
because each run the train and test tweets are chosen randomly.
<br><br>
By default, the models will be saved in "saved_session" folder and can be further used to predict tweets that you provide:<br>
Put the tweets you want to predict their time (before of after Trump's announcement for running for presidency) in the file "my_tweets.txt".<br>
Each tweet in a line.<br><br>
Run the file "predict_custom_tweets.py" and see the results for each tweet in a formatted table.<br><br>
To ensure the tweets you want to predict on aren't some of the tweets the models trained on, <br>
don't predict on tweets in time ranges: 01 Jan 2013 - 20 Nov 2013, 05 May 2015 - 01 Jan 2017, 14 Jan 2014 - 06 Jan 2015.


<h3>About</h3>
<h4>What I checked:</h4>
Is there some pattern in Trumps tweets before his announcement to run to presidency compared to his tweets after that announcement?
Furhtermore, if there is a pattern do the tweets in the range of year before his announcement until his announcement (June 2014 – June 2015) are more like the tweets after the announcement? 

<h4>Datasets:</h4>
First dataset:<br>
Contains: Donald Trump's tweets before and after his announcement to run to presidency.<br>
Not including tweets in the time range of June 2014 – June 2015.<br>
Size: Tweets before the announcement: 7640,  tweets after the announcement: 8661.<br>
Second dataset:<br>
Contains: 7318 tweets in the time range of June 2014 – June 2015.<br>
<h4>How I predicted?</h4>
I used two methods to convert the raw data (tweet text) to vectors:<br>
1. Every tweet is converted to bag of words (I call it BOW).
2. Every tweet is converted to a vector of 27 cells. Every cell corresponds to the number of times each english letter or a space chararcter was in the tweet (I call it Letter vector).<br>

<br><br>
Then I used the models Neural Network, K Nearest Neighbors, Logistic regression and SVM.<br>
One time with letter vector and one time with bag of words.<br>
<br>
<br>
Also I used two Adaboosts models:<br>
1. Every combination of a model from above with a vector representation method is a classfier of the adaboost (8 classifiers).<br>
2. Converted a tweet to BOW and then reduced with PCA it's dimension to 900 (from about 14000).<br>
Then trained the Adaboost with each index of the 900 as classifiers.

<h3>Results</h3>
<h4>As you can see from the below results,<br>
I succeeded in predicting whether Trump's tweet was tweeted before or after his announcement to run for presidency!<br><br>
The best algorithm has a 86% success rate on the 6000 tweets that were allocated for the test from the first dataset!<br><br>
Furthermore, you can see that in the range of June 2014 - June 2015 it was actullay harder (less success rates) for the algorithms to predict correctly.<br>
Meaning Trump's changed his tweet pattern in this time range probably because he was going to announce he is running for presidency.</h4>

<h4>Program's output (shows the percentage of tweets it labeled correctly):</h4>
<samp>
========= Test on the tweets that has been tweeted in the range of June 2014 - June 2015 =========<br>
MLP with BOW 4184 out of 7318 = 57.17409128177098%<br>
NearestCentroid with BOW 5165 out of 7318 = 70.5793932768516%<br>
Log Reg with BOW 4874 out of 7318 = 66.60289696638426%<br>
SVM with BOW 4562 out of 7318 = 62.33943700464608%<br>
MLP with LetVec 3426 out of 7318 = 46.81606996447117%<br>
NearestCentroid with LetVec 2756 out of 7318 = 37.66056299535393%<br>
Log Reg with LetVec 3405 out of 7318 = 46.52910631320033%<br>
SVM with LetVec 2988 out of 7318 = 40.830828095107954%<br>
Adaboost with all the algorithms from before as classifiers 3856 out of 7318 = 52.69199234763596%<br>
Adaboost with every index as a classfier (after PCA to 900) 4779 out of 7318 = 65.30472806777809%<br>
========= Test on random 6000 tweets. Not including those which has been tweeted in the range of June 2014 - June 2015 =========<br>
MLP with BOW 5169 out of 6000 = 86.15%<br>
NearestCentroid with BOW 4202 out of 6000 = 70.03333333333333%<br>
Log Reg with BOW 5119 out of 6000 = 85.31666666666666%<br>
SVM with BOW 5095 out of 6000 = 84.91666666666666%<br>
MLP with LetVec 4047 out of 6000 = 67.45%<br>
NearestCentroid with LetVec 3455 out of 6000 = 57.58333333333333%<br>
Log Reg with LetVec 3881 out of 6000 = 64.68333333333334%<br>
SVM with LetVec 3863 out of 6000 = 64.38333333333334%<br>
Adaboost with all the algorithms from before as classifiers 4524 out of 6000 = 75.4%<br>
Adaboost with every index as a classfier (after PCA to 900) 3033 out of 6000 = 50.55%
</samp>
