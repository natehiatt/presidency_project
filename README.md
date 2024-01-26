# Predictive NLP Classification of Presidential Rhetoric
Nate Hiatt (natehiatt@gmail.com), Nathan Bass (bassn727@gmail.com), Shelley Wang (ShelleyLWang@gmail.com) 
![ReadMe header](images/readmeheader.png)

## Background
In this project, we've gathered a rich collection of political texts, including speeches from [The Republican/Democratic National Convention](https://www.presidency.ucsb.edu/documents/app-categories/elections-and-transitions/convention-speeches), [Presidential Inaugural Addresses](https://www.presidency.ucsb.edu/documents/app-categories/spoken-addresses-and-remarks/presidential/inaugural-addresses) and [transcripts of Presidential Debates](https://www.presidency.ucsb.edu/documents/app-categories/elections-and-transitions/debates). These texts were sourced from the University of California, Santa Barbara's Presidency Project website. The collection focuses on material from 1960 onwards, as this is the period from which we have complete debate transcripts available. We utilized BeautifulSoup for web scraping, effectively extracting the texts and integrating them into a structured dataframe for further processing.

## Data Understanding and Preparation
Our initial dataset comprised over 100 distinct texts, each associated with a name and year. The first step in our data preparation involved tokenizing the textual content. We employed Python and the Natural Language Toolkit (nltk) for this purpose, segmenting the texts into individual tokens. This tokenization process resulted in a single list object, with each token as an element. The tokenized data was then fed into our analytical pipeline, where it was vectorized to facilitate modeling. 

## Modeling
We instatiate a Multinomial Naive Bayes model. This model uses Bayes probability to statistically test the hypothesis that a text of a document belongs to a certain class (in this case, a political party). We also instatiate a Tf-Idf Vectorizer. This type of vectorizer is very powerful for content-based classification because adds importance weight to certain tokens using a tf-idf score. The higher the tf-idf score, the more important that word is in that document compared to how important it is in all the documents. 

We also instatiated a Guassian Bayes model, which is specific to binary classification tasks.

Finally, for text data, the document term matrix returned by a vectorizer is typically a sparse matrix, since there are many tokens that each document does not have (i.e. there are lots of 0 values). This means our model has a very high number of features/columns/words. Tree-based models work very well with high dimensional data, so we instatiated Random Forest model.

After grid searching all three models using different hyperparameters, we compared all their accuracy scores to select the best performing model.


## Results
Our best model is a Multinomial Naive Bayes model. On our training dataset, when our model classified whether a speech was made by a Democrat or a Republican it was right 98% of the time. On unseen testing data, its classification was right 87% of the time.

The model is overfit, which means it could use further tuning to reduce this disparity. One such possible method is Principal Component Analysis. PCA reduces the number of features (and thus the complexity of the model) by creating components consisting of similar features, ones that move in a similar direction. Our group did not have the time for this method, but we consider it in our next steps.

Overall, we’re still getting highly accurate party predictions based just on what a candidate said.

## Next Steps
So what about going forward?

The model is overfit, which means it could use further tuning. One such possible method is Principal Component Analysis. PCA reduces the number of features (and thus the complexity of the model) by creating components consisting of similar features, ones that move in a similar direction. It tries to preserve as much of the variance as possible in those features, so that they describe more of the variance in the target even with a lower dimensional space. Our group did not have the time for this method.

The model would also benefit from a larger data training set. In particular, it would be helpful to pull in campaign stops and other less formal speech occasions. Including candidates for party nominations who nonetheless failed to become the party nominee would also be worthwhile. It is worth considering bringing in other political rhetoric, not merely from those seeking presidential office, although that may go beyond the scope of this particular dataset and model.

With our trained model, there are other analyses that would be worth pursuing. To name a few: how much does rhetoric change before and after a politician becomes his or her party’s nominee? What about once they win the election? And how much does context affect rhetoric: a town hall, versus cable news, versus a formal press conference, and so on?

Finally, we would want to allow others to make use of this model as they see fit. It could be useful to the public to have a front-facing website that allows individuals to input text and get a likelihood prediciton of the speaker's party affiliation.
  
## Repo Structure
```
├── data
│   ├── 
├── Images
│   ├── 
├── Notebooks
│   ├── nathan_working.ipynb
│   ├── nate_scratch.ipynb
│   ├── shelley_scratch.ipynb
├── presentation.pdf
├── .gitignore
├── Final.ipynb
├── LICENSE
├── README.md
```
