---
layout: post
title: Predicting Great Books from Goodreads Data Using Python
subtitle: What makes a book great?
---
![books](https://images.unsplash.com/photo-1550399105-c4db5fb85c18?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1351&q=80)Photo of old books by [Ed Robertson](https://unsplash.com/@eddrobertson) on [Unsplash](https://unsplash.com/)

## What
This is a data set of the first 50,000 book ids pulled from Goodreads' API on July 30th, 2020. A few thousand ids did not make it through because the book id was changed, the URL or API broke, or the information was stored in an atypical format.

## Why
From the reader's perspective, books are a multi-hour commitment of learning and leisure (they don't call it **Good**reads for nothing). From the author's and publisher's perspectives, books are a way of living (with some learning and leisure too). In both cases, knowing which factors explain and predict great books will save you time and money. Because while different people have different tastes and values, knowing how a book is rated in general is a sensible starting point. You can always update it later.

## Environment
It's good practice to work in a virtual environment, a sandbox with its own libraries and versions, so we'll make one for this project. There are several ways to do this, but we'll use [Anaconda](https://www.anaconda.com/products/individual). To create and activate an Anaconda virtual environment called 'gr' (for Goodreads) using Python 3.7, run the following commands in your terminal or command line:

{% gist 26f74ac8a2df83aa02f7e688fc06651a %}

# Installations
You should see ‘gr’ or whatever you named your environment at the left of your prompt. If so, run these commands. Anaconda will automatically install any dependencies of these packages, including matplotlib, numpy, pandas, and scikit-learn.

```python
conda install category_encoders eli5 py-xgboost seaborn
pip install goodreads_api_client
```
<script src="https://gist.github.com/ryankoul/c3ce2d189f2ca640224dbb66ff0c9c67.js"></script>

# Imports
```python import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import category_encoders as ce
import scipy.stats as st
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.metrics import plot_confusion_matrix
import eli5
from eli5.sklearn import PermutationImportance
```

# Data Collection
We pull the first 50,000 book ids and their associated information using a lightweight wrapper around the Goodreads API made by [Michelle D. Zhang](https://medium.com/@mdzhang) (code and documentation [here](https://github.com/mdzhang/goodreads-api-client-python)), then write each as a dictionary to a JSON file called `book_data`.

```python
mport json
import time
import goodreads_api_client as gr

client = gr.Client(developer_key='YOUR DEVELOPER KEY HERE')
book_id = 1
book_list = []
book_titles = set()
starter_keys = [
    'id', 'title', 'publication_date', 'isbn', 'isbn13',
    'publication_year', 'publication_month', 'publication_day',
    'publisher', 'language_code', 'average_rating', 'num_pages', 'format'
    ]

def get_books_data(client, book_id, book_titles, books_list):
    while(book_id < 50001):
        try:
            # A big dump of information
            data = client.Book.show(book_id)
            book_dict = {k:v for k, v in data.items() if k in starter_keys}
            book_dict['rating_dist'] = data['work']['rating_dist']

            # Indexing from built-in book.work returns reivew count for ALL editions
            # book.text_reviews_count returns review count for particular edition
            book_dict['review_count'] = data['work']['text_reviews_count']['#text']

            # Get only the 5 most popular "shelves" (genres)
            genre_list = data['popular_shelves']['shelf']
            book_dict['genres'] = [genre['@name'] for genre in genre_list[:5]]

            try:
                book_dict['author'] = data['authors']['author']['name']
            except:
                book_dict['author'] = data['authors']['author'][0]['name']
            finally:
                pass

            # Prevent duplicates of same book with different goodreads ids
            if book_dict['title'] not in book_titles:
                book_list.append(book_dict)
                book_titles.add(book_dict['title'])
                yield book_dict

            # print(f'Book {book_id} read.')          # Used to test
            book_id += 1
            time.sleep(1)

        except:
            # print(f"Can't read book {book_id}")     # Used to test
            book_id += 1
            time.sleep(1)

for book_dict in get_books_data(client, book_id, book_titles, book_list):
    book_data_js = json.dumps(book_dict, indent=4)
    with open('book_data.json', 'a') as file:
        file.write(book_data_js + ',')
```

# Data Cleaning
We’ll define and describe some key functions below, but we’ll run them in one big wrangle function later.

## Wilson Lower Bound
A rating of 4 stars based on 20 reviews and a rating of 4 stars based on 20,000 reviews are not equal. The rating based on more reviews has less uncertainty about it and is therefore a more reliable estimate of the “true” rating. In order to properly define and predict great books, we must transform `average_rating` by putting a penalty on uncertainty.

We’ll do this by calculating a [Wilson Lower Bound](https://www.evanmiller.org/how-not-to-sort-by-average-rating.html), where we estimate the confidence interval of a particular rating and take its lower bound as the new rating. Ratings based on tens of thousands of reviews will barely be affected because their confidence intervals are narrow. Ratings based on fewer reviews, however, have wider confidence intervals and will be scaled down more.

**Note**: We modify the formula because our data is calculated from a 5-point system, not a binary system as described by Wilson. Specifically, we decrement `average_rating` by 1 for a conservative estimate of the true non-inflated rating, and then normalize it. If this penalty is too harsh or too light, more ratings will over time raise or lower the book’s rating, respectively. In other words, with more information, this adjustment is self-correcting.

```python
def wilson_lower_bound(average_rating, n, confidence=0.95):
    """
    Returns the lower bound of the "true" 5-star rating. Used to scale down
    ratings with fewer observations due to the greater uncertainty (wider
    confidence interval).
    Parameters:
    average_rating: the average book rating, assumed to be out of 5
    n: the total number of ratings
    confidence: statistical confidence level, default 95%
    """
    if n == 0:
        return 0                               # Cannot scale down ratings if
                                               # there are no ratings

    z = st.norm.ppf(1 - (1 - confidence) / 2)  # Z-score, default 1.96
    x_hat = (average_rating - 1) / 4           # Decrement and normalize
    wlb = (
        x_hat + z*z/(2*n) - z * np.sqrt((x_hat *(1-x_hat)+z*z/(4*n))/n))/(1+z*z/n)

    return 1 + (4 * wlb)                       # Add to 1 because minimum rating of
                                               # rated books is 1 star out of 5
```

## Genres
Goodreads’ API returns ‘shelves’, which encompass actual genres like “science-fiction” and user-created categories like “to-read”. We extracted only the 5 most popular shelves when pulling the data to limit this kind of clean-up; here, we’ll finish the job.

After some inspection, we see that these substrings represent the bulk of non-genre shelves. We’ll filter them out using a regular expression. **Note**: We use two strings in the regex so the line doesn’t get cut off. Adjacent strings inside parantheses are joined at compile time.

```python
def filter_fake_genres(genre_list):
    """
    Filters out "genres" that aren't really genres.
    Case doesn't matter since all Goodreads "shelves"
    are lower case.
    Syntax:
    ^ = start of line/string
    .* = any character except \n (0 or more times)
    (...|...) = unique substrings separated by |
    ? = preceding character is optional for match
    $  = end of line/string
    """
    genre_regex = re.compile(
        r'^.*(book|to-|read|my-|favou?rite|own|audio|library|kindle|english|default|'
        'calibre|next|want|[0-9]|best).*$')
    cleaned_genres = [genre for genre in genre_list
                      if not re.search(genre_regex, genre)]

    try:                          # Get most popular real genre from cleaned list
        return cleaned_genres[0]
    except:                       # But if it's empty, return 'unknown'
        return 'unknown'
```


# All-in-one Cleaning
Now we’ll build and run one function to wrangle the data set. This way, the cleaning is more reproducible and debug-able.

```python
def wrangle(X):
    """
    Accepts and cleans a DataFrame. Returns train set.
    """
    # Make shallow copy to avoid destroying original DataFrame
    X = X[:]

    # TITLE
    # Since ratings and reviews are for all editions, drop multiple editions
    # of the same book so every book corresponds to 1 and only 1 row
    X = X[~X.duplicated(subset='title')]

    # Engineer two features based on title
    X['title_length_in_characters'] = X['title'].apply(lambda title: len(title))
    X['title_length_in_words'] = X['title'].apply(lambda title: len(title.split()))

    # RATING DIST
    # Regex lookbehind to extract only total rating count, then cast as int
    X['total_ratings'] = X['rating_dist'].apply(
    lambda row: int(re.findall(r'(?<=:)(\w+)', row)[5]))

    # TOTAL RATINGS
    # Filter out books with less than 50 ratings.
    X = X[X['total_ratings'] >= 50]

    # AUTHOR
    # Strip all leading and trailing whitespace, and any excess middle white space
    X['author'] = [' '.join(author.split())
                    for author in X['author']]

    # GENRES
    # Filter out uninformative genres like "to-read", then extract the most popular one
    X['genres'] = [filter_fake_genres(genre_list)
                   for genre_list in X['genres']]

    # HIGH CARDINALITY
    # These unique-values columns may come in handy later if we want to merge with another dataset.
    # For now, we can drop them.
    high_cardinality = ['id', 'isbn', 'isbn13', 'title']
    X = X.drop(columns=high_cardinality)

    # AVERAGE RATING
    # Scale down average ratings based on their uncertainty, then assign them to a new column
    adjusted_averages = []
    for i in range(len(X)):
        avg = X['average_rating'].iloc[i]
        total = X['total_ratings'].iloc[i]
        adjusted_averages.append(wilson_lower_bound(avg, total))

    X['adjusted_average_rating'] = adjusted_averages

    # FEATURE ENGINEER TARGET
    # Make 'great' column for top 20% of books based on wilson-adjusted average rating
    top_20_percent = np.percentile(X['adjusted_average_rating'], q=80)
    X['great'] = X['adjusted_average_rating'] >= top_20_percent

    return X

train = wrangle(df)
```

# Compare Unadjusted and Adjusted Average Ratings
Numerically, the central measures of tendency of mean (in blue) and median (in green) slightly decrease, and the variance significantly decreases.

Visually, we can see the rating adjustment in the much smoother and wider distribution (although note that the x-axis is truncated). This is from eliminating outlier books with no or very few ratings, and scaling down ratings with high uncertainty.

```python
# Before
plt.figure(figsize=(20,10))
sns.distplot(df['average_rating'])
plt.axvline(df['average_rating'].mean(), color='b')
plt.axvline(df['average_rating'].median(), color='g')

plt.figtext(.5,.9,'Distribution of Unadjusted Average Ratings of Goodreads Books',
            fontsize=25, ha='center')
plt.xticks(fontsize=15, color='k')
plt.yticks(fontsize=15, color='k')
plt.xlabel('Average Rating', fontsize=20, color='k')
plt.show()

print('Unadjusted mean:', round(df['average_rating'].mean(), 2))
print('Unadjusted median:', round(df['average_rating'].median(), 2))
print('Unadjusted variance:', round(df['average_rating'].var(), 2))
```

![unadjusted-ratings](../img/great_reads/unadjusted_goodreads_ratings.png)


```
Unadjusted mean: 3.82
Unadjusted median: 3.93
Unadjusted variance: 0.48
```

```python
# After
plt.figure(figsize=(20,10))
sns.distplot(train['adjusted_average_rating'])
plt.axvline(train['adjusted_average_rating'].mean(), color='b')
plt.axvline(train['adjusted_average_rating'].median(), color='g')

plt.figtext(.5,.9,'Distribution of Adjusted Average Ratings of Goodreads Books',
            fontsize=25, ha='center')
plt.xticks(fontsize=15, color='k')
plt.yticks(fontsize=15, color='k')
plt.xlabel('Average Rating', fontsize=20, color='k')
plt.show()

print('Adjusted mean:', round(train['adjusted_average_rating'].mean(), 2))
print('Adjusted median:', round(train['adjusted_average_rating'].median(), 2))
print('Adjusted variance:', round(train['adjusted_average_rating'].var(), 2))
```

![adjusted-ratings](../img/great_reads/adjusted_goodreads_ratings.png)

```
Adjusted mean: 3.71
Adjusted median: 3.77
Adjusted variance: 0.17
```

# Data Leakage
Because our target is derived from ratings, training our model using ratings is effectively training with the target. To avoid distorting the model, we must drop these columns.

It is also possible that `review_count` is a bit of leakage, but it seems more like a proxy for *popularity*, not greatness, in the same way that pop(ular) songs are not often considered classics. Of course, we'll reconsider this if its permutation importance is suspiciously high.

```python
leaky = ['average_rating', 'total_ratings', 'rating_dist', 'adjusted_average_rating']
train = train.drop(columns=leaky)
```

# Split Data
We’ll do an 85/15 train-test split, then re-split our train set to make the validation set about the same size as the test set.

```python
target = 'great'

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    train, train[target], test_size=0.15,
    stratify=train[target], random_state=50)


# Split into train and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1765,
    stratify=X_train[target], random_state=50)

# Remove target
X_train = X_train.drop(columns=target)
X_val = X_val.drop(columns=target)
X_test = X_test.drop(columns=target)

# Check dimensions
# Test set size approximately equals val set size
print(X_train.shape, y_train.shape, X_val.shape,
      y_val.shape, X_test.shape, y_test.shape)
```

```
(20281, 12) (20281,) (4348, 12) (4348,) (4347, 12) (4347,)
```

# Evaluation Metrics
With classes this imbalanced, accuracy (correct predictions / total predictions) can become misleading. There just aren’t enough true positives for this fraction to be the best measure of model performance. So we’ll also use ROC AUC, a Receiver Operator Characteristic Area Under the Curve. Here is a colored drawing of one, courtesy of Martin Thoma.

![ROC-AUC](../img/great-reads/roc_auc.png)

The ROC curve is a plot of a classification model’s true positive rate (TPR) against its false positive rate (FPR). The ROC AUC is the area from [0, 1] under and to the right of this curve. Since optimal model performance maximizes true positives and minimizes false positives, the optimal point in this 1x1 plot is the top left, where the area under the curve (ROC AUC) = 1.

For imbalanced classes such as `great`, ROC AUC outperforms accuracy as a metric because it better reflects the relationship between true positives and false positives. It also depicts the classifier’s performance across all its values, giving us more information about when and where the model improves, plateaus, or suffers.


# Fit Models
Predicting great books is a binary classification problem, so we need a classifier. Below, we’ll encode, impute, and fit to the data a linear model (Logistic Regression) and two tree-based models (Random Forests and XGBoost), then compare them to each other and to the majority baseline. We’ll calculate their accuracy and ROC AUC, and then make a visualization.

## Majority Class Baseline

First, by construction, `great` books are the top 20% of books by Wilson-adjusted rating. That means our majority class baseline (no books are great) has an accuracy of 80%.

Second, this “model” doesn’t improve, plateau, or suffer since it has no discernment to begin with. A randomly chosen positive would be treated no differently than a randomly chosen negative. In other wrods, its ROC AUC = 0.5.

```python
baseline_accuracy = round(train['great'].value_counts(normalize=True)[0], 4)

# Majority baseline is "No books are great", so make
# whole column of False (condition is never True)
maj_pred = y_val == 'a;sdlkfn' * len(y_val)
baseline_roc_auc = round(roc_auc_score(y_val, maj_pred), 4)

print('Baseline Validation Accuracy:', baseline_accuracy)
print('Baseline Validation ROC AUC:', baseline_roc_auc)   # 0.5 by definition
```

```
Baseline Validation Accuracy: 0.8
Baseline Validation ROC AUC: 0.5
```

## Logistic Regression
Now we’ll fit a linear model with cross-validation, re-calculate evaluation metrics, and plot a confusion matrix.

```python
lr = make_pipeline(
    ce.OrdinalEncoder(),
    SimpleImputer(),
    StandardScaler(),
    LogisticRegressionCV(cv=20, n_jobs=-1, scoring='roc_auc', max_iter=1000)
)

lr.fit(X_train, y_train)
lr_roc_auc = round(lr.score(X_val, y_val), 4)

lr_pred = lr.predict(X_val)
lr_accuracy = round(accuracy_score(y_val, lr_pred), 4)

print('Baseline Validation Accuracy:', baseline_accuracy)
print('Logistic Regression Validation Accuracy:', lr_accuracy)
print('Baseline Validation ROC AUC:', baseline_roc_auc)
print('Logistic Regression Validation ROC AUC:', lr_roc_auc)
```

```
Baseline Validation Accuracy: 0.8
Logistic Regression Validation Accuracy: 0.8013
Baseline Validation ROC AUC: 0.5
Logistic Regression Validation ROC AUC: 0.6424
```

## Logistic Regression Confusion Matrix
```python
disp_lr = plot_confusion_matrix(lr, X_val, y_val,cmap=plt.cm.Blues)
disp_lr.ax_.set_title('Logistic Regression Confusion Matrix')
plt.show()
```

![lr-confusion-matrix](../img/great_reads/lr_confusion_matrix.png)

# Random Forest Classifier
Now we’ll do the same as above with a tree-based model with bagging (**B**ootstrap **AGG**regation).

```python
rf = make_pipeline(
    ce.OrdinalEncoder(),
    SimpleImputer(),
    RandomForestClassifier(n_estimators=1000, n_jobs=-1, max_features=None)
)

rf.fit(X_train, y_train)
rf_accuracy = round(rf.score(X_val, y_val), 4)

rf_probs = rf.predict_proba(X_val)[:,1]   # All probs in 1d array
rf_roc_auc = round(roc_auc_score(y_val, rf_probs), 4)

print('Baseline Validation Accuracy:', baseline_accuracy)
print('Logistic Regression Validation Accuracy:', lr_accuracy)
print('Random Forest Validation Accuracy:', rf_accuracy)
print()
print('Majority Class Baseline ROC AUC:', baseline_roc_auc)
print('Logistic Regression Validation ROC AUC:', lr_roc_auc)
print('Random Forest Validation ROC AUC:', rf_roc_auc)
```

```
Baseline Validation Accuracy: 0.8
Logistic Regression Validation Accuracy: 0.8013
Random Forest Validation Accuracy: 0.8222

Majority Class Baseline ROC AUC: 0.5
Logistic Regression Validation ROC AUC: 0.6424
Random Forest Validation ROC AUC: 0.8015
```

## Random Forest Confusion Matrix
```python
disp_rf = plot_confusion_matrix(rf, X_val, y_val, cmap=plt.cm.Blues)
disp_rf.ax_.set_title('Random Forest Confusion Matrix')
plt.show()
```

![rf-confusion-matrix](../img/great_reads/rf_confusion_matrix.png)

## XGBoost Classifier
Now we’ll do the same as above with another tree-based model, this time with boosting.

```python
gb = make_pipeline(
    ce.OrdinalEncoder(),
    SimpleImputer(),
    XGBClassifier(n_estimators=1000, n_jobs=-1, max_depth=13, learning_rate=0.1)
)

gb.fit(X_train, y_train)
gb_accuracy = round(gb.score(X_val, y_val), 4)

gb_probs = gb.predict_proba(X_val)[:,1]   # All probs in 1d array
gb_roc_auc = round(roc_auc_score(y_val, gb_probs), 4)

print('Baseline Validation Accuracy:', baseline_accuracy)
print('Logistic Regression Validation Accuracy:', lr_accuracy)
print('Random Forest Validation Accuracy:', rf_accuracy)
print('XGBoost Validation Accuracy:', gb_accuracy)
print()
print('Majority Class Baseline ROC AUC:', baseline_roc_auc)
print('Logistic Regression Validation ROC AUC:', lr_roc_auc)
print('Random Forest Validation ROC AUC:', rf_roc_auc)
print('XGBoost Validation ROC AUC', gb_roc_auc)
```

```
Baseline Validation Accuracy: 0.8
Logistic Regression Validation Accuracy: 0.8013
Random Forest Validation Accuracy: 0.8245
XGBoost Validation Accuracy: 0.8427

Majority Class Baseline ROC AUC: 0.5
Logistic Regression Validation ROC AUC: 0.6424
Random Forest Validation ROC AUC: 0.8011
XGBoost Validation ROC AUC 0.84
```

XGBClassifier performes the best in accuracy and ROC AUC.

```python
y_pred = gb.predict(X_test)
class_index = 1
y_pred_proba = gb.predict_proba(X_test)[:, class_index]

print('XGBoost Test Accuracy', round(accuracy_score(y_test, y_pred), 4))
print('XGBoost Test ROC AUC', round(roc_auc_score(y_test, y_pred_proba), 4))
```

# Graph and Compare Models’ ROC AUC
Below, we see that logistic regression lags far behind XGBoost and Random Forests in achieving a high ROC AUC. Among the top two, XGBoost initially outperforms RandomForest, and then the two roughly converge around FPR=0.6. We see in the lower right legend, however, that XGBoost has the highest AUC of 0.84, followed by Random Forest at 0.80 and Logistic Regression at 0.64.

In less technical language, the XGBoost model was the best at classifying great books as great (true positives) and not classifying not-great books as great (false positives).

<script src="https://gist.github.com/ryankoul/505724d82b149a7693c3dddbe77814c8.js"></script>

![roc-auc-of-all-models](../img/great_reads/roc_auc_of_all_models.png)

# Permutation Importances
One intuitive way of identifying whether and to what extent something is important is by seeing what happens when you take it away. This is the best in a situation unconstrained by time and money.

But in the real world with real constrains, we can use permutation instead. Instead of eliminating the column values values by dropping them, we eliminate the column’s *signal* by randomizing it. If the column really were a predictive feature, the order of its values would matter, and shuffling them around would substantially dilute if not destroy the relationship. So if the feature’s predictive power *isn’t* really hurt or is even helped by randomization, we can conclude that it is not actually important.

Let’s take a closer look at the permutation importances of our XGBoost model. We’ll have to refit it to be compatible with eli5.

<script src="https://gist.github.com/ryankoul/5c4f83686ee396990e9084a6034cdd98.js"></script>

## Permutation Importance Analysis
![permutation-importance](../img/great_reads/permutation_importances.png)
As we assumed at the beginning, `review_count` matters but it is not suspiciously high. This does not seem to rise to the level of data leakage. What this means is that if you were wondering what book to read next, a useful indicator is how many reviews it has, a proxy for how many others have read it.

We see that `genres` is the second most important feature for ROC AUC in the XGBoost model.

`author` is third, which is surprising and perhaps a bit concerning. Because our test set is not big, the model may just be identifying authors whose books are the most highly rated in wilson-adjusted terms, such as J.K. Rowling and Suzanne Colins. More data would be useful to test this theory.

Fourth is `num_pages`. I thought this would be higher for two reasons:
1. Very long books’ ratings seem to have a bit of a ratings bias upward in that people willing to start and finish them will rate them higher. The long length screens out less interested marginal readers, who probably wouldn’t have rated the book highly in the first place.
2. Reading and showing off that you’re reading or have read long books is a sign of high social status. The archetypal example: Infinite Jest.

# Takeaway
We’ve seen how to collect, clean, analyze, visualize, and model data. Some actionable takeaways are that when and who publishes a book doesn’t really matter, but its review count does — the more reviews, the better.

For further analysis, we could break down `genres` and `authors` to find out which ones were rated highest. For now, happy reading.