# Feature engineering from volume

We're going to use non-linear models to make more accurate predictions. With linear models, features must be linearly correlated to the target. Other machine learning models can combine features in non-linear ways. For example, what if the price goes up when the moving average of price is going up, and the moving average of volume is going down? The only way to capture those interactions is to either multiply the features, or to use a machine learning algorithm that can handle non-linearity (e.g. random forests).

To incorporate more information that may interact with other features, we can add in weakly-correlated features. First we will add volume data, which we have in the lng_df as the Adj_Volume column.

## Feature instructions

1.    Create a 1-day percent change in volume (use pct_change() from pandas), and assign it to the Adj_Volume_1d_change column in lng_df.
2.    Create a 5-day moving average of the 1-day percent change in Volume, and assign it to the Adj_Volume_1d_change_SMA column in lng_df.
3.    Plot histograms of these two new features we created using the new_features list.


```markdown
# Create 2 new volume features, 1-day % change and 5-day SMA of the % change
new_features = ['Adj_Volume_1d_change', 'Adj_Volume_1d_change_SMA']
feature_names.extend(new_features)
lng_df['Adj_Volume_1d_change'] = lng_df['Adj_Volume'].pct_change(1)
lng_df['Adj_Volume_1d_change_SMA'] = talib.SMA(lng_df['Adj_Volume_1d_change'].values,timeperiod=5)

# Plot histogram of volume % change data
lng_df[new_features].plot(kind='hist', sharex=False, bins=50)
plt.show()
```
# Create day-of-week features

We can engineer datetime features to add even more information for our non-linear models. Most financial data has datetimes, which have lots of information in them -- year, month, day, and sometimes hour, minute, and second. But we can also get the day of the week, and things like the quarter of the year, or the elapsed time since some event (e.g. earnings reports).

We are only going to get the day of the week here, since our dataset doesn't go back very far in time. The dayofweek property from the pandas datetime index will help us get the day of the week. Then we will dummy dayofweek with pandas' get_dummies(). This creates columns for each day of the week with binary values (0 or 1). We drop the first column because it can be inferred from the others.

## Create day-of-week instructions

1    Use the dayofweek property from the lng_df index to get the days of the week.
2    Use the get_dummies function on the days of the week variable, giving it a prefix of 'weekday'.
3    Set the index of the days_of_week variable to be the same as the lng_df index so we can merge the two.
4    Concatenate the lng_df and days_of_week DataFrames into one DataFrame.

```
# Use pandas' get_dummies function to get dummies for day of the week
days_of_week = pd.get_dummies(lng_df.index.dayofweek,
                              prefix='weekday',
                              drop_first=True)

# Set the index as the original dataframe index for merging
days_of_week.index = lng_df.index

# Join the dataframe with the days of week dataframe
lng_df = pd.concat([lng_df, days_of_week], axis=1)

# Add days of week to feature names
feature_names.extend(['weekday_' + str(i) for i in range(1, 5)])
lng_df.dropna(inplace=True)  # drop missing values in-place
print(lng_df.head())
```
# Examine correlations of the new features

Now that we have our volume and datetime features, we want to check the correlations between our new features (stored in the new_features list) and the target (5d_close_future_pct) to see how strongly they are related. Recall pandas has the built-in .corr() method for DataFrames, and seaborn has a nice heatmap() function to show the correlations.

## Examine instructions

1    Extend our new_features variable to contain the weekdays' column names, such as weekday_1, by concatenating the weekday number with the 'weekday_' string.
2    Use Seaborn's heatmap to plot the correlations of new_features and the target, 5d_close_future_pct.

```
# Add the weekday labels to the new_features list
new_features.extend(['weekday_' + str(i) for i in range(1, 5)])

# Plot the correlations between the new features and the targets
sns.heatmap(lng_df[new_features + ['5d_close_future_pct']].corr(), annot=True)
plt.yticks(rotation=0)  # ensure y-axis ticklabels are horizontal
plt.xticks(rotation=90)  # ensure x-axis ticklabels are vertical
plt.tight_layout()
plt.show()
```

# Fit a decision tree

Random forests are a go-to model for predictions; they work well out of the box. But we'll first learn the building block of random forests -- decision trees.

Decision trees split the data into groups based on the features. Decision trees start with a root node, and split the data down until we reach leaf nodes.

decision tree

We can use sklearn to fit a decision tree with DecisionTreeRegressor and .fit(features, targets).

Without limiting the tree's depth (or height), it will keep splitting the data until each leaf has 1 sample in it, which is the epitome of overfitting. We'll learn more about overfitting in the coming chapters.

## Fit instructions

1    Use the imported class DecisionTreeRegressor with default arguments (i.e. no arguments) to create a decision tree model called decision_tree.
2    Fit the model using train_features and train_targets which we've created earlier (and now contain day-of-week and volume features).
3    Print the score on the training features and targets, as well as test_features and test_targets.

```
from sklearn.tree import DecisionTreeRegressor

# Create a decision tree regression model with default arguments
decision_tree = ____

# Fit the model to the training features and targets
decision_tree.fit(____)

# Check the score on train and test
print(decision_tree.score(train_features, train_targets))
print(decision_tree.score(____))
```

# Try different max depths

We always want to optimize our machine learning models to make the best predictions possible. We can do this by tuning hyperparameters, which are settings for our models. We will see in more detail how these are useful in future chapters, but for now think of them as knobs we can turn to tune our predictions to be as good as possible.

For regular decision trees, probably the most important hyperparameter is max_depth. This limits the number of splits in a decision tree. Let's find the best value of max_depth based on the R2
score of our model on the test set, which we can obtain using the score() method of our decision tree models.

## Try different max depths instructions

1 Loop through the values 3, 5, and 10 for use as the max_depth parameter in our decision tree model.
2 Set the max_depth parameter in our DecisionTreeRegressor to be equal to d in each loop iteration.
3 Print the model's score on the train_features and train_targets

```
# Loop through a few different max depths and check the performance
for d in [____]:
    # Create the tree and fit it
    decision_tree = DecisionTreeRegressor(____)
    decision_tree.fit(train_features, train_targets)

    # Print out the scores on train and test
    print('max_depth=', str(d))
    print(decision_tree.score(____))
    print(decision_tree.score(test_features, test_targets), '\n')
    
```
