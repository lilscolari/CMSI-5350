1.	I used ChatGPT to build an initial understanding of the DecisionTreeClassifier from scikit-learn. It showed me how the Classifier can be used to train data, make predictions,
and evaluate the predictions. It even showed me how to graph the tree. From what I saw, it was accurate in its explanations and code. It was helpful for me because I was able to
see the methods of the Classifier that I would be using for this ps.

3.	a. The training error is 0.013.

b. _Majority Vote Classifier_ average training error: 0.404 and average test error: 0.407

_Decision Tree Classifier_ average training error: 0.013 and average test error: 0.239

_Random Classifier_ average training error: 0.489 and average test error: 0.487

c. I found that the best depth limit to use for this data from 3. I do see overfitting in this plot because the training error keeps on falling while the test error falls until a
certain depth and then continues to get bigger. This is a common sign of overfitting. At depth=3, you can see how the test error stops getting better and continues to get worse
afterward while the training error only gets better.

d.  ![image](https://github.com/user-attachments/assets/f2622dc0-c8e9-47ff-9c20-13a09a4e1bf0)
This plot shows how the training error increases and then stabilizes at a certain point as the amount of training data increases. On the other hand, the test error, which is what
we truly care about, continues to decrease as the amount of training data increases because the model is no longer overfitting and is generalizing.

4. I wanted to test if max_features would reduce the test error so I ran everything at my max depth that I found earlier of 3 and tried max_features with inputs from 1 through 5 as
well as using random_state from 0 to 100. The lowest test error I encountered was 0.189. To see if I could improve upon this error, I kept the max_features parameter and with the
random_state from 0 to 100. On top of that, I thought it might be interesting to combine the SibSp (siblings and spouse on board) and Parch (parents and children on board) columns
to just be one column. The smallest test error I encountered from this was 0.182. I stored this model and used it to predict on the test dataset provided. I noticed that when sex=0,
my model seemed to predict survived=1 and when sex=1, my model seemed to predict survived=0.

__Additional Questions:__

0. 5 hours

2. I enjoyed this ps because I feel like I got to apply a real machine learning model to real world data and see firsthand how a model can be trained and evaluated. This was super fun.
