---
title       : Dynamic Time Warping
description : Dynamic time warping is a useful similarity score that allows comparisons of unequal length time-series, with different phases, periods, and speeds.
attachments :
slides_link : http://sflscientific.com/presentations-and-conference-talks/

--- type:MultipleChoiceExercise lang:python xp:50 skills:1 key:5277a8088f
## Introduction

Typical time-series techniques usually apply algorithms, be it SVMs, logistic regressions, decision trees etc,  after transforming away temporal dynamics through feature engineering. By creating features, we often remove any underlying temporal information, resulting in a loss of predictive power.

Dynamic time warping (DTW) is a useful distance-like/similarity score that allows comparisons of two time-series sequences with varying lengths and speeds. Simple examples include detection of people 'walking' via wearable devices, arrhythmia in ECG, and speech recognition. 


Is DTW a distance/similarity score?
*** =instructions
- True 
- False

*** =hint


*** =pre_exercise_code
```{r}
```

*** =sct
```{r}
msg_bad = "That is not correct!"
msg_success = "Exactly!"
test_mc(1, [msg_success,msg_bad])
```

This measure distinguishes the underlying pattern rather than looking for an exact match in the raw time-series. As its name suggestions, the usual Euclidean distance in problems is replaced with a dynamically adjusted score. DTW thus allows us to retain the temporal dynamics by directly modeling the time-series. 

Much of the following material is taken from our blog and case studies from our website: http://www.sflscientific.com

What cases is DTW useful for when comparing two time-series? Time-series have:
*** =instructions
- different phases 
- different total lengths
- different pause lengths
- all of the above

*** =hint
DTW is very useful!

*** =pre_exercise_code
```{r}
```

*** =sct
```{r}
msg_bad = "That is not correct!"
msg_success = "Exactly!"
test_mc(4, [msg_bad,msg_bad,msg_bad,msg_success])
```

--- type:MultipleChoiceExercise lang:python xp:50 skills:2 key:7f5cb08579
## DTW as a distance

Figure 1 shows what happens when we compare two time-series, symbolised as a red wave and a blue wave. The top image shows Euclidean matching, which is 1-1. The similarity score between these waves measured by a Euclidean metric would be poor, even though the rough shape of the waves (i.e. the peaks, troughs and plateaus) are similar.

The bottom image shows a similarity score that would be given by Dynamic Time Warping. You can see that the rough shapes of the waves would match up, giving a high similarity score. Notice that DTW is not exactly a distance metric as the triangle inequality does not hold.

In general, events that have similar shapes but different magnitudes, lengths and especially phases can prevent a machine from correctly identifying sequences as similar events using traditional distance metrics. DTW allows us to get around this issue by asynchronously mapping the curves together.

<center><img src="https://raw.githubusercontent.com/mmluk/DTWTutorial/master/Images/path_differences.png"></center>

The figure above left shows the typical Euclidean matching between two waves.  Starting in the bottom left, the first instance in the sequence of the time-series A and B are compared to each other. Then the second instance is compared to the second and so on until the end of one of the shorter sequences.

For DTW, thec figure above right, represents a walk over the optimal path. The optimal path is determined by finding the maximum similarity score between the two time-series.

<center><img src="https://raw.githubusercontent.com/mmluk/DTWTutorial/master/Images/allpaths.png"></center>

To find the optimal path, DTW checks all possible paths (subject to certain constraints) from the bottom left to the top right, computing the equivalent of a similarity score between the waves for each. The one with the largest similarity is kept.

Have a look at the plot that showed up in the viewer. Which type of measure would give a greater similarity between red and blue curves, when you use the black vertical lines to compare points?

In which case will the similarity score be better (i.e. the case where the waves are more similar): 
*** =instructions
- Euclidean matching better
- DTW matching better

*** =hint
Have a look at the plot. In which case, are points in the curves connected by the black lines more similar.

*** =pre_exercise_code
```{r}
# The pre exercise code runs code to initialize the user's workspace.
# You can use it to load packages, initialize datasets and draw a plot in the viewer

# from IPython.display import Image
# import matplotlib.pyplot as plt
# %matplotlib inline
# %pylab inline
```

*** =sct
```{python}
msg1 = "Great job!"
msg2 = "Wrong, try again. Maybe have a look at the hint."
test_mc(correct = 2, msgs = [msg2, msg1])

success_msg("Well done! Now move on and explore some of the features in more detail.")
```

--- type:NormalExercise lang:python xp:100 skills:1 key:fd14aa3eaa
## A Simple Example

Let's start with a naive speech recognition example of DTW to show how the algorithm works, and then we will suggest a more complicated version of the analysis that can be found on our website.

A dataset of file labels, `labels`, and data `data`,  is available in the workspace.

Both `labels` and `data` are stored in numpy arrays and can be accessed as a standard array. Let's have a look at what the raw data looks like.

What labels to the files 0 and 8 have? 
*** =instructions
- Import matplotlib.pyplot as `plt`
- Use `plt.plot()` to plot `data[0]` and `data[8]` onto the same image. You should use the first positional argument, and the `label` keyword, `alpha` keyword with 0.2 as the shading.
- Show the plot using `plt.show()`.

*** =hint
You don't have to program anything for the first instruction, just take a look at the first line of code.
- Use `import ___ as ___` to import `matplotlib.pyplot` as `plt`.
- Use `plt.plot(___,label=___,alpha=0.2)` for the third instruction.

*** =pre_exercise_code
```{python}
# import scipy libraries to load data
import scipy.io.wavfile
import scipy.signal as sig
# https://raw.githubusercontent.com/mmluk/DTWTutorial/master/

#with open('https://raw.githubusercontent.com/mmluk/DTWTutorial/master/data/sounds/wavToTag.txt') as f:
#    labels = np.array([l.replace('\n', '') for l in f.readlines()])

import pandas as pd
f = pd.read_csv('http://raw.githubusercontent.com/mmluk/DTWTutorial/master/data/sounds/wavToTag.txt',delim_whitespace=True,header=None)
labels = f.ix[:,0].tolist()

data = []
data.append(scipy.io.wavfile.read('https://drive.google.com/open?id=0B7F8BsDEet-FSDluM0JzSjd5NDg')[1])

for i in range(187):
  data.append(scipy.io.wavfile.read('https://raw.githubusercontent.com/mmluk/DTWTutorial/master/data/sounds/{}.wav'.format(i))[1])

import numpy as np
```

*** =sample_code
```{python}
# Show the labels for file 0 and 8.
plot('label for file 0 is:',labels[0])



# Import matplotlib.pyplot

# Make a scatter plot: with data of files 0 and 8 and set c to ints
plt.plot(data[0], label='Sample '+str(id1))



plt.title('Raw data for two speakers',size=20)
plt.ylabel('Amplitude')
plt.xlabel('Measurement')
plt.legend()
```

*** =solution
```{python}
# Get integer values for genres
print('label for file 0:',labels[0])
print('label for file 8:',labels[8])

# Import matplotlib.pyplot
import matplotlib.pyplot as plt


# plot the raw audio data
plt.plot(data[0], label='Sample '+str(id1))
plt.plot(data[8],alpha=0.2, label='Sample '+str(id2))
plt.title('Raw data for two speakers',size=20)
plt.ylabel('Amplitude')
plt.xlabel('Measurement')
plt.legend()

```

*** =sct
```{python}
# SCT written with pythonwhat: https://github.com/datacamp/pythonwhat/wiki

test_object("data",
            undefined_msg = "Don't remove the definition of the predefined `data` object.",
            incorrect_msg = "Don't change the definition of the predefined `data` object.")
test_object("labels",
            undefined_msg = "Don't remove the definition of the predefined `labels` object.",
            incorrect_msg = "Don't change the definition of the predefined `labels` object.")
            
test_import("matplotlib.pyplot", same_as = True)

test_function("matplotlib.pyplot.plot",
              incorrect_msg = "You didn't use `plt.plot()` correctly, have another look at the instructions.")


success_msg("Great work!")
```


Let's also check the lengths of the file:
```{python}
print('File Lengths:',len(data[0]),',',len(data[8]))
```

Notice the file lengths are significantly different. Comparing directly two uneven length vectors is already quite unnatural in most situations when using standard distance metrics. 

DTW gets around these problems since the underlying algorithm and distance metrics don't really care about the lengths of the file.

In our data set we have the following different labels:
```{python}
print('unique labels:',set(labels))
```


For computational ease, we will downsample all the data to 1000 Hz, from 44kHz and also normalise the data.
```{python}
# Downsample the data from rate [/s] to new_rate [/s] and normalise (mean and variance) it by hand
rate = 44100
new_rate = 1000
data_sampled = sig.decimate(data[0], int(rate/new_rate), axis=0, ftype='fir')
data_sampled -= np.nanmean(data_sampled)
data_sampled /= np.nanstd(data_sampled)

data_sampled2 = sig.decimate(data[8], int(rate/new_rate), axis=0, ftype='fir')
data_sampled2 -= np.nanmean(data_sampled2)
data_sampled2 /= np.nanstd(data_sampled2)
```

We plot the normalised data for the same samples:
```{python}
plt.plot(data_sampled, label='Sample '+str(id1))
plt.plot(data_sampled2, alpha=0.2, label='Sample '+str(id2))
plt.title('Normalised Data for two speakers',size=20)
plt.ylabel('Amplitude')
plt.xlabel('Measurement')
plt.legend()
```

For convenience, let's put the normalisation into a single function
```{python}
def normalise_data(data, original_rate, final_rate):
  import scipy.signal as sig
   
  data_sampled = sig.decimate(data, int(rate/new_rate), axis=0, ftype='fir')
  data_sampled -= np.nanmean(data_sampled)
  data_sampled /= np.nanstd(data_sampled)

  return data_sampled
```

Let's reshape the data for input into dtw function
```{python}
x = data_sampled.reshape(-1,1)
y = data_sampled2.reshape(-1,1)
```

Let's compute our first DTW distance between these two audio clips:
```{python}
# compute the norm (x-y)
dist, cost, acc, path = dtw(x, y, dist=lambda x, y: np.linalg.norm(x - y, ord=1))
print('Minimum distance found:', dist)
```

Notice that we need to specify a distance score `dist`. This is a norm used to compare the elements of each step, with the sum of these the total cost of the path. We can plot the optimal path that the algorithm using this particular norm as follows:

```{python}
# plot the path
plt.imshow(acc.T, origin='lower', cmap=cm.gray, interpolation='nearest')
plot(path[0], path[1], 'w')
xlim((-0.5, acc.shape[0]-0.5))
ylim((-0.5, acc.shape[1]-0.5))
xlabel('Sample '+str(id1))
ylabel('Sample '+str(id2))
plt.title('np.linalg.norm(x-y)')
```

--- type:NormalExercise lang:python xp:100 skills:1 key:10223ad899
## Needs Exercise title
You can also specify your own norm used to determine the cost measure by the DTW.
*** =instructions
- Define a `my_custom_norm` function that that takes two arguments `x` and `y` and returns the square of the difference
- Use your custom norm in the computation for the dtw algorithm
- Print out the minimum distance
- Plot the minimum distance path taken by the algorithm

*** =hint
- The square of the difference is (x-y)*(x-y)
- Replace the argument dist=... with dist=my_custom_norm
- use the above code example to replot with your results

*** =sample_code
```{python}
# define your custom norm
def my_custom_norm(x,y):

# run dtw

# print out the minimum distance between x and y

# plot the path
```

*** =solution
```{python}
def my_custom_norm(x, y):
    return (x - y)*(x - y)

dist, cost, acc, path = dtw(x, y, dist=my_custom_norm)
print('Minimum distance found:', dist)

# with path
plt.imshow(acc.T, origin='lower', cmap=cm.gray, interpolation='nearest')
plot(path[0], path[1], 'w')
xlim((-0.5, acc.shape[0]-0.5))
ylim((-0.5, acc.shape[1]-0.5))
plt.title('custom norm')
```


*** =sct
```{python}
# SCT written with pythonwhat: https://github.com/datacamp/pythonwhat/wiki

test_function("my_custom_norm",
              incorrect_msg = "You didn't define `my_custom_norm` correctly, have another look at the instructions.")

success_msg("Great work!")
```


You can obviously play with the variety of norm you would like, some will be more suitable for different cases as with all normalisation methods and represent a good paramter to tune.

For simplicity, let's just use the built-in norm from numpy, which gives good results (determined by the minimum distance found). 
```{python}
from numpy.linalg import norm
dist, cost, acc, path = dtw(x, y, dist=norm)

imshow(acc.T, origin='lower', cmap=cm.gray, interpolation='nearest')
plot(path[0], path[1], 'w')
xlim((-0.5, acc.shape[0]-0.5))
ylim((-0.5, acc.shape[1]-0.5))
plt.title('np.linalg default norm')
print('Minimum distance found:', dist)
```

Lets apply this to all files in the dataset. For this naive example, let's find one file of each word type and use this as a template training example. Obviously this is not ideal, in accuracy or complexity but there are use cases where we know some underlying true 'template' that we want to dig out of some time-series data. 

We will revisit a better algorithm afterwards.


## Naive DTW Classifier - using only minimum distance to training example
```{python}
template_labels = {}
template_data = {}
training_indices = []
new_rate = 200
for l in set(labels):
    first_index = (list(labels).index(str(l)))
    rate,data = scipy.io.wavfile.read('data/sounds/{}.wav'.format(first_index))
    data_sampled = normalise_data(data,rate,new_rate)
    
    template_data[l] = data_sampled
    template_labels[l] = first_index
    training_indices.append(first_index)
```

We will use one example of each label as a traning set.
```{python}
print('The training indices that we will use are:',training_indices)
```
We also relabel our labels as truth, for emphasis:
```{python}
true_label = labels
```

We now loop over all files in our dataset and compare it with each template. The comparison that yields the smallest distance (i.e. the highest similarity) will be used as the label. This is basically a naive k-Nearest Neigbhour algorithm with only one of each label in the training set. 

**THIS WILL TAKE A WHILE!!!**
```{python}
pred_label = []
new_rate = 200
print('total files:',len(labels))
for f in range(0,len(labels)):
    if not (f % 25):
        print('working on file',f, 'true label = ',true_label[f])
    test_rate,test_data  = scipy.io.wavfile.read('data/sounds/{}.wav'.format(f))
    
    # down sample and normalise
    data_sampled = normalise_data(test_data,test_rate,new_rate)
    
    # initialise some variables
    min_dist = np.inf
    for template in template_data:
        # compute the distance to each template one at a time
        dist, _, _, _ = dtw(data_sampled.reshape(-1,1), template_data[str(template)].reshape(-1,1), dist=lambda x, y: np.linalg.norm(x - y, ord=1))
        if dist < min_dist:
            # save the current best template details
            min_dist = dist
            pred = template
  
    if not (f % 25):
        print('Completed file',f,'closest match',pred)
    
    # save the predicted labels
    pred_label.append(pred)
  ```

We now have a list of predicted labels `pred_label`, one for each of the original 245 files - i.e. we have also included the template waves in this list. To get a proper validation score, we will remove the training template instances.
  
To do this, let's define a plotting script:
```{python}
# define a plotting script for the confusion matrix
def plot_confusion_matrix(cm,target_names=[], title='Confusion matrix', cmap=plt.cm.Greens):
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    
    print(classification_report(test_pred_int, test_true_int, target_names=target_names))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(5,5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(label_list))
    plt.xticks(tick_marks, label_list, rotation=90)
    plt.yticks(tick_marks, label_list)
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
```
The plotting script requires a list of ints rather than strs so we convert the list of str to list of ints. We first define a dictionary `label_dict` to convert the labels to ints, and also an array `label_list` to convert the ints back to the labels.
```{python}
# create a dict of str names to int names
label_dict = {}
label_list = []
i=0
for x in set(true_label):
    label_dict[x] = i
    label_list.append(x)
    i+=1
```

--- type:NormalExercise lang:python xp:100 skills:1 key:c146087ba6
## Needs Exercise title also
Now use the dictionary to do the conversion for both the `true_label` and `pred_label` for the test set.

*** =instructions
- transform `pred_label` to a list of ints called `pred_int`
- subset only elements in the test set (i.e. not in the array `training_indices`), denote these `test_true_int` and `test_pred_int`

*** =hint
- There are many ways to do this, you can use python's list comprehension (i.e. something like this `[x for x in true_label]`)
- Use an `if not in training_indices` clause to subset out the training elements and keep the others

*** =sample_code
```{python}
# list of ints for true and predicted labels 
true_int = [label_dict[l] for l in true_label]
pred_int = ...

# list of ints for true and predicted only test set
test_true_int = ...
test_pred_int = ...
```

*** =solution
```{python}
# int classes for the true and predicted. 
true_int = [label_dict[l] for l in true_label]
pred_int = [label_dict[l] for l in pred_label]

# int classes for true and predicted, just the test set.
test_pred_int = [v for i, v in enumerate(pred_int) if i not in training_indices]
test_true_int = [v for i, v in enumerate(true_int) if i not in training_indices]
```


*** =sct
```{python}
# SCT written with pythonwhat: https://github.com/datacamp/pythonwhat/wiki
test_object("true_int",
            undefined_msg = "Don't remove the definition of the predefined `true_int` object.",
            incorrect_msg = "Don't change the definition of the predefined `true_int` object.")
test_object("pred_int",
            undefined_msg = "Don't remove the definition of the predefined `pred_int` object.",
            incorrect_msg = "Don't change the definition of the predefined `pred_int` object.")
test_object("test_pred_int",
            undefined_msg = "Don't remove the definition of the predefined `test_pred_int` object.",
            incorrect_msg = "Don't change the definition of the predefined `test_pred_int` object.")
test_object("test_true_int",
            undefined_msg = "Don't remove the definition of the predefined `test_true_int` object.",
            incorrect_msg = "Don't change the definition of the predefined `test_true_int` object.")

success_msg("Great work!")
```


Let's now see how this does in terms of accuracy on the test set. 
```{python}
cm = confusion_matrix(test_pred_int,test_true_int)
plot_confusion_matrix(cm,target_names=label_list)
```

--- type:MultipleChoiceExercise lang:python xp:50 skills:2 key:7f5cb08571
## Final question

Notice that this is TERRIBLE! Can you think of why? 

*** =instructions
- Do you think it's caused by downsampling?
- Only one training example for each class?
- Is DTW just a poor similarity measure?
- Is the norm a poor choice?

*** =hint
Think about how different two files with the same labels were, were they very similar or very different?


*** =sct
```{python}

msg1 = "Great job!"
msg2 = "Wrong, try again. Maybe have a look at the hint."
msg3 = "Not so good... Maybe have a look at the hint."
msg4 = "Incorrect. Maybe have a look at the hint."
test_mc(correct = 1, msgs = [msg1, msg2, msg3, msg4])

success_msg("Well done! Now move on and explore some of the features in more detail.")

```

The main reason is likely that we are using 1 known training example as our template. Downsampling to 200Hz, will have an impact, which you can check but even at the full rate, the accuracy will not improve vastly.

So, how would we improve this? If we do something more intelligent with a full kNN as the underlying algorithm and DTW as the distance metric. We do exactly that in the full analysis, whose details can be found: 
 slides_link : http://sflscientific.com/presentations-and-conference-talks/

The methodology of comparing time-sequences is identical to the above, except we use a k-Nearest Nieghbour algorithm:
<center><img src="https://raw.githubusercontent.com/mmluk/DTWTutorial/master/Images/dtw_knn_schematic.png"></center>

Where, the k nearest neighbours vote on the class label of the test elements. Indeed, with the same methodology but with a k=3 Nearest Neighbour algorithm, we end up with the following confusion matrix:
<center><img src="https://raw.githubusercontent.com/mmluk/DTWTutorial/master/Images/final_confmat.png"></center>

which is definitely very reasonable!

# Final Thoughts

Just with this minimal setup, we have already achieved very high accuracy without the need for feature engineering.  These methods can be applied in conjunction with any distance-based algorithms and have shown great success in time-series analyses in healthcare, sports, finance and more. 

Notice that whilst the methods we have shown are quite slow, the process above with finding neighbours lends itself nicely to distributed computation, with the usual caveats that come with kNN of memory allocation for the training data. Further, faster implementations of the DTW computation (in particular check out UCR's DTW suite and Python's fast DTW library, which runs an approximation of the DTW).

Finally, we can also attempt to ensemble these techniques with more-standard feature-generation methods to achieve even better results.
