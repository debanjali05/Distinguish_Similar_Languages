# Distinguish Similar Languages

Implementation of a method to distinguish between similar languages such as Croatian, Bosnian and Serbian. 
We implement an ensemble of SVM and Naive Bayes classifiers using a hard voting classifier usng character level n-gram (2-6) Tfidf Vectorizer.
The model predicts the class label based on the argmax of the sums of the predicted probabilities estimated from both the classifiers.


## Dataset
DSL Corpus Collection ([here](http://ttg.uni-saarland.de/resources/DSLCC/)) is used for this work. 
The training set consists of 18k lines for each language present in the dataset and the test set consists of 1000 lines for each language.

A subset of the DSLCC v4.0 for the group of similar languages: (Croatian - 'hr', Bosnian - 'bs' and ' Serbian - 'sr') is used for training and testing our model. 
The subset is generated using [this code](https://github.com/debanjali05/Distinguish_Similar_Languages/blob/master/Data/subset_script.py). 
This script generates 2 files: **train.txt** and **test.txt** (already generated and present in [Data](https://github.com/debanjali05/Distinguish_Similar_Languages/tree/master/Data) folder). 
The new training set and test set consists of 54k lines (18k * 3) and 3k (1000 * 3) respectively.

**Note:**
The original dataset is not present in this repository. To generate the files again, please download the DSLCC v4.0 dataset in the Data folder before running the script:
```bash
python Data/subset_script.py
```
Also, update the correct path to the generated dataset (subset of the original dataset) in [utils.py](https://github.com/debanjali05/Distinguish_Similar_Languages/blob/master/utils.py) before running the model. 

## Requirement
Python 3.6.2 and SciKit Learn 0.19.0

## Running
```bash
python language_detection
```
## Output
| Model   |   Accuracy        |  F1 score |
|----------|:-------------:|------:|
| Ensemble |  76.87 % | 0.7689 | 

The confusion matrix generated in this case:

![Confusion Matrix](https://github.com/debanjali05/Distinguish_Similar_Languages/blob/master/sample_output/confusion_matrix.png)

A sample output can be found in [output.txt](https://github.com/debanjali05/Distinguish_Similar_Languages/blob/master/sample_output/output.txt). 

