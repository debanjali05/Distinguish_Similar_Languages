# Distinguish Similar Languages

Implementation of a method to distinguish between similar languages such as Croatian, Bosnian and Serbian. 
We implement an ensemble of SVM and Naive Bayes classifiers using a hard voting classifier usng character level n-gram (2-6) Tfidf Vectorizer.
The model predicts the class label based on the argmax of the sums of the predicted probabilities estimated from both the classifiers.
