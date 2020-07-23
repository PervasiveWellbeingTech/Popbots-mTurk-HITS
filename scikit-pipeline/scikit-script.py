# SETUP
#
# Refs
# https://github.com/UKPLab/sentence-transformers
# https://towardsdatascience.com/nlp-extract-contextualized-word-embeddings-from-bert-keras-tf-67ef29f60a7b
# https://towardsdatascience.com/bert-for-dummies-step-by-step-tutorial-fb90890ffe03

# Standard includes
import csv
import pickle
import pandas as pd
import string
from tkinter import Tk, filedialog
# import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from pandas import DataFrame
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sn
from sentence_transformers import SentenceTransformer
from sklearn.metrics import precision_recall_fscore_support, classification_report

# Select File
root = Tk()
root.filename = filedialog.askopenfilename()
file = root.filename
root.withdraw()


# (NLTK) Helper Settings
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

# (NLTK) Helper Functions
def clean(doc):
    stop_free = " ".join([i for i in doc.split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized


# Settings & Variables
key_dict = {0: 'Social Relationships', 1: 'Health, Fatigue, or Physical Pain', 2: 'Emotional Turmoil', 3: 'Work', 
           4: 'Family Issues', 5: 'Everday Decision Making', 6: 'School', 7: 'Other', 8: 'Financial Problem'}
Input_File = file
output_model_filename = 'finalized_model.sav'
output_probs = "output_probs.csv"
Label = "Multi-class"
Features = "BERT"
Algorithm = "SVC"
Sweep = False

# Silly additional settings that need to reflect the number of classes in your dataset
# And, unfortunately you'll need to edit some of the display functions below as well
# TODO: Cleanup

# ----------------------------------------
# SCRIPT PROCESSING
# This is where the main processing takes place

# Read data from converted/compiled CSV (Assumes data is sorted by 'Set' column ascending)
# TODO: See how the TF_IDF features parse the list and use that here instead of relying on the ordering of 'Set'
df = pd.read_csv(Input_File)
TargetNamesStrings = [str(x) for x in df[Label].unique().tolist()]
TargetNames = df[Label].unique().tolist()
TargetNames.sort()

dataset = (df['Set'] == 0).sum()

class_report = open('scikit_report.txt', 'w')

# Preview the first 5 lines of the loaded data
print(df.head())
class_report.write(str(df.head()))
class_report.write('\n')

# Cast labels
df[Label] = df[Label].astype(int)

# Read each document and clean it.
df["Sentence"] = df["Sentence"].apply(clean)

# Let's do some quick counts
# TODO: Make this dynamic so we don't have to interact with the code here to change # of labels above
CategoryLabels = list(df[Label])
label_sum = float(len(CategoryLabels))


print(" ")
class_report.write(" \n")
print("===============")
class_report.write("===============\n")
print("Data Distribution:")
class_report.write("Data Distribution:\n")
for label in TargetNames:
    print(str(label) + ' ' + str(key_dict[label]) +  ' contains:', CategoryLabels.count(label), round(float(CategoryLabels.count(label) / label_sum), 2))
    class_report.write(str(label) + ' ' + str(key_dict[label]) +  ' contains:' + ' ' + str(CategoryLabels.count(label)) + ' ' + str(round(float(CategoryLabels.count(label) / label_sum), 2)))
    class_report.write('\n')

# Beginning to calculate features include BERT and TF-IDF; this process can be a bit of bottleneck
# TODO: Consider writing these variables to a file to "pre-compute" them if experiments are taking awhile
print(" ")
class_report.write(" \n")
print("===============")
class_report.write("===============\n")
print("Fitting Features: ")
class_report.write("Fitting Features: \n")
print(" ")
class_report.write('\n')
bert_dimension = 0
if Features == "All" or Features == "BERT":
    # Create BERT Features and add to data frame
    print('Fitting BERT Features')
    class_report.write('Fitting BERT Features')
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    sentences = df['Sentence'].tolist()
    sentence_embeddings = model.encode(sentences)
    encoded_values = pd.DataFrame(np.row_stack(sentence_embeddings))

    FeatureNames = []
    bert_dimension = encoded_values.shape[1]
    for x in range(0, bert_dimension):
        FeatureNames.append("BERT_" + str(x))

    training_corpus = encoded_values.head(dataset)
    test_corpus = encoded_values.tail((df['Set'] == 1).sum())

tf_dimension = 0
if Features == "All" or Features == "TF":
    # Create TF-IDF Features and add to data frame
    print('Fitting TF-IDF Features')
    tf_train, tf_test = df[df['Set'] != 1], df[df['Set'] == 1]
    tf_training_corpus = tf_train['Sentence'].values
    tf_training_labels = tf_train[Label].values
    tf_test_corpus = tf_test['Sentence'].values
    tf_test_labels = tf_test[Label].values

    tf_idf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=5000, stop_words='english')
    tfidf = tf_idf_vectorizer.fit_transform(tf_training_corpus)
    X = tf_idf_vectorizer.fit_transform(tf_training_corpus).todense()

    featurized_training_data = []
    for x in range(0, len(X)):
        tfid_Features = np.array(X[x][0]).reshape(-1, )
        featurized_training_data.append(tfid_Features)

    FeatureNames = []
    tf_dimension = X.shape[1]
    for x in range(0, tf_dimension):
        FeatureNames.append("TFIDF_" + str(x))

    X = tf_idf_vectorizer.transform(tf_test_corpus).todense()
    featurized_test_data = []
    for x in range(0, len(X)):
        tfid_Features = np.array(X[x][0]).reshape(-1, )
        featurized_test_data.append(tfid_Features)

# Merge the feature data if 'All' or get the TF-IDF Features if 'TF'
if Features == 'All':
    featurized_training_data_df = DataFrame(featurized_training_data, columns=FeatureNames)
    training_corpus = pd.concat([training_corpus, featurized_training_data_df], axis=1)

    test_corpus = test_corpus.reset_index()
    test_corpus = test_corpus.drop(['index'], axis=1)
    featurized_test_data_df = DataFrame(featurized_test_data, columns=FeatureNames)
    test_corpus = pd.concat([test_corpus, featurized_test_data_df], axis=1)

elif Features == 'TF':
    featurized_training_data_df = DataFrame(featurized_training_data, columns=FeatureNames)
    training_corpus = featurized_training_data_df
    featurized_test_data_df = DataFrame(featurized_test_data, columns=FeatureNames)
    test_corpus = featurized_test_data_df

# Get the labels from the original data frame
temp1 = df.head(dataset)
temp2 = df.tail((df['Set'] == 1).sum())
training_labels = temp1[Label].values
test_labels = temp2[Label].values
training_labels = training_labels.astype(int)
test_labels = test_labels.astype(int)

# Create final dataset for Testing & Training by joining Labels
train = pd.DataFrame(training_corpus)
test = pd.DataFrame(test_corpus)
train[Label] = pd.Categorical.from_codes(training_labels, TargetNames)
test[Label] = pd.Categorical.from_codes(test_labels, TargetNames)

# Show the number of observations for the test and training data frames
print(" ")
class_report.write('\n')
print("===============")
class_report.write("===============\n")
print("Fold Information: ")
class_report.write("Fold Information: \n")
print('Number of observations in the training data:', len(train))
class_report.write('Number of observations in the training data: ' +  str(len(train)) + '\n')
print('Number of observations in the test data:', len(test))
class_report.write('Number of observations in the test data: ' +  str(len(test)) + '\n')
print('Number of features generated:', str(tf_dimension + bert_dimension))
class_report.write('Number of features generated: ' +  str(tf_dimension + bert_dimension) + '\n')

# Create a list of the feature column's names
features = train.columns[:(tf_dimension + bert_dimension)]

# Create a classifier. By convention, clf means 'classifier'
if Algorithm == "SVC":
    clf = SVC(kernel='rbf', class_weight='balanced', probability=True)
if Algorithm == "SVC-Sweep":
    clf = SVC(kernel='poly', class_weight='balanced', C=1, decision_function_shape='ovo', gamma=0.0001,
              probability=True)
if Algorithm == "LSVC":
    clf = svm.LinearSVC()
if Algorithm == "RF":
    clf = RandomForestClassifier(n_jobs=-1, class_weight="balanced")
if Algorithm == "GBT":
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
if Algorithm == "VT":
    clf1 = SVC(kernel='rbf', class_weight="balanced", probability=True)
    clf2 = RandomForestClassifier(n_jobs=-1, class_weight="balanced")
    clf3 = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    clf = VotingClassifier(estimators=[('svc', clf1), ('rf', clf2), ('gbt', clf3)], voting='soft', weights=[1, 1, 1])

# Train the classifier to take the training features and learn how they relate
clf.fit(train[features], train[Label])

# Apply the classifier we trained to the test data (which, remember, it has never seen before)
preds = clf.predict(test[features])
if Algorithm == "SVC" or Algorithm == "SVC-Sweep":
    # Output the probabilities for the SVC, it's possible this could be extended to other alogirthms
    # TODO: Investigate
    # Below this is some legacy code which will allow you to filter the output and see it reflected
    # in the stats below by swapping in y_pred but this can have a whacky interaction with other classifiers
    preds_proba = clf.predict_proba(test[features])
    y_pred = (clf.predict_proba(test[features])[:, 1] >= 0.695).astype(bool)

# View the PREDICTED classes for the first five observations
print(" ")
class_report.write('\n')
print("===============")
class_report.write("===============\n")
print("Example Prediction: ")
class_report.write("Example Prediction: \n")
print(preds[0:5])
class_report.write(str(preds[0:5]))
if Algorithm == "SVC" or Algorithm == "SVC-Sweep":
    with open(output_probs, 'w', newline='') as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerows(preds_proba)

# View the ACTUAL classes for the first five observations
print(" ")
class_report.write('\n')
print("===============")
class_report.write("===============\n")
print("Actual: ")
class_report.write("Actual: \n")
print(str(test[Label].head()))
class_report.write(str(test[Label].head()) + '\n')

# Create confusion matrix
print(" ")
class_report.write('\n')
print("===============")
class_report.write("===============\n")
print("Confusion Matrix: ")
class_report.write("Confusion Matrix: \n")
print(" ")
class_report.write('\n')
confusion_matrix = pd.crosstab(test[Label], preds, rownames=['Actual Categories'], colnames=['Predicted Categories'])
print(str(pd.crosstab(test[Label], preds, rownames=['Actual Categories'], colnames=['Predicted Categories'])))
class_report.write(str(pd.crosstab(test[Label], preds, rownames=['Actual Categories'], colnames=['Predicted Categories'])))

# Show confusion matrix in a separate window
sn.set(font_scale=1.4)  # for label size
g = sn.heatmap(confusion_matrix, annot=True, annot_kws={"size": 12}, cmap="YlGnBu", cbar=False)  # font size
bottom, top = g.get_ylim()
g.set_ylim(bottom + 0.5, top - 0.5)
plt.show()

# Precion, Recall, F1
print(" ")
print("===============")
print("Classification Report: ")
print(" ")
class_report.write('\n')
class_report.write("===========\n")
class_report.write("Classification Report: \n")
class_report.write(" \n")
print("Precision, Recall, Fbeta Stats: ")
print('Macro:  ', precision_recall_fscore_support(test[Label], preds, average='macro'))
print('Micro:  ', precision_recall_fscore_support(test[Label], preds, average='micro'))
print('Weighted', precision_recall_fscore_support(test[Label], preds, average='weighted'))
print(" ")
class_report.write("Precision, Recall, Fbeta Stats: \n")
class_report.write('Macro:  ' + str(precision_recall_fscore_support(test[Label], preds, average='macro')) + '\n')
class_report.write('Micro:  ' + str(precision_recall_fscore_support(test[Label], preds, average='micro')) + '\n')
class_report.write('Weighted' + str(precision_recall_fscore_support(test[Label], preds, average='weighted')) + '\n')
class_report.write(" \n")


target_names_int = [int(name) for name in TargetNamesStrings]
target_names_int.sort()
TargetNamesStrings = [str(key_dict[x]) + '-' + str(x) for x in target_names_int]
print(classification_report(test[Label], preds, target_names=TargetNamesStrings))
class_report.write(str(classification_report(test[Label], preds, target_names=TargetNamesStrings)))



# Generate PR Curve (if doing a binary classification)
if (Algorithm == "SVC" or Algorithm == "SVC-Sweep") and (Label != "Multi-class" and Label != "Original-Multi-Class"):
    y_score = clf.decision_function(test[features])
    average_precision = average_precision_score(test[Label], y_score)
    print('Average precision-recall score: {0:0.2f}'.format(average_precision))
    class_report.write('Average precision-recall score: {0:0.2f}'.format(average_precision) + '\n')
    disp = plot_precision_recall_curve(clf, test[features], test[Label])
#     TODO: Bug below
    disp.ax_.set_title('2-class Precision-Recall curve: ', 'AP={0:0.2f}'.format(average_precision))
    plt.show()

# save the model to disk
pickle.dump(clf, open(output_model_filename, 'wb'))

# parameter sweep
if Sweep and Algorithm == "SVC":
    param_grid = [
        {'C': [1, 10, 100, 1000], 'kernel': ['linear'], 'decision_function_shape': ['ovo', 'ovr']},
        {'C': [1, 10, 100, 1000], 'gamma': ['scale', 'auto', 0.001, 0.0001], 'kernel': ['rbf'],
         'decision_function_shape': ['ovo', 'ovr']},
        {'C': [1, 10, 100, 1000], 'gamma': ['scale', 'auto', 0.001, 0.0001], 'kernel': ['poly'],
         'decision_function_shape': ['ovo', 'ovr']},
        {'C': [1, 10, 100, 1000], 'gamma': ['scale', 'auto', 0.001, 0.0001], 'kernel': ['sigmoid'],
         'decision_function_shape': ['ovo', 'ovr']}
    ]

    print("")
    class_report.write('\n')
    print("Starting GridSearch; this could take some time...")
    search = GridSearchCV(clf, param_grid, cv=5, n_jobs=-1).fit(train[features], train[Label])
    print(search.best_params_)
    print(search.best_score_)
    print(search.best_estimator_)
    class_report.write(str(search.best_params_))
    class_report.write(str(search.best_score_))
    class_report.write(str(search.best_estimator_))

class_report.close()
exit()
