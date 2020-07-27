# Copyright 2019 Google Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#Adapted by Thierry Lincoln in November,2019 from this Colab notebook:
#https://colab.research.google.com/github/google-research/bert/blob/master/predicting_movie_reviews_with_bert_on_tf_hub.ipynb.
#Changes includes 

# - Reading our stressor data and parsing it properly
# - reconfiguring the last layer to include N neurons corresponding to N categories
# - correcting the probability output so that it follows [0,1] proper pattern 
# - better analysis with confusion matrix
# - exporting to pb format for tensorflow serving api
import os

os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-10.0/lib64'
import sys
print(sys.executable)

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime

import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import f1_score,confusion_matrix,classification_report,accuracy_score

import logging
logging.basicConfig(stream=sys.stdout, level=logging.ERROR)



pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 1000)

config = tf.ConfigProto()
#config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
#config.gpu_options.visible_device_list="0"



from tensorflow.python.client import device_lib

device_lib.list_local_devices()

import bert
from bert import run_classifier_with_tfhub
from bert import optimization
from bert import tokenization
from bert import modeling 
import numpy as np


############ Utils functions ##################

def create_examples_prediction(df):
    """Creates examples for the training and dev sets."""
    examples = []
    for index, row in df.iterrows():
        
        #labels = row[LABEL_HOT_VECTOR].strip('][').split(', ')
        #labels = [float(x) for x in labels]
        labels = list(row[label_list_text])
        examples.append(labels)
        
    return pd.DataFrame(examples)

def f(x):
    n = 2  # index of the second proability to get labeled 

    index = np.argsort(x.values.flatten().tolist())[-n:][0]
    print(f"index is {index}")
    label  = label_list_text[index]
    print(f"label is {label}")
    
    return label

final_columns = ["sOrder","Input.text","is_stressor","is_stressor_conf","top_label","second_label","Branch", "Above SD-THRESHOLD","SD-THRESHOLD","SD","Other","Everyday Decision Making","Work","Social Relationships","Financial Problem","Health, Fatigue, or Physical Pain","Emotional Turmoil","Family Issues","School","avg_severity","median_severity","SD_severity","Votes","Source"]

def get_test_experiment_df(test):
    test_predictions = [x[0]['probabilities'] for x in zip(getListPrediction(in_sentences=list(test[DATA_COLUMN])))]
    test_live_labels = np.array(test_predictions).argmax(axis=1)
    test[LABEL_COLUMN_RAW] = [label_list_text[x] for x in test_live_labels] # appending the labels to the dataframe
    
    probabilities_df_live = pd.DataFrame(test_predictions) # creating a proabilities dataset
    probabilities_df_live.columns = [x for x in label_list_text] # naming the columns
    probabilities_df_live['second_label'] = probabilities_df_live.apply(lambda x:f(x),axis=1)
    
    #print(test)
    #label_df = create_examples_prediction(test)
    #label_df.columns = label_list_text
    #label_df['label 2'] = label_df.apply(lambda x:f(x),axis=1)

    test.reset_index(inplace=True,drop=True) # resetting index 
    
    test_removed_columns =  list(set(test.columns)-set(probabilities_df_live.columns))
    
    test_temp = test[test_removed_columns]
    
    experiment_df = pd.concat([test_temp,probabilities_df_live],axis=1, ignore_index=False)
    
    
    missing_cols = list(set(experiment_df.columns)-set(final_columns))
    experiment_df[missing_cols] = np.nan#.loc[:, missing_cols] = np.nan
        
    experiment_df = experiment_df.reindex(columns = final_columns)

    
    #experiment_df = experiment_df.reindex(sorted(experiment_df.columns), axis=1)
    
    return test,experiment_df

def getListPrediction(in_sentences):
    #1
    input_examples = [bert.run_classifier.InputExample(guid="", text_a = x, text_b = None, label = 0) for x in in_sentences] # here, "" is just a dummy label
    
    #2
    input_features = bert.run_classifier.convert_examples_to_features(input_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
    
    #3
    predict_input_fn = bert.run_classifier.input_fn_builder(features=input_features, seq_length=MAX_SEQ_LENGTH, is_training=False, drop_remainder=False)
    
    print(input_features[0].input_ids)
    #4
    predictions = estimator.predict(input_fn=predict_input_fn,yield_single_examples=True)
    
    return predictions

is_normalize_active=False

def get_confusion_matrix(y_test,predicted,labels):
    class_names=labels
    # plotting confusion matrix
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plot_confusion_matrix(y_test, predicted, classes=class_names,
                        title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plot_confusion_matrix(y_test, predicted, classes=class_names, normalize=True,
                        title='Normalized confusion matrix')
    plt.show()
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes =classes
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        test =1
        #print('Confusion matrix, without normalization')

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    #ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    #fig.tight_layout()
    return ax

####### Loading the data ##### 

def data_prep_bert(df,test_size):
    
    #print("Filling missing values")
    #df[DATA_COLUMN] = df[DATA_COLUMN].fillna('_NA_')
    
    print("Splitting dataframe with shape {} into training and test datasets".format(df.shape))
    X_train, X_test  = train_test_split(df, test_size=test_size, random_state=2018,stratify = df[LABEL_COLUMN_RAW])

    return X_train, X_test


def open_dataset(NAME,mapping_index,excluded_categories):
    df = pd.read_csv(PATH+NAME+'.csv',sep =',')
    df.head(10)
    df = df[df[LABEL_COLUMN_RAW].notna()]
    
    
    
    #df.columns = [LABEL_COLUMN_RAW,'Severity',DATA_COLUMN,'Source']
    
    if excluded_categories is not None:
        for category in excluded_categories:

            df = df[df[LABEL_COLUMN_RAW] !=category]

    label_list=[]
    label_list_final =[]
    if(mapping_index is None):
        df[LABEL_COLUMN_RAW] = df[LABEL_COLUMN_RAW].astype('category')
        df[LABEL_COLUMN], mapping_index = pd.Series(df[LABEL_COLUMN_RAW]).factorize() #uses pandas factorize() to convert to numerical index
        
        
        
        label_list_final = [None] * len(mapping_index.categories)
        label_list_number = [None] * len(mapping_index.categories)
        
        for index,ele in enumerate(list(mapping_index.categories)):
            lindex = mapping_index.get_loc(ele)
            label_list_number[lindex] = lindex
            label_list_final[lindex] = ele
    else:
        df[LABEL_COLUMN] = df[LABEL_COLUMN_RAW].apply(lambda x: mapping_index.get_loc(x))
    
    frequency_dict = df[LABEL_COLUMN_RAW].value_counts().to_dict()
    df["class_freq"] = df[LABEL_COLUMN_RAW].apply(lambda x: frequency_dict[x])
    
    
    return df,mapping_index,label_list_number,label_list_final
    


# Require user changes > Start Here 

### Experiment Name

PATH = './datasets/'
TODAY_DATE = "27_07_2020/"
EXPERIMENT_NAME = 'single_label'
EXPERIMENTS_PATH = PATH + 'experiments/'+TODAY_DATE+EXPERIMENT_NAME


if not os.path.exists(PATH + 'experiments/'+TODAY_DATE):
    os.mkdir(PATH + 'experiments/'+TODAY_DATE)
if not os.path.exists(EXPERIMENTS_PATH):
    os.mkdir(EXPERIMENTS_PATH)

### Model Hyperparameters

# Compute train and warmup steps from batch size
# These hyperparameters are copied from this colab notebook (https://colab.sandbox.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 3.0
# Warmup is a period of time where hte learning rate 
# is small and gradually increases--usually helps training.
WARMUP_PROPORTION = 0.1
# Model configs
SAVE_CHECKPOINTS_STEPS = 1000
SAVE_SUMMARY_STEPS = 100

# We'll set sequences to be at most 32 tokens long.
MAX_SEQ_LENGTH = 32


OUTPUT_DIR = './models/'+ TODAY_DATE+EXPERIMENT_NAME+'/'



DATASET_NAME = '2020-06-20-MainTurkAggregation-5-Turkers_v0'

DATA_COLUMN = 'Input.text'
LABEL_COLUMN_RAW = 'top_label'#'Answer.Label'

LABEL_COLUMN = 'label_numeric'
MTURK_NAME = 'mTurk_synthetic'
LIVE_NAME = 'popbots_live'
INQUIRE_NAME = 'Inquire'
MTURK_COVID_NAME = 'mTurk_synthetic_covid'


#dataset,mapping_index,label_list, label_list_text = open_dataset('mturk900balanced',None)

EXCLUDED_CATEGORIES = None #['Other'] #None # # if nothing to exclude put None, THIS ALWAYS MUST BE A LIST 

dataset,mapping_index,label_list, label_list_text = open_dataset(DATASET_NAME,None,EXCLUDED_CATEGORIES)





# This is a path to an uncased (all lowercase) version of BERT
BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"#"https://tfhub.dev/digitalepidemiologylab/covid-twitter-bert/1"#

def create_tokenizer_from_hub_module():
  """Get the vocab file and casing info from the Hub module."""
  with tf.Graph().as_default():
    bert_module = hub.Module(BERT_MODEL_HUB)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    with tf.Session() as sess:
      vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                            tokenization_info["do_lower_case"]])
      
  return bert.tokenization.FullTokenizer(
      vocab_file=vocab_file, do_lower_case=do_lower_case)

tokenizer = create_tokenizer_from_hub_module()



# Creating a model



def create_model(is_predicting, input_ids, input_mask, segment_ids, labels,
                 num_labels):
  """Creates a classification model."""

  bert_module = hub.Module(
      BERT_MODEL_HUB,
      trainable=True) # fined tuning the complete weights of all the model
    
  bert_inputs = dict(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids)
  bert_outputs = bert_module(
      inputs=bert_inputs,
      signature="tokens",
      as_dict=True)

  # Use "pooled_output" for classification tasks on an entire sentence.
  # Use "sequence_outputs" for token-level output.
  output_layer = bert_outputs["pooled_output"] # 768 dimention vector

  hidden_size = output_layer.shape[-1].value

  # Create our own layer to tune for politeness data.
  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):

    # Dropout helps prevent overfitting
    output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
    
    # does the Ax multiplication
    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    # add the bias eg: Ax+B
    logits = tf.nn.bias_add(logits, output_bias)
    
    
    ########################### HERE ADDITIONAL LAYERS CAN BE ADDED ######################
    
    # compute the log softmax for each neurons/logit
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    
    #compute the normal softmax to get the probabilities
    probs = tf.nn.softmax(logits, axis=-1)
    
    # Convert labels into one-hot encoding 
    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
    
    #classes_weights = tf.constant([1.0,1.0,1.0,1.0,1.0,1.0,0.7], dtype=tf.float32)
    #sample_weights = tf.multiply(one_hot_labels, classes_weights)
    
    
    predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))
    # If we're predicting, we want predicted labels and the probabiltiies.
    if is_predicting:
      return (predicted_labels, log_probs,probs)

    # If we're train/eval, compute loss between predicted and actual label
    #per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    
    loss = tf.reduce_mean(per_example_loss)
    return (loss, predicted_labels, log_probs)



# model_fn_builder actually creates our model function
# using the passed parameters for num_labels, learning_rate, etc.
def model_fn_builder(num_labels, learning_rate, num_train_steps,
                     num_warmup_steps):
  """Returns `model_fn` closure for TPUEstimator."""
  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]

    is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)
    
    # TRAIN and EVAL
    if not is_predicting:

      (loss, predicted_labels, log_probs) = create_model(
        is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

      train_op = bert.optimization.create_optimizer(
          loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)

      # Calculate evaluation metrics. 
      def metric_fn(label_ids, predicted_labels):
        accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
        """
        f1_score = tf.contrib.metrics.f1_score(
            label_ids,
            predicted_labels)
        
        auc = tf.metrics.auc(
            label_ids,
            predicted_labels)"""
        recall = tf.metrics.recall(
            label_ids,
            predicted_labels)
        precision = tf.metrics.precision(
            label_ids,
            predicted_labels) 
        true_pos = tf.metrics.true_positives(
            label_ids,
            predicted_labels)
        true_neg = tf.metrics.true_negatives(
            label_ids,
            predicted_labels)   
        false_pos = tf.metrics.false_positives(
            label_ids,
            predicted_labels)  
        false_neg = tf.metrics.false_negatives(
            label_ids,
            predicted_labels)
        return {
            "eval_accuracy": accuracy,
            #"f1_score": f1_score,
            #"auc": auc,
            "precision": precision,
            "recall": recall,
            "true_positives": true_pos,
            "true_negatives": true_neg,
            "false_positives": false_pos,
            "false_negatives": false_neg
        }

      eval_metrics = metric_fn(label_ids, predicted_labels)

      if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode=mode,
          loss=loss,
          train_op=train_op)
      else:
          return tf.estimator.EstimatorSpec(mode=mode,
            loss=loss,
            eval_metric_ops=eval_metrics)
    else:
      (predicted_labels, log_probs,probs) = create_model(
        is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

      predictions = {
          'probabilities': probs#,
          #'labels': predicted_labels
      }
      return tf.estimator.EstimatorSpec(mode, predictions=predictions)

  # Return the actual model function in the closure
  return model_fn







def train_evaluate(train, test):

    # Use the InputExample class from BERT's run_classifier code to create examples from the data
    train_InputExamples = train.apply(lambda x: bert.run_classifier.InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this example
                                                                    text_a = x[DATA_COLUMN], 
                                                                    text_b = None, 
                                                                    label = x[LABEL_COLUMN]), axis = 1)

    test_InputExamples = test.apply(lambda x: bert.run_classifier.InputExample(guid=None, 
                                                                    text_a = x[DATA_COLUMN], 
                                                                    text_b = None, 
                                                                    label = x[LABEL_COLUMN]), axis = 1)
    # Convert our train and test features to InputFeatures that BERT understands.
    train_features = bert.run_classifier.convert_examples_to_features(train_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)
    test_features = bert.run_classifier.convert_examples_to_features(test_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)



    # Compute # train and warmup steps from batch size
    num_train_steps = int(len(train_features) / BATCH_SIZE * NUM_TRAIN_EPOCHS)
    num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

    # Specify outpit directory and number of checkpoint steps to save

    run_config = tf.estimator.RunConfig(
        model_dir=OUTPUT_DIR,
        save_summary_steps=SAVE_SUMMARY_STEPS,
        save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)

    model_fn = model_fn_builder(
    num_labels=len(label_list),
    learning_rate=LEARNING_RATE,
    num_train_steps=num_train_steps,
    num_warmup_steps=num_warmup_steps)

    estimator = tf.estimator.Estimator(
    model_fn=model_fn,
    config=run_config,
    params={"batch_size": BATCH_SIZE})



    # Create an input function for training. drop_remainder = True for using TPUs.
    train_input_fn = bert.run_classifier.input_fn_builder(
        features=train_features,
        seq_length=MAX_SEQ_LENGTH,
        is_training=True,
        drop_remainder=False)


    try:
        import shutil
        shutil.rmtree(OUTPUT_DIR) #removes the model
    except:
        print('Failed to remove')
        pass

    print(f'Beginning Training!')
    current_time = datetime.now()
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
    print("Training took time ", datetime.now() - current_time)

    # Evaluating the model on Test Set

    test_input_fn = bert.run_classifier.input_fn_builder(
        features=test_features,
        seq_length=MAX_SEQ_LENGTH,
        is_training=False,
        drop_remainder=False)

    info=estimator.evaluate(input_fn=test_input_fn, steps=None)

    #fetching all the probabilities for each line of the test set
    test_probabilities = [x[0]['probabilities'] for x in zip(estimator.predict(test_input_fn,yield_single_examples=True))]

    #taking the argmex for the highest category
    test_final_labels = np.array(test_probabilities).argmax(axis=1)

    ### Classification Report

    report = pd.DataFrame(classification_report(list(test[LABEL_COLUMN]),list(test_final_labels),zero_division=0, output_dict=True)).T

    print(report)
    return info,report



np.set_printoptions(suppress=True)
boostrap_nb = 2
TEST_PERCENTAGE = 0.2

eval_info = []
eval_classification_report = []

for i in range(boostrap_nb):

    train,test = data_prep_bert(dataset,TEST_PERCENTAGE)

    info,report = train_evaluate(train, test)

    eval_info.append(info)
    eval_classification_report.append(report.to_numpy().tolist())


with open(EXPERIMENTS_PATH+"/eval_info.txt", "w") as output:
    output.write(str(eval_info))

with open(EXPERIMENTS_PATH+"/eval_classifier_report.txt", "w") as output:
    output.write(str(eval_classification_report))
