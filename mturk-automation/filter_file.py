"""
*** MUST READ ***

The purpose of this script is to take in a dataframe after being processed through the M-Turk scripts,
and return a dataframe that is filtered with the desired compenents of the user.

This function is to be ran from the command line.
Download this file, open your terminal and navigate to the directory it is saved in
and then run the following command:
'python filter_file' for windows
'python3 filter_file' for mac

You should be prompted with a file explorer window to select the dataframe that you want to fitler,
and then you will be prompted with another window requesting you to uncheck some boxes to get the desired dataframe
"""

from tkinter import *
from tkinter import Tk, filedialog
import pandas as pd
import random


root = Tk()
root.geometry('1000x600')
root.title('DataFrame Filter Tool')

# Setting up the canvas to project the window on to create scrollbar feature
canvas = Canvas(root)
scroll_y = Scrollbar(root, orient="vertical", command=canvas.yview)
frame = Frame(canvas)


def create_test_set(fil_df, percent, source):
    """
    This Function is for creating a big list of all the sentences that have been randomly selected
    to be apart of the test set for the classifier across all the labels. So it grabs about 20% of the sentences
    from each label and adds them to a list. This list is returned from this function.
    """
    if source == 1:
        source = True
    # Getting all the unique labels from the dataframe
    labels = fil_df['top_label'].unique().tolist()

    # Creating the test set list
    test_set = []  # create big list that has all the sentences that are going to make the test set


    for label in labels:
        sentence_list = fil_df[fil_df['top_label'] == label]['Input.text'].tolist() # list of all sentences for that label

        # The case if you want to distribute the test set amount through each source
        if source:
            unique_source = fil_df['Source'].unique().tolist()
            for sourced in unique_source:
                df = fil_df[fil_df['Source'] == sourced]
                source_sent_list = df[df['top_label'] == label]['Input.text'].tolist()


                count = len(source_sent_list)

                # In the case there are no sentences for that source
                if count == 0:
                    continue

                test_amount = count * percent

                # randomly choose sentences from sentence list and append to test set list
                added = 0
                while added <= test_amount:
                    index = random.randint(0, count - 1)
                    test_set.append(sentence_list[index])
                    del sentence_list[index]
                    added += 1
                    count -= 1

        # If test set amount is not to be distributed through each source
        else:


            count = len(sentence_list)
            test_amount = count * percent
            added = 0

            # randomly pick sentences from the sentence list to add the the test set
            while added <= test_amount:
                index = random.randint(0, count - 1)
                test_set.append(sentence_list[index])
                del sentence_list[index]
                added += 1
                count -= 1

    # returning the big lsit of sentences that are going to be marked to be apart of the test set
    return test_set

def create_set_list(test_set, fil_df):
    """
    This function creates the list that is going to be appended to the dataframe to determine which specific sentences
    in the data frame are a part of the testing and training sets. When this list is added to the dataframe as a column
    called 'set' it is going to have a 1 in the row where that sentence in that row is a part of the test set
    and a 0 in the row where the sentence in that row is a part of the training set. It returns the filtered dataframe
    with this list appended on to it with the column label 'set' .
    """

    # Create a list of 0's
    set_list = []
    sentences = fil_df['Input.text'].tolist()
    for i in range(len(fil_df['top_label'])):
        set_list.append(0)

    # put a 1 in the row where a testing sentence is
    for sentence in test_set:
        index = sentences.index(sentence)
        set_list[index] = 1

    # add the testing and training list to the dataframe
    fil_df['Set'] = set_list

    return fil_df

def add_label_code(new_df):
    """
    This function creates a list where it represents the labels in integers instead of strings.
    The function appends this list to the dataframe as a column named, 'class'. This function returns the filtered
    dataframe with the 'class' list column appended to it. The key to make sense of these coded labels
    will be printed to the terminal
    """

    # Get the list of unique labels to create numerical key for
    labels = new_df['top_label'].unique().tolist()

    # create a dictionary that holds a value for each label
    key_dict = {'Social Relationships': 0, 'Health, Fatigue, or Physical Pain': 1, 'Emotional Turmoil': 2, 'Work': 3,
                'Family Issues': 4, 'Everyday Decision Making': 5, 'School': 6, 'Other': 7, 'Financial Problem': 8}
    # for i in range(len(labels)):
    #     key_dict[labels[i]] = i
    # print(key_dict)

    # create a list that is going to represent each label as the value from the dictionary
    Class = []
    og_class = []
    for label in new_df['top_label']:
        Class.append(key_dict[label])
    for label in new_df['original_label']:
        og_class.append(key_dict[label])
    # add the column to the dataframe
    new_df['Multi-class'] = Class
    new_df['Original-Multi-Class'] = og_class

    return new_df, key_dict

def add_binary_columns(fil_df):
    """
    This function adds several columns to the dataframe. For each unique label present in the 'top_label' column
    of the dataframe, there is a column added to the dataframe that has a '1' in the row where the label in the 'top_label'
    column of the label for that column. For example, for the 'work_binary' column, the column that was generated
    by this function because the label 'Work' was present in the 'top_label' column, this column would have a 1
    at each row where the label in the same row under the 'top_label' column was also 'Work' and would have 0's
    everywhere else. This function returns the filtered dataframe with all of these columns for each unique label
    appened to it.
    """

    # Create a list of the unique labels
    labels = fil_df['top_label'].unique().tolist()

    # Create a list that marks each index that label is present with a 1
    for label in labels:
        temp_list = [0 for x in range(len(fil_df['Input.text']))]
        for i in range(len(fil_df['top_label'])):
            top = fil_df['top_label'].tolist()[i]
            if top == label:
                temp_list[i] = 1

        # Add the list to the dataframe
        fil_df[str(label) + '_' + 'binary'] = temp_list

    return fil_df


def file_selection():
    """
    This function creates the file selection GUI with the user to get the dataframe that is in needed of filtering.
    It returns a dataframe with all of the duplicate sentences removed and the values in the dataframe scrambled.
    """

    # Creating instance of the file selection GUI
    root.filename = filedialog.askopenfilename()
    file = root.filename
    df = pd.read_csv(file)

    # drop the duplicate sentences
    df.drop_duplicates(subset='Input.text', keep='first', inplace=True)

    return df

def mix_df(df, shuff_num):
    # Don't scramble
    if shuff_num == 0:
        return df

    # mix the df by a random sequence of numbers
    for i in range(int(shuff_num)):
        random_list = random.sample(range(len(df['Input.text'])), len(df['Input.text']))
        df['scramble_seq'] = random_list  # add the list to the dataframe
        df = df.sort_values(by='scramble_seq')

    # drop the random number sequence column that scrambles
    df.drop(columns=['scramble_seq'], inplace=True)  # drop the column used to scramble data
    df.reset_index(drop=True, inplace=True)  # reset the index

    return df

def save_csv(df):
    """
    This function takes in the final dataframe after all the filtering and column appending
    and saves it to the directory where you have this code saved.
    """

    # rename sentence column to 'Sentence' for clarity
    df.rename(columns={'Input.text': 'Sentence'}, inplace=True)

    # Sort by 'Set' ascending
    df.sort_values(by='Set', ascending=True, inplace=True)

    # Save df to working directory
    pd.DataFrame(df).to_csv("Filtered_df.csv", index=False, header=True)

    # save dataframe settings




def filter(df, dict, sliders):
    """
    This function does the filtering of the dataframe. It returns the dataframe with the items
    that the user chose to keep.
    """

    # reducing the sample sentence size to what was indicated by the sliders


    # loop through each value of the checkbutton
    for key in dict:
        for value in dict[key]:
            if dict[key][value].get() == 0:
                if value == 'is_stressor' or value == 'is_covid':
                    value = int(value)

                # keep everything in df except for the checkbuttons that were unselected
                df = df[df[key] != value]

    for label in sliders:
        count = int(sliders[label].get())
        sentences = df[df['top_label'] == label]['Input.text'].tolist()
        if len(sentences) < int(count):
            continue

        for sentence in sentences[int(count):]:
            df = df[df['Input.text'] != sentence]

    return df


def make_checkbox(column, name, dict, row, col):
    """
    This function creates the checkbutton for each unique value in a column of the dataframe.
    """

    # Creates checknbutton on GUI display
    Checkbutton(frame, text=str(name), variable=dict[column][name]).grid(row=row, column=col, sticky=W)

def delete(test, shuff):
    """
    This function aids in the closing of the tkinter window.
    """
    canvas.delete("all")
    root.quit()
    return test.get(), shuff.get()


def options():
    """
    This function asks the user for the options they want considered before the dataframe is filtered.
    Such as the number of times to shuffle the dataset, the distribution of the test set, and whether
    or not to distribute through each Source.
    """

    # initializing variables
    test_percent = DoubleVar()
    test_percent.set(0.2) # sets a default variable
    shuff_num = DoubleVar()
    shuff_num.set(4)
    og_label = IntVar()


    # creating interactive widgets
    Label(root, bg='#b3ffb3', text='Select Size of Test Set', font=("TimesNewRoman", 8)).pack()
    Scale(root, bg='white',from_=0, to=1, resolution = 0.05, length=1000, orient=HORIZONTAL,
          variable=test_percent, width=10, font=("TimesNewRoman", 8)).pack()
    Label(root, bg='#b3ffb3',text='Select Number of times to mix DataFrame', font=("TimesNewRoman", 8)).pack()
    Scale(root, bg='white',from_=0, to=10, resolution=1, length=1000, orient=HORIZONTAL,
          variable=shuff_num, width=10, font=("TimesNewRoman", 8)).pack()
    # Checkbutton(root, bg='white', text='Use top_label or original_label?', variable=og_label).pack()


    return test_percent, shuff_num#, og_label

def check_box(df):
    """
    This function creates the display for where the checkbuttons will appear and grabs the unique values
    for each of the columns to create a checkbutton for. This function returns a dictionary
    that holds all the values for the checkbuttons.
    """

    Label(frame, text="Deselect Items you don't want in Data Frame", font='Helvetica 10 bold').grid(row=0, column=1, sticky=E) # message at top of display

    labels = df['top_label'].unique().tolist()
    grand_dict = {} # initialize dictionary
    columns = df.columns.tolist() # get list of unique columns

    # selected most important columns for now
    for selector in columns[2:]:
        if 'severity' in selector or (selector in labels) or ('Votes' in selector) or ('second' in selector) or \
                ('ID' in selector) or ('Answer' in selector) or ('origi' in selector):
            continue

        grand_dict[selector] = {}

    # variables to track grid pattern on GUI
    col = 1
    row = 1
    loop = 0

    for name in grand_dict:
        list_items = df[name].unique().tolist()

        # handling special case for appearence on GUI
        if name == 'is_seed':
            row += 3

        # display name of each column in GUI
        Label(frame, bg='white', text=str(name)).grid(row=row, column=col, sticky=W)
        if loop % 2 == 0:
            last_label = row
        last = col
        row += 1

        # Creating all the checkboxes to be displayed on the Checkbutton GUI
        for item in list_items:
            grand_dict[name][item] = IntVar(value=1)
            make_checkbox(name, item, grand_dict, row, col)
            row += 1

        # updating variables to change grid location on GUI
        loop += 1
        if loop % 2 != 0:
            row = last_label
        if last == 2:
            col = 1
        if last == 1:
            col = 2


    # Display Source Button at Bottom
    source = IntVar()
    Checkbutton(frame, bg='white', text='Do You Want to Distribute Test Set by Source?', variable=source).grid(column=1, sticky=E)

    # create an exit button
    Button(frame, bg='#66ff66', text='Done', command=root.quit).grid(column=1, sticky=E)

    # put the frame in the canvas
    canvas.create_window(0, 0, anchor='nw', window=frame)
    # make sure everything is displayed before configuring the scrollregion
    canvas.update_idletasks()

    # configure canvas and scrollbar
    canvas.configure(scrollregion=canvas.bbox('all'),
                     yscrollcommand=scroll_y.set)
    canvas.pack(fill='both', expand=True, side='left', padx=250)
    scroll_y.pack(fill='y', side='right')

    root.mainloop() # Necessary to display GUI
    root.withdraw() # Exits GUI

    return grand_dict, source

def add_label_sliders(df):
    labels = df['top_label'].unique().tolist()
    slider_dict = {}
    for label in labels:
        count = len(df[df['top_label'] == label])
        slider_dict[label] = DoubleVar()
        slider_dict[label].set(count)
        Label(root, bg='#b3ffb3', text='Select Max Sample For ' + str(label), font=("TimesNewRoman", 8)).pack()
        make_slider(count, label, slider_dict)

    return slider_dict


def make_slider(cap, variable, dict):
    Scale(root, bg='white', from_=0, to=cap, length=1000, orient=HORIZONTAL,
          variable=dict[variable], width=10, font=("TimesNewRoman", 8)).pack()


def df_settings(dictionary, source, dict):

    settings = ["DATA FRAME SETTINGS \nThis is what is included in the data frame: \n\n"]

    for column in dictionary:
        settings.append(str(column) + '\n')
        for option in dictionary[column]:
            if dictionary[column][option].get() == 1:
                settings.append(str(option) + " \n")
        settings.append('\n')

    if source == 1:
        settings.append('Test Set Distributed by Source? \n' + 'yes\n\n')
    else:
        settings.append('Test Set Distributed by Source? \n' + 'no\n\n')

    settings.append('Multi-Class Encoding Key:\n')
    settings.append(str(dict))

    file = open("DataFrame_Settings.txt", "w")
    file.writelines(settings)

def main():
    """
    This function run all the code in the neccessary order.
    """
    df_o = file_selection()
    test, shuff= options()
    slider_dict = add_label_sliders(df_o)
    dictionary, source = check_box(df_o)
    df_o = mix_df(df_o, shuff.get())
    fil_df = filter(df_o, dictionary, slider_dict)
    test_set = create_test_set(fil_df, test.get(), source.get())
    set_df = create_set_list(test_set, fil_df)
    class_df, key_dict = add_label_code(set_df)
    final_df = add_binary_columns(class_df)
    df_settings(dictionary, source.get(), key_dict)
    save_csv(final_df)

if __name__ == '__main__':
    main()