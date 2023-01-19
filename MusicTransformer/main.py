from processor import encode_midi, decode_midi
import os
import librosa
import numpy as np
import pretty_midi
import pypianoroll
from pypianoroll import Multitrack, Track
import matplotlib.pyplot as plt
import librosa.display
import os
import torch

'''
    This file is used to generate the output from the re-trained model.
    The model is re-trained on the Tegridy files.
    The output is generated in the output folder.
    The output is generated in the form of midi files.
    MIDI pre-processor provided by jason9693 et al. (https://github.com/jason9693/midi-neural-processor),
    which is used to convert the MIDI file into discrete ordered message types for training and evaluating.
'''

#@title Graph the results
import argparse
import os
import csv
import math
import matplotlib.pyplot as plt

RESULTS_FILE = "results.csv"
EPOCH_IDX = 0
LR_IDX = 1
EVAL_LOSS_IDX = 4
EVAL_ACC_IDX = 5

SPLITTER = '?'


def graph_results(input_dirs="C:\\Users\\krish\\Downloads\\EE641\\EE641_project_12\\MusicTransformer-Pytorch\\rpr\\results\\", output_dir=None, model_names=None, epoch_start=0, epoch_end=None):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Graphs model training and evaluation data
    ----------
    """

    input_dirs = input_dirs.split(SPLITTER)

    if(model_names is not None):
        model_names = model_names.split(SPLITTER)
        if(len(model_names) != len(input_dirs)):
            print("Error: len(model_names) != len(input_dirs)")
            return

    #Initialize Loss and Accuracy arrays
    loss_arrs = []
    accuracy_arrs = []
    epoch_counts = []
    lrs = []

    for input_dir in input_dirs:
        loss_arr = []
        accuracy_arr = []
        epoch_count = []
        lr_arr = []

        f = os.path.join(input_dir, RESULTS_FILE)
        with open(f, "r") as i_stream:
            reader = csv.reader(i_stream)
            next(reader)

            lines = [line for line in reader]

        if(epoch_end is None):
            epoch_end = math.inf

        epoch_start = max(epoch_start, 0)
        epoch_start = min(epoch_start, epoch_end)

        for line in lines:
            epoch = line[EPOCH_IDX]
            lr = line[LR_IDX]
            accuracy = line[EVAL_ACC_IDX]
            loss = line[EVAL_LOSS_IDX]

            if(int(epoch) >= epoch_start and int(epoch) < epoch_end):
                accuracy_arr.append(float(accuracy))
                loss_arr.append(float(loss))
                epoch_count.append(int(epoch))
                lr_arr.append(float(lr))

        loss_arrs.append(loss_arr)
        accuracy_arrs.append(accuracy_arr)
        epoch_counts.append(epoch_count)
        lrs.append(lr_arr)

    if(output_dir is not None):
        try:
            os.mkdir(output_dir)
        except OSError:
            print ("Creation of the directory %s failed" % output_dir)
        else:
            print ("Successfully created the directory %s" % output_dir)

    ##### LOSS #####
    for i in range(len(loss_arrs)):
        if(model_names is None):
            name = None
        else:
            name = model_names[i]

        #Create and save plots to output folder
        plt.plot(epoch_counts[i], loss_arrs[i], label=name)
        plt.title("Loss Results")
        plt.ylabel('Loss (Cross Entropy)')
        plt.xlabel('Epochs')
        fig1 = plt.gcf()

    plt.legend(loc="upper left")

    if(output_dir is not None):
        fig1.savefig(os.path.join(output_dir, 'loss_graph.png'))

    plt.show()

    ##### ACCURACY #####
    for i in range(len(accuracy_arrs)):
        if(model_names is None):
            name = None
        else:
            name = model_names[i]

        #Create and save plots to output folder
        plt.plot(epoch_counts[i], accuracy_arrs[i], label=name)
        plt.title("Accuracy Results")
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')
        fig2 = plt.gcf()

    plt.legend(loc="upper left")

    if(output_dir is not None):
        fig2.savefig(os.path.join(output_dir, 'accuracy_graph.png'))

    plt.show()

    ##### LR #####
    for i in range(len(lrs)):
        if(model_names is None):
            name = None
        else:
            name = model_names[i]

        #Create and save plots to output folder
        plt.plot(epoch_counts[i], lrs[i], label=name)
        plt.title("Learn Rate Results")
        plt.ylabel('Learn Rate')
        plt.xlabel('Epochs')
        fig2 = plt.gcf()

    plt.legend(loc="upper left")

    if(output_dir is not None):
        fig2.savefig(os.path.join(output_dir, 'lr_graph.png'))

    plt.show()


def output_wav(path):
# Plot and Graph the Output :
    graphs_length_inches = 50 
    notes_graph_height = 10
    highest_displayed_pitch = 92
    lowest_displayed_pitch = 24
    piano_roll_color_map = "Blues"
    midi_data = pretty_midi.PrettyMIDI(path)

    def plot_piano_roll(pm, start_pitch, end_pitch, fs=100):
        librosa.display.specshow(pm.get_piano_roll(fs)[start_pitch:end_pitch],
                                hop_length=1, sr=fs, x_axis='time', y_axis='cqt_note',
                                fmin=pretty_midi.note_number_to_hz(start_pitch))

    roll = np.zeros([int(graphs_length_inches), 128])
    # Plot the output

    track = pypianoroll.read(path)
    fig = track.plot()
    plt.figure(figsize=[graphs_length_inches, notes_graph_height])
    ax2 = plot_piano_roll(midi_data, int(lowest_displayed_pitch), int(highest_displayed_pitch))
    plt.show()



def main():
    no_tokens = 512
    sequence_length = 512
    maximum_output_length = 512
    cur = os.getcwd()
    output = os.path.join(cur,"output","rand.mid")
    model = os.path.join(cur,"rpr","results","best_loss_weights.pickle")
    loss_weights = os.path.join(cur,"rpr","results","best_loss_weights.pickle")
    accu_weights = os.path.join(cur,"rpr","results","best_acc_weights.pickle")

    print(model)

    # os.system(f"python evaluate.py -model_weights {loss_weights} --rpr -max_sequence {maximum_output_length}")
    # os.system(f"python evaluate.py -model_weights {accu_weights} --rpr  -max_sequence {maximum_output_length}")

    graph_results(model_names='rpr',epoch_start=150, epoch_end=200)

    os.system(f"python generate.py -output_dir output -model_weights {model} --rpr -target_seq_length {no_tokens}  -num_prime {sequence_length} -max_sequence {maximum_output_length}")
    print('Generated the output')

    output_wav(output)
    print("output:")
    
if __name__ == "__main__":
    main()