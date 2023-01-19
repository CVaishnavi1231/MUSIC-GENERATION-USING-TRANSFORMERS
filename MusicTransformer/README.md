# Generating Music From Transformer using rpr

The file contains the information required to use the Music Transformer 
(which is a PyTorch's Transformer with relative position encoding implemented) on your own MIDI datasets. 

## File Structure

For checking the training scripts see train.py. 
The libraries used to preprocess the MIDI files such as encoding and decoding are cloned from the repositories using 

!git clone https://github.com/asigalov61/midi-neural-processor

The training scripts reside in the MusicTransformer-Pytorch directory. 

-- preprocess.py  
Used to preprocess midi files using the library pretty_midi.
We thank the authors for their valuable contributions.

## How to run
Run main.py for getting the evaluations and outputs on test data for the best model based on highest accuracy or lowest loss.

Outputs will be stored in the /output folder.
rpr folder has 
results folder has loss and accuracy results over the entire range of epochs
weights folder has weights of the best model.

Model was trained for 150 epochs baseline and 50 more on custom dataset.

## model params
rpr: True
lr: None
ce_smoothing: None
batch_size: 4
max_sequence: 512
n_layers: 6
num_heads: 8
d_model: 512
dim_feedforward: 1024
dropout: 0.1


Size of the model was approximately equal to 50MB.

