import argparse
import os
import pickle
import json
from processor import encode_midi
import random
import pretty_midi
import processor as midi_processor
 
JSON_FILE = "maestro-v2.0.0.json"
 
# prep_midi
def prep_midi(custom_MIDI_DataSet_dir, output_dir):
   """
   ----------
   Author: Damon Gwinn
   ----------
   Pre-processes the maestro dataset, putting processed midi data (train, eval, test) into the
   given output folder
   ----------
   """
 
   train_dir = os.path.join(output_dir, "train")
   os.makedirs(train_dir, exist_ok=True)
   val_dir = os.path.join(output_dir, "val")
   os.makedirs(val_dir, exist_ok=True)
   test_dir = os.path.join(output_dir, "test")
   os.makedirs(test_dir, exist_ok=True)
   f_ext = '.pickle'
   fileList = os.listdir(custom_MIDI_DataSet_dir)
   # print(fileList)
 
   total_count = 0
   train_count = 0
   val_count   = 0
   test_count  = 0
 
   for file in fileList:
       # we gonna split by a random selection for now
      
       split = random.randint(1, 2)
       if (split == 0):
           o_file = os.path.join(train_dir, file+f_ext)
           train_count += 1
 
       elif (split == 2):
           o_file0 = os.path.join(train_dir, file+f_ext)
           train_count += 1
           o_file = os.path.join(val_dir, file+f_ext)
           val_count += 1
 
       elif (split == 1):
           o_file0 = os.path.join(train_dir, file+f_ext)
           train_count += 1
           o_file = os.path.join(test_dir, file+f_ext)
           test_count += 1
       try:
           print("dcdvxdxv",os.path.join(custom_MIDI_DataSet_dir,file))
           prepped = encode_midi(os.path.join(custom_MIDI_DataSet_dir,file))
          
          
           o_stream = open(o_file0, "wb")
           pickle.dump(prepped, o_stream)
           o_stream.close()
          
           prepped = encode_midi(os.path.join(custom_MIDI_DataSet_dir,file))
           o_stream = open(o_file, "wb")
           pickle.dump(prepped, o_stream)
           o_stream.close()
  
           print(file)
           print(o_file)
           print('Coverted!') 
       except KeyboardInterrupt:
           raise  
       except:
           print('Bad file. Skipping...')
           exit()
 
   print('Done')
   print("Num Train:", train_count)
   print("Num Val:", val_count)
   print("Num Test:", test_count)
   print("Total Count:", train_count)
   return True
 
 
 
# parse_args
def parse_args():
   """
   ----------
   Author: Damon Gwinn
   ----------
   Parses arguments for preprocess_midi using argparse
   ----------
   """
 
   parser = argparse.ArgumentParser()
 
   parser.add_argument("maestro_root", type=str, help="Root folder for the Maestro dataset")
   parser.add_argument("-output_dir", type=str, default="./dataset/e_piano", help="Output folder to put the preprocessed midi into")
 
   return parser.parse_args()
 
# main
def main():
   """
   ----------
   Author: Damon Gwinn
   ----------
   Entry point. Preprocesses maestro and saved midi to specified output folder.
   ----------
   """
 
   # args            = parse_args()
   # custom_MIDI_DataSet_dir    = args.custom_MIDI_DataSet_dir
   # output_dir      = args.output_dir
 
   # print("Preprocessing midi files and saving to", output_dir)
   prep_midi('C:\\Users\\krish\\Downloads\\EE641\\EE641_project_12\\MusicTransformer-Pytorch\\dataset\\e_piano\\custom_midis', 'C:\\Users\\krish\\Downloads\\EE641\\EE641_project_12\\MusicTransformer-Pytorch\\dataset\\e_piano')
   print("Done!")
   print("")
 
if __name__ == "__main__":
   main()
 
 

