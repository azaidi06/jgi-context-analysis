#!/bin/bash/python

import os
os.environ['HF_HOME'] = '/pscratch/sd/a/azaidi/llm/cache'

from utils import *
from tqdm import tqdm
import argparse
import json


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--config", help=load_json, )
    debug_mode = True

    print('starting')
    
    '''
    Lets get the dataset
    '''
    ds = jgi_dataset()
    if debug_mode: print('got dataset\n')
    
    '''
    Now lets get our model
        To-do: make this a field that can be set in the json -- default to llama 8
    '''
    model_type = 'l8' #l8 --> llama8B; l70 --> 70B; l405 --> 405B
    pipeline = get_pipeline(model_type, eigth_bit=False, four_bit=True,)
    print(f'got pipeline: we are using {model_type}\n')
    
    
#     system_dirs = "Your output should only feature valid json\n\
# No text should be outside of the json format\n\
# The user would like to know more information about specific items of genomic data\n\
# Please include information about tool use and please validate the JSON output prior to showing it."
    system_dirs = "The user would like to know more information about specific items of genomic data\n\
        Please only include information that you were provided in the prompt"
    '''
    # base_prompt defaults to
     base_base --> f'Can you help me understand the role the genbank identifier {target_key} has in the following paper?\
     base_prompt_top --> '\nPlease note whether the dataset is used in the paper or just mentioned -- if it is used, please clarify how it was used \n'
    '''
    #base_prompt_top = '\nPlease note whether the dataset is used in the paper or just mentioned -- if it is used, please explain its use cases individually \n'
    base_prompt_top = '\nPlease note what tools were used with respect to the dataset in the paper \n'
    
    # paper_foll... defaults to--> '\nPlease ensure your output is in this json format:\n'
    paper_followup_prompt = "ç"
    

    identifiers = get_test_target_keys()
    if debug_mode: 
        print(f'These are the identifiers we are looking at: {identifiers}')
    
    
    out_df = run_model(pipeline, ds,
                       model_type=model_type,
                       system_dir=system_dirs,
                       num_samples=2,  #len(identifiers), 
                       target_keys=identifiers,
                       append_prompts=True, 
                       print_prog=False, 
                       one_shot_ids=None, 
                       save=True,
                       trial_name='testing',
                       csv_name=None,
                       debug_prompt=False,
                       base_prompt_top=base_prompt_top,
                       paper_followup_prompt=paper_followup_prompt,
                       include_example_output=False,)
    
    print(out_df.shape)