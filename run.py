#!/bin/bash/python

import os
os.environ['HF_HOME'] = '/pscratch/sd/a/azaidi/llm/cache'

from utils import *
from tqdm import tqdm
import argparse
import json


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help=load_json, )#default='l8')
    #parser.add_argument("-v", "--verbose", action="store_true")
    # args = parser.parse_args()
    # config_dict = vars(args).get('config', {})
    # print(config_dict['model'])
    print('starting')
    ds = jgi_dataset()
    print('got dataset\n')
    #print(ds)
    model_type = 'l8' #l8 --> llama8B; l70 --> 70B; l405 --> 405B
    #pipeline = get_pipeline(model_type, eigth_bit=False, four_bit=False,)
    print(f'got pipeline: we are using {model_type}\n')
    _, _, _ = debug_stuff()
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
    paper_followup_prompt = "Please ensure that the tools are output in a list format"
    
    prelim_idens = [x.split('.')[:1][0].split('_')[1:] for x in os.listdir('labels')]
    identifiers = [x[0] if len(x) == 1 else '_'.join(x) for x in prelim_idens]
    identifiers = [ids for ids in identifiers if len(ids) > 2]
    print(f'These are the identifiers we are looking at: {identifiers}')
    
    
    # out_df = run_model(pipeline, ds,
    #                    model_type=model_type,
    #                    system_dir=system_dirs,
    #                    num_samples=len(identifiers), 
    #                    target_keys=identifiers,
    #                    append_prompts=True, 
    #                    print_prog=False, 
    #                    one_shot_ids=None, 
    #                    save=True,
    #                    trial_name='tools',
    #                    csv_name=None,
    #                    debug_prompt=False,
    #                    base_prompt_top=base_prompt_top,
    #                    paper_followup_prompt=paper_followup_prompt,
    #                    include_example_output=False,)
    
    # print(out_df.shape)