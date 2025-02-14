#!/bin/bash/python

import os
# comment the line below if using locally
#os.environ['HF_HOME'] = '/pscratch/sd/a/azaidi/llm/cache'

import pdb
from utils import *
from tqdm import tqdm
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help='Path of config Json')
    config_path = parser.parse_args().config
    with open(config_path, 'r') as config_fp:
        config = json.load(config_fp)
    debug_mode = True

    print('starting')
    
    ds = jgi_dataset()
    if debug_mode: print('got dataset\n')
    
    model_type = config['model_type']
    pipeline = get_pipeline(model_type, eigth_bit=False, four_bit=False,)
    print(f'got pipeline: we are using {model_type}\n')

    identifiers = get_test_target_keys()
    if debug_mode: 
        print(f'These are the identifiers we are looking at: {identifiers}')

    out_df = run_model(pipeline=pipeline,
                       ds=ds, 
                       config=config)
    print(out_df.shape)

''' 
Things to declare upfront

FOR PROMPT
    [0] System Directions
    [1] Generic Prompt
        [A] [B] [C]
    [2] Generic Prompt <OPTIONAL>
        -- would be used if rag key included
        -- Rag would go in front and the normal
        --      prompt/target key would come here

Therefore the following needs to be declared:
    - Target Key
    - Prompt Stuff
        - System Directions
        - Front
        - Middle
        - End
    - Include Template <OPTIONAL -- no field means no template>
    - Rag Key <OPTIONAL -- None provided means no rag>

Stuff to Log:
    - Target Key
    - Rag Key             --> 
    - Template Used       --> gp.include_example_output
    - Prompt -- W/O PAPER --> gp.bare_prompt
    - Output
    
'''   

'''
        self.target_key = target_key
        
        self.prompt_front = prompt_front        # [A] 
        self.prompt_middle = prompt_middle      # [B]
        self.prompt_end = prompt_end            # [C]
        
        self.include_paper = include_paper
        self.paper_name = self.get_pmcid(self.target_key) #PMCID

        self.include_rag_example = include_rag_example
        self.include_example_output = include_example_output
        self.example_output = self.get_example_output_file()

        self.prompt = self.build_prompt()
'''


'''
#     system_dirs = "Your output should only feature valid json\n\
# No text should be outside of the json format\n\
# The user would like to know more information about specific items of genomic data\n\
# Please include information about tool use and please validate the JSON output prior to showing it."
    system_dirs = "The user would like to know more information about specific items of genomic data\n\
        Please only include information that you were provided in the prompt"
    
    # base_prompt defaults to
     #base_base --> f'Can you help me understand the role the genbank identifier {target_key} has in the following paper?\
     #base_prompt_top --> '\nPlease note whether the dataset is used in the paper or just mentioned -- if it is used, please clarify how it was used \n'
    #base_prompt_top = '\nPlease note whether the dataset is used in the paper or just mentioned -- if it is used, please explain its use cases individually \n'
    base_prompt_top = '\nPlease note what tools were used with respect to the dataset in the paper \n'
    
    # paper_foll... defaults to--> '\nPlease ensure your output is in this json format:\n'
    paper_followup_prompt = "ç"
'''