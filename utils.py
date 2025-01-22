import os
os.environ['HF_HOME'] = '/pscratch/sd/a/azaidi/llm/cache'
from tqdm import tqdm
import datetime
import pandas as pd
import transformers
import requests
import argparse
import json
import torch
import json


def read_paper(paper_name):
    delimiter = '|~|'
    # Read the string back from the file
    with open(f'papers/{paper_name}.txt', 'r', encoding='utf-8') as file:
        content = file.read()
    # Recreate the list by splitting the string by the delimiter
    return content.split(delimiter)

# TO be DEPRECATED
def build_prompt(system_directions, about_user=None, prompt=None):
    messages = [{"role": "system", "content": system_directions}]
    if about_user:   
        messages.append({"role": "user", "content": f"{about_user}"})
    if prompt:
        messages.append({"role": "user", "content": f"{prompt}"})
    return messages


def construct_prompt(system=None, user=None, assistant=None):
    '''
    Will better handle multi-turn prompting
    Allows for addition of assistant prompt type
    **  Will need to use X.extend functionality to ensure multiple calls of this
        end up in a single list vs a list of lists if you use append
    '''
    messages = []
    if system:
        messages.append(get_prompt_module(
            prompt_role='system',
            prompt = system
        ))
    if user:
        messages.append(get_prompt_module(
            prompt_role='user',
            prompt = user
        ))
    if assistant:
        messages.append(get_prompt_module(
            prompt_role='assistant',
            prompt = assistant
        ))
    return messages


def get_prompt_module(prompt_role=None, prompt=''):
    '''
    General function to return appropriate dictionary 
    With user type and prompt content 
    '''
    if prompt_role is None:
        prompt_role = 'user'
    return {'role': prompt_role, 'content': prompt}


def get_quant_config(eigth_bit, four_bit):
    config = None
    if eigth_bit is True:
        config = get_8bit_config()
    if four_bit is True:
        config = get_4bit_config()
    return config


def get_8bit_config():
    quantization_config = transformers.BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_skip_modules=None,
        llm_int8_enable_fp32_cpu_offload=False,
        llm_int8_has_fp16_weight=False
    )
    return quantization_config

def get_4bit_config():
    quantization_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    return quantization_config

#can pass in l8,l70 or l405 -- or can just pass the full model_name
def get_model_name(model_name='l70'):
    if model_name == 'l8':
        model_name = "meta-llama/Llama-3.1-8B-Instruct"
    if model_name == 'l70':
        model_name = "meta-llama/Llama-3.3-70B-Instruct"
    if model_name == 'l405':
        model_name = "meta-llama/Meta-Llama-3.1-405B-Instruct"
    if model_name == 'qwen':
        model_name = "Qwen/Qwen2.5-72B-Instruct"
    return model_name


def get_pipeline(model='l8', 
                 four_bit=False, 
                 eigth_bit=False,
                 device='auto',
                 ):
    model_name = get_model_name(model)
    # Get appropriate quantization config values
    quantization_config = get_quant_config(four_bit=four_bit, eigth_bit=eigth_bit)
    
    pipeline = transformers.pipeline(
            "text-generation",
            model=model_name,
            torch_dtype=torch.float16, #needed for 70B with bigger context windows
            model_kwargs={
                "quantization_config" : quantization_config,
            },
                device_map=device,
            )
    return pipeline


## temperature and top_p defaults are from llama3 docs:
## https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
def get_output(pipeline, prompt, max_new_tokens=100, temp=0.6, top_p=0.9,):
              #rep_penalty=5.5, length_penalty=1.0):
    outputs = pipeline(
        prompt,
        max_new_tokens=max_new_tokens,
        return_full_text=False,
        eos_token_id=128009,
        pad_token_id = 128009,
        temperature=temp,
        top_p = top_p,
        )
        #repetition_penalty=rep_penalty, 
        #length_penalty=length_penalty,
        #)
    return outputs[0]["generated_text"]


def get_base_df():
    return pd.DataFrame({}, columns=['system_prompt', 'user_prompt', 'output'])


def get_row_df(full_prompt, model_output):
        df = pd.DataFrame([full_prompt[0]['content'], 
                           full_prompt[1]['content'], 
                           model_output],
                       index=['system_prompt', 'user_prompt', 'output']).T
        return df
    
    
def store_output(pipeline,
                 df, 
                 system_directions=None, 
                 about_user=None, 
                 prompt=None, 
                 max_new_tokens=100, 
                 temp=0.6, 
                 top_p=0.9,): #rep_penalty=5.5, length_penalty=1.0):
    if prompt is None:
        prompt = "What can you tell me about CP000046.1?" 
    full_prompt = build_prompt(system_directions, about_user, prompt)
    model_output = get_output(pipeline, full_prompt, max_new_tokens=max_new_tokens, temp=temp, top_p=top_p)
    row_df = get_row_df(full_prompt, model_output)
    df = pd.concat([df, row_df], ignore_index=True)
    return df


def get_json(path):
    with open(path, 'r') as file:
        json_string = file.read()
        json_out = json.loads(json_string)
    return json_out


class jgi_dataset:
    def __init__(self):
        self.df = pd.read_csv('labels/hand_lbl.csv')
        self.target_keys = self.df.target_keys.unique()
        self.key_paper_dict = {tk: self.get_paper_name(tk, setup=True) for tk in self.target_keys}
        self.paper_dict = self.get_papers()
       
    # helpful for logging but also to help index into paper dict
    def get_paper_name(self, target_key, setup=False):
        if setup: ## yikes -- lol
            return self.df[self.df.target_keys == target_key].pmcid.unique().item()
        return self.key_paper_dict[target_key]
    
    #If target key is provided --> get paper name first
    def get_paper(self, target_key=None, paper_name=None):
        if target_key is None:
            return self.paper_dict[paper_name]
        else:
            return self.paper_dict[self.get_paper_name(target_key)]
    
    def get_metadata(self, target_key):
        url = f'https://dce.jgi.doe.gov/api/prompt?id={target_key}'
        res = requests.get(url)
        return res.text
    
    def get_papers(self):
        paper_names = [fname.split('.')[0] for fname in os.listdir('papers/') if fname.endswith('.txt')]
        papers = [read_paper(paper_name) for paper_name in paper_names]
        paper_dict = dict(zip(paper_names, papers))
        return paper_dict


def inject_metaprompt(target_key, 
                      ds, 
                      example_out=None, 
                      paper_len=None,
                      include_metadata=False,
                      base_prompt_top=None,
                      paper_followup_prompt=None,
                      include_example_output=False,
                      debug_prompt=False):
    paper_name = ds.key_paper_dict[target_key]
    paper = ds.paper_dict[paper_name]
    meta = ds.get_metadata(target_key)
    if paper_len is None:
        paper_len = len(paper)

    base_prompt_base = f'Can you help me understand the role the genbank identifier {target_key} has in the following paper?'
    if base_prompt_top is None:
        base_prompt_top = '\nPlease note whether the dataset is used in the paper or just mentioned -- if it is used, please clarify how it was used \n'
    base_prompt = base_prompt_base + base_prompt_top
    prompt = base_prompt
    
    #if we want to include metadata -- not being used right now
    if include_metadata:
        prompt += 'here is some associated metadata:\n{meta}\n'
    
    #adding paper here -- this is huge with respect to character count and logging
    prompt += f'\nhere is the paper {paper[:paper_len]}\n'

    if paper_followup_prompt is True:
        paper_followup_prompt = '\nPlease ensure your output is in this json format:\n'
    else:
        paper_followup_prompt = ''

    prompt = prompt + paper_followup_prompt

    # adding the response template here -- a json with some constraint
    if type(example_out) is not dict:
        example_out = get_json(f'labels/{paper_name}_{target_key}.json')
    if include_example_output:
        prompt = prompt + str(example_out)
    else:
        example_out = ''

    if debug_prompt:
        prompt_debugger(base_prompt, paper_followup_prompt, example_out)

    return prompt, base_prompt

def prompt_debugger(base_prompt, paper_followup_prompt, example_out):
        print(f'Here is our base prompt: {base_prompt}')
        print('/n------/n')
        print(f'Here is our paper followup prompt {paper_followup_prompt}')
        print('/n------/n')
        print(f'here is our example output {example_out}')
        print('/n------/n')


class PromptBuilder():
    def __init__(self, 
                 ds,
                 system_direction=None,
                 one_shot_example='MINA00000000',
                 include_metadata=False,
                 system_rag = False,
                 user_rag = False,
                 double_directions = False,
                 base_prompt_top=None,
                 paper_followup_prompt=None,
                 include_example_output=True
                ):
        self.ds = ds
        self.target_keys = ds.target_keys
        self.one_shot_example = one_shot_example
        self.response_template = get_json('labels/response_template.json')
        self.include_metadata = include_metadata
        self.add_system_rag = system_rag
        self.use_rag = user_rag
        self.double_directions = double_directions
        self.base_prompt_top = base_prompt_top
        self.paper_followup_prompt = paper_followup_prompt
        self.include_example_output = include_example_output

        if self.use_rag | self.add_system_rag:    
            self.one_shot_prompt = inject_metaprompt(self.one_shot_example, 
                                                     ds, 
                                                     example_out=True, 
                                                     paper_len=None,
                                                     include_metadata=self.include_metadata,
                                                     include_example_output=include_example_output,
                                                     base_prompt_top=self.base_prompt_top)
  
        if system_direction is None:
            self.system_directions =  "The user would like to know more information about specific items of genomic data\n\
            Please provide clear and concise answers.\n\
            Only state what you know to be true and if something is unclear please state that clearly or indicate that you do not\
            know the answer"
        else:
            self.system_directions = system_direction
        # if self.add_system_rag:
        #     self.system_directions = self.build_one_shot_prompt(self.one_shot_key,
        #                                                    self.one_shot_example,)
                                                           
       
    def build_one_shot_prompt(target_key, example_output, paper_len=None):
        prompt = inject_metaprompt(target_key, self.ds, example_out=True)
    
    #build the user prompt in question
    def get_user_prompt(self, 
                        target_key, 
                        paper_len=None,
                        debug=False,
                        ):
        return inject_metaprompt(target_key, 
                                 self.ds, 
                                 example_out=self.response_template, 
                                 paper_len=paper_len,
                                 include_example_output=None,
                                 debug_prompt=debug,
                                 base_prompt_top=self.base_prompt_top,)

    
    def build_full_prompt(self, 
                          target_key, 
                          append_prompts=False, 
                          paper_len=None,
                          debug=False,):
                        #   base_prompt_top=self.base_prompt_top):
        user_prompt, base_prompt = self.get_user_prompt(target_key, 
                                                        paper_len=None,
                                                        debug=debug,
                                                        )
        if append_prompts:
            if self.use_rag:
                '''
                prompt will look like this
                (1) system_directions -- role: system
                (2) User prompt -- role: user
                     (a) Rag example (paper + annotated json)
                     (b) Paper in question
                     (c) Json template
                '''
                user_prompt = self.one_shot_prompt + user_prompt
            if self.double_directions:
                '''
                prompt will look like this
                (1) system_directions -- role: system
                (2) User prompt -- role: user
                     (a) Rag example (paper + annotated json)
                     (b) Paper in question
                     (c) Json template
                     (d) system direction -- emphasize/remind model
                '''
                user_prompt = user_prompt + self.system_directions
            full_prompt = build_prompt(system_directions=self.system_directions, 
                                       about_user=None, 
                                       prompt=user_prompt)
        else: # this will put the rag example into the user prompt
            '''In this case the prompt will look like this:
            (1) system_directions -- role: system
            (2) Rag example -- role: user
            (3) Paper in question -- role: user
            '''
            if self.use_rag:
                full_prompt = build_prompt(
                                system_directions=self.system_directions, 
                               about_user=self.one_shot_prompt, 
                               prompt=user_prompt)
            else:
                full_prompt = build_prompt(system_directions=self.system_directions, 
                               prompt=user_prompt)
        return full_prompt, target_key, self.ds.get_paper_name(target_key), base_prompt
    
    
    
def get_log_df():
    return pd.DataFrame(columns=['target_key', 'pmcid', 
                                 'one_shot_key', 'one_shot_pmcid',
                                 'system_directions', 
                                 'init_user_prompt', 'user_prompt',
                                 'output', 'temp', 'top_p', 'max_new_tokens'])


def log_output(pipeline, 
               prompt, 
               target_key, 
               pmcid, 
               one_shot_key, 
               one_shot_pmcid, 
               max_new_tokens=100, 
               temp=0.6, 
               top_p=0.9, 
               append_prompts=False,
               base_prompt=None):
    df = get_log_df()
    output = get_output(pipeline, prompt, max_new_tokens=max_new_tokens,
                        temp=temp, top_p=top_p)
    if append_prompts:
         df.loc[0] = [target_key, pmcid,
                      one_shot_key, one_shot_pmcid,
                      prompt[0]['content'], "None", base_prompt,#prompt[1]['content'],
                      output, temp, top_p, max_new_tokens]
    else:
        df.loc[0] = [target_key, pmcid,
                     prompt[0]['content'], prompt[1]['content'], prompt[2]['content'],
                     output, temp, top_p, max_new_tokens]
    return df


def run_model(pipeline, ds,
              system_dir,
              model_type,
              num_samples=10,
              target_keys=['CP000029',
                          #'MINA00000000',
                           # 'CU928158'
                           #'CP000046'
              ],
              one_shot_ids = None,
              append_prompts=False, 
              max_new_tokens=250, 
              temp=0.025, 
              csv_name=None,
              trial_name=None,
              save=False,
              print_prog=False,
              system_rag = False,
              user_rag = False,
              double_directions = True,
              base_prompt_top=None,
              paper_followup_prompt=None,
              include_example_output=False,
              debug_prompt=False,
             ):
    holder = []
    if one_shot_ids is None:
        one_shot_ids = [None for x in range(num_samples)]
    date, time = get_time()
    if trial_name is None:
        trial_name = f'testing_{time}'
    make_trial_folder(f'results/{trial_name}')
    for x in tqdm(range(num_samples)):
        if one_shot_ids:
            one_shot_key = one_shot_ids[x]
            if one_shot_key is None:
                        one_shot_pmcid = None
            else:
                        one_shot_pmcid = ds.key_paper_dict[one_shot_key]
            # all of the above if/then to determine how to log one_shot_example
                        
            pb = PromptBuilder(ds, system_direction=system_dir,
                               one_shot_example=one_shot_key,
                               system_rag = system_rag,
                               user_rag = user_rag,
                               double_directions = double_directions,
                               base_prompt_top=base_prompt_top,
                               paper_followup_prompt=paper_followup_prompt,
                               include_example_output=include_example_output,
                              )

        if print_prog: print(x)
        if len(target_keys) > 1:
            target_key = target_keys[x]
        else:
            target_key = target_keys[0]
        prompt, tar_key, pmcid, base_prompt = pb.build_full_prompt(target_key,
                                                     append_prompts=append_prompts,
                                                     debug=debug_prompt)
        holder.append(log_output(pipeline, 
                                 prompt, 
                                 tar_key, 
                                 pmcid, 
                                 one_shot_key=one_shot_key, 
                                 one_shot_pmcid=one_shot_pmcid,
                                 max_new_tokens=max_new_tokens, 
                                 temp=temp, 
                                 append_prompts=append_prompts,
                                 base_prompt=base_prompt))
        df = pd.concat(holder).reset_index(drop=True)
        df.attrs['time'] = time
        df.attrs['date'] = date
    if user_rag | system_rag: #one target key to many rag examples
        csv_name = f'results/{trial_name}/{csv_name}_{model_type}_{target_key}.csv'
    else: # bunch of target keys and rag examples, so one csv title works
        csv_name = f'results/{trial_name}/{model_type}.csv'
    if save:
        df.to_csv(csv_name, index=False)
    return df


def get_time():
    dt = datetime.datetime.now()
    date = f'{dt.month}_{dt.day}'
    return date, time
    time = f'{dt.hour}:{dt.minute}'


def make_trial_folder(trial_name):
    os.makedirs(trial_name, exist_ok=True)


def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)
   

def get_test_target_keys():
        prelim_idens = [x.split('.')[:1][0].split('_')[1:] for x in os.listdir('labels')]
        identifiers = [x[0] if len(x) == 1 else '_'.join(x) for x in prelim_idens]
        identifiers = [ids for ids in identifiers if len(ids) > 2]
        return identifiers


def debug_stuff(fast=True): 
    ds = jgi_dataset()
    model_type = 'l8'
    pipeline = get_pipeline(model_type, eigth_bit=False, four_bit=fast,)
    system_dirs = "The user would like to know more information about specific items of genomic data\n\
        Please only include information that you were provided in the prompt"
    
    prelim_idens = [x.split('.')[:1][0].split('_')[1:] for x in os.listdir('labels')]
    identifiers = [x[0] if len(x) == 1 else '_'.join(x) for x in prelim_idens]
    identifiers = [ids for ids in identifiers if len(ids) > 2]

    pb = PromptBuilder(ds, system_direction=system_dirs,
                    include_example_output=False,
                    )
    return pipeline, pb, identifiers