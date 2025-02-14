import os
# comment out line below if running locally and NOT on perlmutter
#os.environ['HF_HOME'] = '/pscratch/sd/a/azaidi/llm/cache'
from tqdm import tqdm
import datetime
import pandas as pd
import transformers
import requests
import argparse
import json
import torch
import json
import pdb


def read_paper(paper_name):
    delimiter = '|~|'
    # Read the string back from the file
    with open(f'papers/{paper_name}.txt', 'r', encoding='utf-8') as file:
        content = file.read()
    # Recreate the list by splitting the string by the delimiter
    return content.split(delimiter)


def construct_prompt(system=None, user=None, assistant=None, previous_chat=None):
    '''
    Will better handle multi-turn prompting
    Allows for addition of assistant prompt type
    **  Will need to use X.extend functionality to ensure multiple calls of this
        end up in a single list vs a list of lists if you use append
    '''
    messages = []
    if previous_chat:
        messages.append(previous_chat)
    if system:
        messages.append(
            get_prompt_module(
                prompt_role='system',
                prompt = system
        ))
    if user:
        messages.append(
            get_prompt_module(
                prompt_role='user',
                prompt = user
        ))
    if assistant:
        messages.append(
            get_prompt_module(
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
    if model_name == 'l1':
        model_name = "meta-llama/Llama-3.2-1B-Instruct"
    if model_name == 'l8':
        model_name = "meta-llama/Llama-3.1-8B-Instruct"
    if model_name == 'l70':
        model_name = "meta-llama/Llama-3.3-70B-Instruct"
    if model_name == 'l405':
        model_name = "meta-llama/Meta-Llama-3.1-405B-Instruct"
    if model_name == 'qwen':
        model_name = "Qwen/Qwen2.5-72B-Instruct"
    return model_name


def determine_hardware():
    device = torch.device("cuda" if torch.cuda.is_available()
                          #else "" if torch.backends.mps.is_available() 
                          else "cpu")
    return device


def get_pipeline(model='l8', 
                 four_bit=False, 
                 eigth_bit=False,
                 device='auto',
                 ):
    model_name = get_model_name(model)
    # Get appropriate quantization config values
    quantization_config = get_quant_config(four_bit=four_bit, eigth_bit=eigth_bit)
    if determine_hardware() != 'cuda':
        cpu = determine_hardware()
        device =  cpu
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

def get_debug_pipeline():
    pipeline = transformers.pipeline("text-generation", 
                                     model="unsloth/Llama-3.2-1B-Instruct")
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
    full_prompt = construct_prompt(system=system_directions, user=prompt)
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


def prompt_debugger(base_prompt, paper_followup_prompt, example_out):
        print(f'Here is our base prompt: {base_prompt}')
        print('/n------/n')
        print(f'Here is our paper followup prompt {paper_followup_prompt}')
        print('/n------/n')
        print(f'here is our example output {example_out}')
        print('/n------/n')


class PromptModule():
    '''
    Generic prompt component
    [A] - [B] - [C] positons
    User prompt, paper, model output, metadata
        or anything else can be added
    '''
    def __init__(self,
                 front_position='',
                 middle_position='',
                 end_position='',
                ):
        self.front_position = front_position
        self.middle_position = middle_position
        self.end_position = end_position
        self.full_prompt = construct_prompt(
            self.front_position,
            self.middle_position,
            self.end_position,
        )
    def construct_prompt(self):
        return self.front_position + self.middle_position + self.end_position

'''
Prompt could look like this: [Based on PromptModule + GenericPrompt classes]
    [0] Pre Module
        A) Front 
        B) Middle
        C) End --> Normally not used
    [1] Paper Module
        A) Front --> Preamble to paper or maybe a summary
        B) Middle --> Paper normally always goes here
        C) End  --> Follow up to paper or maybe a summary 
    [2] Post Paper Module
        A) Front --> Reminder of task/constraints
        B) Middle --> Example output
        C) End --> Normally NOT used

'''
class   GenericPrompt():
    def __init__(self,
                dataset,
                target_key, 
                include_metadata=False, # NOT BEING USED right now
                prompt_front='', # [A]  ALWAYS includes text (never paper or json)
                prompt_middle='', # [B] i.e. here's the paper in question:
                prompt_end='', # [C]
                include_rag_example=False, # pulls relevant rag based on target key provided
                include_example_output=2, # Position 3
                include_paper=1, # Middle prompt
                paper_length=None
                ):

        self.dataset = dataset
        self.target_key = target_key
        
        self.include_metadata = include_metadata
        if self.include_metadata:
            self.metadata = ds.get_metadata(target_key)
        
        self.prompt_front = prompt_front        # [A] 
        self.prompt_middle = prompt_middle      # [B]
        self.prompt_end = prompt_end            # [C]
        
        #lets set some variables
        self.include_paper = include_paper
        self.paper_name = self.get_pmcid(self.target_key) #PMCID
        self.paper_length = paper_length
        self.paper = self.get_paper() # Stored separately bc we don't want to log this

        self.include_rag_example = include_rag_example
        self.include_example_output = include_example_output
        self.example_output = self.get_example_output_file()

        self.prompt = self.build_prompt()
        self.bare_prompt = self.build_bare_prompt()

    def build_bare_prompt(self):
        sections = [self.prompt_front, self.prompt_middle, self.prompt_end]
        return '\n'.join(sections)
    
    def build_prompt(self):
        sections = [self.prompt_front, self.prompt_middle, self.prompt_end]
        if self.include_paper:
            sections[1] += self.paper
        if self.include_example_output:
            sections[2] += self.example_output
        return '\n'.join(sections)
        

    def get_pmcid(self, target_key):
        return self.dataset.key_paper_dict[target_key]
    

    def get_paper(self):
        paper = self.dataset.paper_dict[self.paper_name]
        paper = ' '.join(paper)
        if self.paper_length:
            paper = paper[:self.paper_length]
        return paper
    

    def get_example_output_file(self):
        path = (
            f'labels/{self.paper_name}_{self.target_key}.json'
            if self.include_rag_example
            else 'labels/response_template.json'
        )
        data = get_json(path)
        return json.dumps(data)

    
'''
Too hardcoded -- can the prompt info all be in one column?
'''
def get_log_df():
    return pd.DataFrame(columns=['target_key', 
                                 'target_pmcid', 
                                 'one_shot_key', 
                                 'one_shot_pmcid',
                                 'system_directions', 
                                 'init_user_prompt',
                                 'user_prompt',
                                 'output', 
                                 'temp', 
                                 'top_p', 
                                 'max_new_tokens'])

'''
        Can just pass pmdoule to log_output
        pipeline, system_directions, pmodule
        
        * pmodule --> prompt, target_key, pmcid, 
                    one_shot_key/pmcid, bare_prompt
'''
def new_log_output(output, 
                   pmodule, 
                   config, 
                   output_path,
                   target_key,
                   rag_key=None):
    file_name = f'{output_path}/tk_{target_key}'
    config['output'] = output
    config['THIS_target_key'] = target_key
    if rag_key:
        config['THIS_rag_key'] = rag_key
        file_name += f'rk_{rag_key}'
    with open(f'{file_name}.json', "w") as f:
        json.dump(config, f, indent=4)


'''
This needs alot of work
'''
def log_output(pipeline, 
               prompt, 
               target_key, 
               pmcid, 
               one_shot_key, 
               one_shot_pmcid, 
               max_new_tokens=100, 
               temp=0.6, 
               top_p=0.9, 
               base_prompt=None,
               config_json=None):
    df = get_log_df()
    #output = get_output(pipeline, prompt, max_new_tokens=max_new_tokens,
    #                    temp=temp, top_p=top_p)
    output = ''
    '''
    NOT COOL
    ** Want to log the json ** 
    '''
    # import pdb
    # pdb.set_trace()
    df.loc[0] = [target_key, pmcid,
                one_shot_key, one_shot_pmcid,
                prompt[0]['content'], "None", base_prompt,#prompt[1]['content'],
                output, temp, top_p, max_new_tokens]
    return df
'''
/output_file
    /results
        jsons
    log dataframe
    input json
'''

def setup_output_directory(trial_name,
                           parent_directory=None):
    date, time = get_time()
    if parent_directory is None:
        parent_directory = 'results'
    if trial_name is None:
        trial_name = f'testing_{time}'
    output_directory = f'{parent_directory}/{trial_name}'
    make_trial_folder(output_directory)
    return output_directory

## one_shot_ids

def run_model(pipeline, ds, config, rag_keys=None):
    model_type = config['model_type']
    num_samples=config['num_samples'] #needs to be at or below #of target keys
    target_keys=config['target_keys'] # want this to be a list that we iterate over
    rag_keys = config['rag_keys']
    max_new_tokens=config['max_new_tokens'] 
    temp = config['temperature']
    csv_name = config['trial_name']
    trial_name = config['trial_name']
    save=True,
    user_rag = config['include_rag_example']
    system_directions=config['system_directions']
    prompt_front=config['prompt_front']
    prompt_middle = config['prompt_middle']
    prompt_end = config['prompt_end']
    include_example_output=config['include_example_output']
    holder = []
    #if rag_keys is None:
    #    rag_keys = [None for x in range(num_samples)]
    output_directory = setup_output_directory(trial_name)
    #pdb.set_trace()
    for x in tqdm(range(num_samples)):            
        GenP = GenericPrompt(ds, 
                            ds.target_keys[x],
                            prompt_front=prompt_front,
                            prompt_middle=prompt_middle,
                            prompt_end=prompt_end,
                            include_paper=False,
                            include_example_output=False)
        
        pmodule = PromptModule(GenP.prompt)
        prompt = construct_prompt(system=system_directions, 
                                  user=pmodule.full_prompt)
        target_key = GenP.target_key
        pmcid = GenP.paper_name
        bare_prompt = GenP.bare_prompt
        output = get_output(pipeline, 
                            prompt, 
                            max_new_tokens=max_new_tokens,
                            temp=temp, 
                            top_p=0.9)
        holder.append(log_output(pipeline, 
                                 prompt, 
                                 target_key, 
                                 pmcid, 
                                 one_shot_key=None, 
                                 one_shot_pmcid=None,
                                 max_new_tokens=max_new_tokens, 
                                 temp=temp, 
                                 base_prompt=bare_prompt))
        df = pd.concat(holder).reset_index(drop=True)
        new_log_output(#output='meow', 
                output=output,
                pmodule=pmodule, 
                config=config, 
                output_path=output_directory,
                target_key=target_key)
    #if user_rag: #one target key to many rag examples
    #    csv_name = f'results/{trial_name}/{csv_name}_{model_type}_{target_key}.csv'
    #else: # bunch of target keys and rag examples, so one csv title works
    #    csv_name = f'results/{trial_name}/{model_type}.csv'
    #if save:
    #    df.to_csv(csv_name, index=False)
    return df


def get_time():
    dt = datetime.datetime.now()
    date = f'{dt.month}_{dt.day}'
    time = f'{dt.hour}:{dt.minute}'
    return date, time


def make_trial_folder(trial_name):
    os.makedirs(trial_name, exist_ok=True)


def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)
   

def get_test_target_keys():
        prelim_idens = [x.split('.')[:1][0].split('_')[1:] for x in os.listdir('labels') if x[:3] == 'PMC']
        identifiers = [x[0] if len(x) == 1 else '_'.join(x) for x in prelim_idens]
        identifiers = [ids for ids in identifiers if len(ids) > 2]
        return identifiers


def debug_stuff(fast=True): 
    ds = jgi_dataset()
    model_type = 'l8'
    pipeline = get_pipeline(model_type, eigth_bit=False, four_bit=fast,)
    system_dirs = "The user would like to know more information about specific items of genomic data\n\
        Please only include information that you were provided in the prompt"

    identifiers = get_test_target_keys()

    pb = PromptBuilder(ds, system_direction=system_dirs,
                    include_example_output=False,
                    )
    return pipeline, pb, identifiers