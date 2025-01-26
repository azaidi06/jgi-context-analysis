# jgi-context-analysis
JGI Data Context Analysis

To-Do:
    - Json input option
    - Correct logging


Prompt would look like this: [Based on PromptModule + GenericPrompt classes]
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
