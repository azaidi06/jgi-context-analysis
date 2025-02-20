# jgi-context-analysis
JGI Data Context Analysis

To-Do:
    - Json input option --> done
    - Correct logging --> done


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




These are the target_key identifiers we have in our test set: ['CP031311', 'CP020347', 'CP020345', 'Gs0095506', 'NC_007624', 'CP011526', 'NC_004311', 'AM933173', 'AE004439', 'CP003313', 'CP017183', 'NC_004310', 'PPFD00000000', 'CP007500', 'NC_003317', 'CP000046', 'CP003096', 'CP031314', 'MINA01000000', 'NC_002516', 'Gc02016', 'KC344732', 'BX470250', 'MINA00000000', 'CP026503', 'CU928158', 'NC_034156', 'NC_007618', 'CP003097', 'CR377818', 'CP000029', 'NC_009504', 'NC_003318', 'CP031304', 'CP031298', 'KP100338', 'NC_009505']


easier format for json config file:
"target_keys": ["CP031311",
                    "CP020347",
                    "CP020345",
                    "Gs0095506",
                    "NC_007624",
                    "CP011526",
                    "NC_004311",
                    "AM933173",
                    "AE004439",
                    "CP003313",
                    "CP017183",
                    "NC_004310", 
                    "PPFD00000000", 
                    "CP007500", 
                    "NC_003317", 
                    "CP000046", 
                    "CP003096", 
                    "CP031314", 
                    "MINA01000000", 
                    "NC_002516", 
                    "Gc02016", 
                    "KC344732", 
                    "BX470250", 
                    "MINA00000000", 
                    "CP026503", 
                    "CU928158", 
                    "NC_034156", 
                    "NC_007618", 
                    "CP003097", 
                    "CR377818", 
                    "CP000029",
                    "NC_009504", 
                    "NC_003318", 
                    "CP031304",
                    "CP031298", 
                    "KP100338"],