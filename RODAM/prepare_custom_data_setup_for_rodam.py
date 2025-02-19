import pandas as pd
import os


datasets = ['xsum','yelp','hswag','roct','eli5','sci_gen','tldr','wp','squad','cmv']


def get_real_fake(df):
   ''''
   spilt the dataframe into 'real' (i.e. human-written [label = 1]) and 'fake' (i.e. AI-Generated [label = 0]
   '''
   real = df[df['label']==1]
   fake = df[df['label']==0]
   
   return real, fake


def prepare_ovr_datasets(dataset_path = '../../data/HealthCare/perturbed_bioasq_gpt', storing_data_path = '../../data/RODAM_Data/bioasq_gpt_setup/train_valid_source_test_target'):
    for file_name in ['train.jsonl', 'valid.jsonl', 'test.jsonl']:
        train_df = pd.read_json(f"{dataset_path}/bioasq_gpt_train.jsonl", orient='records',lines=True)
        valid_df = pd.read_json(f"{dataset_path}/bioasq_gpt_valid.jsonl", orient='records',lines=True)
        test_df = pd.read_json(f"{dataset_path}/bioasq_gpt_test.jsonl", orient='records',lines=True)
    
        if file_name == 'train.jsonl':
            source_df = pd.concat([train_df, valid_df])
            target_df = test_df.copy()
        if file_name == 'valid.jsonl':
            source_df = valid_df.copy()
            target_df = test_df.copy()
        if file_name == 'test.jsonl':
            source_df = valid_df.copy()
            target_df = test_df.copy()
        
        
        source_real, source_fake = get_real_fake(source_df)
        target_real, target_fake = get_real_fake(target_df)
           
        source_data_save_path = f'{storing_data_path}/source_data'
        target_data_save_path = f'{storing_data_path}/target_data'
           
        os.makedirs(source_data_save_path, exist_ok=True)
        os.makedirs(target_data_save_path, exist_ok=True)
           
        source_real.to_json(f"{source_data_save_path}/real.{file_name}", orient='records', lines=True)
        source_fake.to_json(f"{source_data_save_path}/fake.{file_name}", orient='records', lines=True)
           
        target_real.to_json(f"{target_data_save_path}/real.{file_name}", orient='records', lines=True)
        target_fake.to_json(f"{target_data_save_path}/fake.{file_name}", orient='records', lines=True)
            
    print(f'Data Preparation is Completed..')


def prepare_single_datasets(dataset_path = '../perturbed_data', src_llm_name = 'GPT', tgt_llm_name = 'LLAMA',
                            storing_data_path = '../rodam_data/Single_Dataset'):
    '''
    This function prepares datasets for the setting 2 in our experiment (i.e Single [one vs one]). 
    for example, for 'cmv' dataset the source data is 'cmv [GPT]' and the target data is 'cmv [LLAMA]'.

    Parameters:
        dataset_path : path of perturbed data, (for training rodam/conda model)
        src_llm_name : name of LLM (which text is used for preparing source data)
        tgt_llm_name : name of LLM (which text is used for preparing target data)
        storing_data_path : path, where the prepared data is to be stored.
    
    Note : In our experiment 2, we used different llm for both source and target, but use the same data.

    Output File Format :
        Each of the train/valid/test jsonl files are split into 'real' (i.e, human-written) & 'fake' (i.e,AI-generated) samples in separate files. each file is a jsonl file; one json object per line, with keys ['text', 'text_perturb', 'label']
    '''

    for dataset in datasets:
        for file_name in ['train.jsonl', 'valid.jsonl', 'test.jsonl']:
            
            source_df = pd.read_json(f"{dataset_path}/{src_llm_name}_Version/{dataset}/{file_name}",
                                     orient='records', lines=True)
            
            
            target_df = pd.read_json(f"{dataset_path}/{tgt_llm_name}_Version/{dataset}/{file_name}",
                                     orient='records',lines=True)
                
            source_real, source_fake = get_real_fake(source_df)
            target_real, target_fake = get_real_fake(target_df)

            source_data_save_path = f'{storing_data_path}/{src_llm_name}_{tgt_llm_name}_{dataset}_Single/source_data'
            target_data_save_path = f'{storing_data_path}/{src_llm_name}_{tgt_llm_name}_{dataset}_Single/target_data'

            os.makedirs(source_data_save_path, exist_ok=True)
            os.makedirs(target_data_save_path, exist_ok=True)

            source_real.to_json(f"{source_data_save_path}/real.{file_name}", orient='records', lines=True)
            source_fake.to_json(f"{source_data_save_path}/fake.{file_name}", orient='records', lines=True)

            target_real.to_json(f"{target_data_save_path}/real.{file_name}", orient='records', lines=True)
            target_fake.to_json(f"{target_data_save_path}/fake.{file_name}", orient='records', lines=True)
                
        print(f'{dataset} is Done..')




prepare_ovr_datasets()

# To get Dataset for Setting 1
#prepare_ovr_datasets(dataset_path = '../perturbed_data', src_llm_name = 'BLOOM7B', tgt_llm_name = 'BLOOM7B',
#                     storing_data_path = '../rodam_data/OVR_Dataset')


# To get Dataset for Setting 2
#prepare_single_datasets(dataset_path = '../perturbed_data', src_llm_name = 'GPT', tgt_llm_name = 'LLAMA',
#                        storing_data_path = '../rodam_data/Single_Dataset')


