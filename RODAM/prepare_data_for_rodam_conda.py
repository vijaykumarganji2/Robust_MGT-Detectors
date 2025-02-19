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


def prepare_ovr_datasets(dataset_path = '../../data/perturbed_data', src_llm_name = 'GPT', tgt_llm_name = 'GPT',
                         storing_data_path = '../../data/RODAM_Data/OVR_Dataset'):
    '''
    This function prepares datasets for the setting 1 in our experiment (i.e OVR[one vs rest]). 
    for example, for 'cmv' dataset the source data is all other datasets except 'cmv' dataset and the target data is 'cmv' dataset.

    Parameters:
        dataset_path : path of perturbed data, (for training rodam/conda model)
        src_llm_name : name of LLM (which text is used for preparing source data)
        tgt_llm_name : name of LLM (which text is used for preparing target data)
        storing_data_path : path, where the prepared data is to be stored.
    
    Note : In our experiment 1, we used the same llm for both source and target.

    Output File Format :
        Each of the train/valid/test jsonl files are split into 'real' (i.e, human-written) & 'fake' (i.e,AI-generated) samples in separate files. each file is a jsonl file; one json object per line, with keys ['text', 'text_perturb', 'label']
    '''

    for target in datasets:
        for file_name in ['train.jsonl', 'valid.jsonl', 'test.jsonl']:
           source_df = pd.DataFrame()
           
           for other in datasets:
           
              if other!= target:
                 df = pd.read_json(f"{dataset_path}/{src_llm_name}_Version/{other}/{file_name}",
                                   orient='records', lines=True)
                 if source_df.shape[0] == 0:
                     source_df = df
                 else:
                     source_df = pd.concat([source_df,df])
                
          
           target_df = pd.read_json(f"{dataset_path}/{tgt_llm_name}_Version/{target}/{file_name}", orient='records',lines=True)

           source_real, source_fake = get_real_fake(source_df)
           target_real, target_fake = get_real_fake(target_df)
           
           source_data_save_path = f'{storing_data_path}/{tgt_llm_name}_Version/{target}_OVR/source_data'
           target_data_save_path = f'{storing_data_path}/{tgt_llm_name}_Version/{target}_OVR/target_data'
           
           os.makedirs(source_data_save_path, exist_ok=True)
           os.makedirs(target_data_save_path, exist_ok=True)
           
           source_real.to_json(f"{source_data_save_path}/real.{file_name}", orient='records', lines=True)
           source_fake.to_json(f"{source_data_save_path}/fake.{file_name}", orient='records', lines=True)
           
           target_real.to_json(f"{target_data_save_path}/real.{file_name}", orient='records', lines=True)
           target_fake.to_json(f"{target_data_save_path}/fake.{file_name}", orient='records', lines=True)
            
        print(f'{target} is Done..')


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


