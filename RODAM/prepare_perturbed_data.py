import pandas as pd
import os
from synonym_replacement import augment_dataset


datasets = ['xsum','yelp','hswag','roct','eli5','sci_gen','tldr','wp','squad','cmv']



def perturb_datasets(dataset_path = '../../data/All_LLM_Version_Dataset', llm_name = 'LLAMA', perturb_dataset_path = '../../data/perturbed_data'):
  '''
  This function generates the perturbed text for each dataset of the given llm_version text

  Parameters : 
    dataset_path : path where data is data is available, which is to be perturb
    llm_name : The name of LLM, which text is going to perturb, available llm texts are ['BLOOM7B', 'OPT30B',  'FLAN-T5', 'NeoX20B', 'GPT',  'LLAMA']
    perturb_dataset_path : path where the perturbed data is going to store.
  
  File Format :
    Input File Format : jsonl file ; one json object per line, with keys ['text', 'label']
    Output File Format : jsonl file; one json object per line, with 3 keys: ['text', 'text_perturb', 'label']

  '''

  for dataset in datasets:

    for file_name in ['train.csv', 'valid.csv', 'test.csv']:
      df = pd.read_csv(f'{dataset_path}/{llm_name}_Version/{dataset}/{file_name}')
      
      data_save_path = f'{perturb_dataset_path}/{llm_name}_Version/{dataset}'
      os.makedirs(data_save_path, exist_ok=True)

      file_pre = file_name[:-4]
      augment_dataset(df, output_filepath = f"{data_save_path}/{file_pre}.jsonl", percentage=0.1)



if __name__ == '__main__':
  perturb_datasets()
