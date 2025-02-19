import pandas as pd
import os
from synonym_replacement import augment_dataset



def perturb_datasets(dataset_path = '../../data/HealthCare/pubmed', perturb_dataset_path = '../../data/HealthCare/perturbed_pubmed_gpt'):
    for file_name in ['pubmed_gpt_train.csv', 'pubmed_gpt_valid.csv', 'pubmed_gpt_test.csv']:
      df = pd.read_csv(f'{dataset_path}/{file_name}')
      
      data_save_path = f'{perturb_dataset_path}'
      os.makedirs(data_save_path, exist_ok=True)

      file_pre = file_name[:-4]
      augment_dataset(df, output_filepath = f"{data_save_path}/{file_pre}.jsonl", percentage=0.1)



if __name__ == '__main__':
  perturb_datasets()
