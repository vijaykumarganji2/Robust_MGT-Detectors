import pandas as pd
from sklearn.metrics import *
from transformers import RobertaTokenizer
import torch
from tqdm.auto import tqdm

from models import RobertaForContrastiveClassification, T5Paraphraser

def predict_label(model, tokenizer, text, device):
    # Tokenize input text
    # try:
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs['logits']


    # Get predicted label and its confidence score
    _, predicted_label = torch.max(logits, 1)
    predicted_label = predicted_label.item()


    # Calculate confidence score -contributed @ ObaidTamboli
    confidence_score = logits[0][predicted_label].item()

    return predicted_label, confidence_score
   



tqdm.pandas(desc = 'Inferencing..')
def get_performace(test_dataset, model, tokenizer, device, prediction_save_path, prediction_exist = False):

  if not prediction_exist:
    predictions = test_dataset['text'].progress_apply(lambda x: predict_label(model, tokenizer, x, device))
    test_dataset['predict_label'], test_dataset['confidence_score'] = zip(*predictions)

    # save the predictions
    # predictions_path = f"./prediction/model_{}_data_{}"
    test_dataset.to_csv(f'{prediction_save_path}', index=False)
    print(f'Prediction Saved to {prediction_save_path}')

  test_dataset = pd.read_csv(f'{prediction_save_path}')
  y_test, y_pred = test_dataset['label'].values, test_dataset['predict_label'].values

  print('Results: Human Is 1, AI is 0')
  print('Accuracy:', round(accuracy_score(y_test, y_pred),2))
  print('MCC Score:', round(matthews_corrcoef(y_test, y_pred),2))
  print('F1 Score:', round(f1_score(y_test, y_pred, average = 'macro'),2))
  print('Human Precision:', round(precision_score(y_test, y_pred, pos_label=1),2))
  print('AI Precision:', round(precision_score(y_test, y_pred, pos_label=0),2))
  print('Human Recall:', round(recall_score(y_test, y_pred, pos_label=1),2))
  print('AI Recall:', round(recall_score(y_test, y_pred, pos_label=0),2))



max_sequence_length = 512
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

########## Initialization Section##########
# Set Model and Dataset Paths
dataset = 'cmv'
llm_name = 'GPT'
model_save_path = f'/home/tarun/MTP/Trained_Models/RODAM/setting1/{llm_name}/rodam_{dataset}_{llm_name.lower()}_ovr.pt'
model_save_path = f'/home/tarun/MTP/Trained_Models/RODAM/medquad/checkpoint_rodam_train_valid_source_test_target_medquad.pt'
test_dataset_path = '/home/tarun/MTP/data/HealthCare2K/bioasq/bioasq_2k.csv'
prediction_save_path = '/home/tarun/MTP/data/HealthCare2K/bioasq/bioasq_2k_predictions_of_train_valid_source_test_target_medquad_model.csv'

#Load Dataset
test_dataset = pd.read_csv(test_dataset_path)

# Load Model
checkpoint = torch.load(model_save_path , map_location=torch.device(device))
model = RobertaForContrastiveClassification.from_pretrained('roberta-base').to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print('Model Loaded')

#Get Predictions
get_performace(test_dataset, model, tokenizer, device, prediction_save_path, prediction_exist = False)

