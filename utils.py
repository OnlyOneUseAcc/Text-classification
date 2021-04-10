import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import RobertaTokenizer
from datasets import load_dataset
import pandas as pd


def get_dataset():
    dataset = load_dataset("tweets_hate_speech_detection")
    data = pd.DataFrame(data=dataset['train'], columns=['tweet', 'label'])
    return data


def get_dataloaders(data, batch_size):
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)
    data['tweet'] = data['tweet'].apply(lambda x: x.replace('@user', 'user_link'))

    train_data = data.sample(frac=0.8, random_state=42)
    test_data = data.drop(train_data.index)

    validation_data = train_data.sample(frac=(1 / 8), random_state=42)
    train_data = train_data.drop(validation_data.index)

    validation_data = validation_data.reset_index(drop=True)
    train_data = train_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)

    dataset_list = [train_data, validation_data, test_data]
    keys = ['train', 'validation', 'test']
    dataloaders = {}

    for i, phase in enumerate(keys):
        input_ids = []
        input_masks = []

        for tweet in dataset_list[i]['tweet']:
            token_data = tokenizer(tweet,
                                   max_length=64,
                                   add_prefix_space=True,
                                   truncation=True,
                                   padding='max_length',
                                   add_special_tokens=True)
            input_ids.append(token_data['input_ids'])
            input_masks.append(token_data['attention_mask'])

        tensor_dataset = TensorDataset(torch.tensor(input_ids),
                                       torch.tensor(input_masks),
                                       torch.nn.functional.one_hot(
                                           torch.tensor(dataset_list[i]['label'].values),
                                           num_classes=2)
                                       )

        dataloaders[keys[i]] = DataLoader(tensor_dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
    return dataloaders