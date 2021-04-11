import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import RobertaTokenizer
from datasets import load_dataset
import pandas as pd
import numpy as np
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score


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
    dataloaders = defaultdict()

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


def get_cm(pred_np, labels_np):
    cm = np.zeros((2, 2), dtype=np.int)
    pred_np = np.argmax(pred_np, axis=1)
    labels_np = np.argmax(labels_np, axis=1)
    for p, l in zip(pred_np, labels_np):
        cm[l][p] += 1
    return cm


def show_cm(cm):
    plt.figure(figsize=(6, 6))
    plt.tick_params(axis='both', which='major', labelsize=10,
                    labelbottom=False, bottom=False,
                    top=False, labeltop=True)

    ax = sns.heatmap(cm, annot=True)
    ax.xaxis.set_label_position('top')


def pr_curve(y_true, y_pred):
    precision = dict()
    recall = dict()
    average_precision = dict()
    labels = []

    for i in range(2):
        precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], y_pred[:, i])
        average_precision[i] = average_precision_score(y_true[:, i], y_pred[:, i])
        labels.append('Precision-recall for class {0} (area = {1:0.4f})'
                      ''.format(i, average_precision[i]))

    plt.figure(figsize=(8, 6))
    plt.step(recall[0], precision[0])
    plt.step(recall[1], precision[1])
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.title('Precision-recall curve')
    plt.legend(labels)
    plt.show()
