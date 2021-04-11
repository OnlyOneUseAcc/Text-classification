import copy
import time
from collections import defaultdict
import torch
import numpy as np
import gc


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append(f"{k}: {metrics[k] / epoch_samples :.2e}")
    print(f'{phase}: {", ".join(outputs)}')


def train_model(model, optimizer_fn, loss_fn, epochs, dataloaders, device):
    model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_train_loss = 1e10
    best_val_loss = 1e10
    best_epoch = -1

    full_metrics = dict(train_loss=[],
                        val_loss=[],
                        LR=[],
                        time_elapsed=[]
                        )
    for epoch in range(epochs):
        print('-' * 10)
        print(f'Epoch {epoch + 1}/{epochs}')
        since = time.time()

        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            metrics = defaultdict(float)
            epoch_samples = 0

            for ids, masks, labels in dataloaders[phase]:
                ids = ids.to(device)
                masks = masks.to(device)
                labels = labels.to(device)

                optimizer_fn.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(ids, masks)
                    loss = loss_fn(outputs, torch.argmax(labels, dim=1))
                    metrics['loss'] = loss

                    if phase == 'train':
                        loss.backward()
                        optimizer_fn.step()
                epoch_samples += ids.size(0)

            if phase == 'train':
                train_loss = metrics['loss'] / epoch_samples
                full_metrics['train_loss'].append(train_loss)

            else:
                val_loss = metrics['loss'] / epoch_samples
                full_metrics['val_loss'].append(val_loss)

            print_metrics(metrics, epoch_samples, phase)

            if phase == 'validation':
                if 0 < val_loss < best_val_loss:
                    print("saving best model")
                    best_train_loss = train_loss
                    best_val_loss = val_loss
                    best_epoch = epoch
                    best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        full_metrics['time_elapsed'].append(time_elapsed)
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        torch.cuda.empty_cache()
        gc.collect()

    best_loss_dict = dict(best_train_loss=best_train_loss, best_val_loss=best_val_loss)
    print(f'Best epoch: {best_epoch + 1}')
    print(f'best_train_loss: {best_train_loss:.2e}')
    print(f'best_val_loss: {best_val_loss:.2e}')

    model.load_state_dict(best_model_wts)
    return model, best_loss_dict, best_epoch, full_metrics


def model_predict(model, test_dataloader, device):
    prediction_list = []
    labels_list = []
    model.to(device)
    model.eval()

    for idx, (ids, masks, labels) in enumerate(iter(test_dataloader)):
        ids = ids.to(device)
        masks = masks.to(device)
        pred = model(ids, masks)
        for pred_tensor in pred:
            prediction_list.append(pred_tensor.cpu().detach().numpy())
        for true_tensor in labels:
            labels_list.append(true_tensor.cpu().detach().numpy())
        torch.cuda.empty_cache()
        gc.collect()

    return np.array(prediction_list), np.array(labels_list)
