import config

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from utils import AttnLabelConverter, Averager

from coco_dataset import get_synth_datasets, get_cocotext_datasets, get_cocotext_single_image_data

def get_dataset(dataset='cocotext'):
    if dataset == 'cocotext':
        train_dataloader, val_dataloader = get_cocotext_datasets()
        return train_dataloader, val_dataloader
    elif dataset == 'synth':
        train_dataloader, val_dataloader = get_synth_datasets()
        return train_dataloader, val_dataloader
    elif dataset == 'cocotext_single_image_val':
        val_data = get_cocotext_single_image_data(return_loader=False)
        return val_data

def train(model, dataset, validation_steps=100, iteration_limit=None):
    import torch.optim as optim

    train_dataloader, validation_dataloader = get_dataset(dataset)

    converter = AttnLabelConverter(config.CHARS)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(config.PRIMARY_DEVICE) 

    loss_avg = Averager()

    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))

    optimizer = optim.AdamW(filtered_parameters, lr=0.0001)
    #scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.1)

    df = pd.DataFrame(columns=['iter','cost_avg','val_acc','train_acc'])

    start_iter = 0
    #start_time = time.time()
    iteration = start_iter
    torch.cuda.empty_cache()
    epochs = config.EPOCHS

    print('--- Training for ' + str(epochs) + ' epochs. Number of parameters:', sum(params_num))

    val_acc = validate(model, validation_dataloader, print_samples=True)


    df = df.append({'iter': '0', 'cost_avg':'n/a', 'val_acc':val_acc, 'train_acc':'n/a'}, ignore_index=True)

    print(df)

    train_correct = 0
    total = 0

    best_accuracy = config.MODEL_SAVE_THRESHOLD

    for epoch in range(config.EPOCHS):

        model.train()
        epoch_cost = 0
        print('  - Epoch: ' + str(epoch+1))


        for image, text, overlap, scene in tqdm(train_dataloader):

            # Put samples on devices
            image = image.to(config.PRIMARY_DEVICE)
            overlap = overlap.to(config.PRIMARY_DEVICE)   
            scene = scene.to(config.PRIMARY_DEVICE)

            # Encode text (ground truth) to indicies
            encoded_text, _ = converter.encode(text, batch_max_length=config.MAX_TEXT_LENGTH)

            # Pass samples through model
            preds = model(image, encoded_text[:, :-1], overlap, scene)

            target = encoded_text[:, 1:]  # without [GO] Symbol
            cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))
            model.zero_grad()
            cost.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            optimizer.step()
            loss_avg.add(cost)
            epoch_cost += cost.item()

            # Decode predictions
            length_for_pred = torch.IntTensor([config.MAX_TEXT_LENGTH] * len(image)).to(config.PRIMARY_DEVICE)
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)

            # Calculate training accuracy
            # print(zip(text, preds_str))
            for text, preds in zip(text, preds_str):
                pred_EOS = preds.find('[s]')
                pred = preds[:pred_EOS]
                if(text == str(pred)):
                    train_correct += 1

                total += 1

            train_acc = round(train_correct*100/total,5)


            iteration += 1

            # Validation
            if iteration % validation_steps == 0:
                val_acc = validate(model, validation_dataloader, print_samples=True)
                print(config.EXPERIMENT)
                print('  - iter ' + str(iteration) + ':', str(val_acc) +'% | Best:' + str(best_accuracy) + '%')

                if val_acc > best_accuracy:
                    print('  - New best model')

                    df = df.append({'iter': iteration, 'cost_avg':0, 'val_acc':val_acc, 'train_acc':train_acc}, ignore_index=True)
                    df.to_csv('./results/' + config.EXPERIMENT + '_training_log.csv', index=False)

                    best_accuracy = val_acc
                    results_path = './results/models/' + config.EXPERIMENT
                    torch.save(model.state_dict(), results_path + '.pt')

                    train_correct = 0
                    total = 0
                print(df)
            if iteration_limit:
                if iteration == iteration_limit:
                    print('--- Iteration limit reach:', iteration)

            

    print('--- Finished Training')

def validate(model, validation_dataloader, print_samples=False):
    print('  - Running Validation')
    model.eval()

    total = 0
    correct = 0

    converter = AttnLabelConverter(config.CHARS)

    with torch.no_grad():
        for image, text, overlap, scene in tqdm(validation_dataloader):
            # Put samples on device
            image = image.to(config.PRIMARY_DEVICE)
            overlap = overlap.to(config.PRIMARY_DEVICE)
            scene = scene.to(config.PRIMARY_DEVICE)

            length_for_pred = torch.IntTensor([config.MAX_TEXT_LENGTH] * len(image)).to(config.PRIMARY_DEVICE)

            # Zero tensor for target validation
            text_for_pred = torch.LongTensor(config.BATCH_SIZE, config.MAX_TEXT_LENGTH + 1).fill_(0).to(config.PRIMARY_DEVICE)

            # Pass samples through model
            preds = model(image, text_for_pred, overlap, scene, is_train=False)

            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)

            if print_samples:
                print('  - Ground truth:', text[0])
                print('  - Prediction:  ', preds_str[0], '\n\n')

            for text, pred in zip(text, preds_str):

                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])

                if text == str(pred):
                    correct += 1

                total += 1

    return round(correct*100/total,5)

def evaluate(model, print_sem=False):
    print('--- Running Evaluation')
    model.eval()

    from utils import AttnLabelConverter
    converter = AttnLabelConverter(config.CHARS)

    correct = 0
    total = 0

    val_data = get_dataset('cocotext_single_image_val')
    print('  -', len(val_data), 'images loaded')

    with open('./annotations/features/' + config.SEMANTIC_SOURCE.lower() + '_classes.txt') as f:
        obj_class_labels = f.read().splitlines()

    with open('./results/base_error_ids.txt') as f:
        base_errors_ids = f.read().splitlines()

    with torch.no_grad():
        for anno, image, label, overlap, scene in tqdm(val_data):
            if str(anno) in base_errors_ids:
                text_in = torch.LongTensor(config.BATCH_SIZE, config.MAX_TEXT_LENGTH + 1).fill_(0).to(config.PRIMARY_DEVICE)

                if print_sem:
                    if config.SEMANTIC_SOURCE.lower() == 'overlap':
                        overlap_list = list(overlap.numpy())
                        tags = [obj_class_labels[int(i)-1] for i in overlap_list if i != 0]
                    elif config.SEMANTIC_SOURCE.lower() == 'scene':
                        scene_list = list(scene.numpy())
                        tags = [obj_class_labels[int(i)-1] for i in scene_list if i != 0]
                    else:
                        raise Exception('training_functions.py Evaluation error', config.SEMANTIC_SOURCE, 'not recognised')

                    print(tags)
                

                # Add batch dimension
                image = image.unsqueeze(0).to(config.PRIMARY_DEVICE)
                overlap = overlap.unsqueeze(0).to(config.PRIMARY_DEVICE)
                scene = scene.unsqueeze(0).to(config.PRIMARY_DEVICE)

                # Pass through model
                pred = model(input=image, text=text_in, overlap=overlap, scene=scene, is_train=False)

                length_for_pred = torch.IntTensor([config.MAX_TEXT_LENGTH]).to(config.PRIMARY_DEVICE)

                _, pred_index = pred.max(2)
                pred_str = converter.decode(pred_index, length_for_pred)[0]
                pred_str = pred_str[:pred_str.find('[s]')]

                
                if label == pred_str:
                    correct += 1
                    print(label, pred_str)
                    print(round(correct*100/total,3))
                total += 1

    print('Correct:', correct)
    print('Total:  ', total)