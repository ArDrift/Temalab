import pandas as pd
from torch.utils.data import DataLoader
from transformers import AdamW, BertTokenizerFast, BertForTokenClassification
import gc
import numpy as np
from torch.utils.data import DataLoader
import torch
import os
import datetime
import time
from transformers import get_linear_schedule_with_warmup
import data_loader
import id_tag_linker as id_tag
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

# If there's a GPU available...
if torch.cuda.is_available():

    # Tell PyTorch to use the GPU.
    device = torch.device("cpu")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

train_dataset = data_loader.loadDataset("dataset/train.csv")
train_dataset.dropna()
print("Number of training samples:" + str(len(train_dataset)))
train_tokens, train_labels = data_loader.convertToLists(train_dataset.token.values,
                                                        train_dataset.label.values, len(train_dataset) / 3)

val_dataset = data_loader.loadDataset("dataset/dev.csv")
val_dataset.dropna()
print("Number of validation samples:" + str(len(val_dataset)))
val_tokens, val_labels = data_loader.convertToLists(val_dataset.token.values,
                                                    val_dataset.label.values, len(val_dataset) / 3)

torch.cuda.empty_cache()
gc.collect()

tokenizer = BertTokenizerFast.from_pretrained("SZTAKI-HLT/hubert-base-cc", model_max_length=512, do_lower_case=False,
                                              num_labels=len(id_tag.label_list))

val_encodings = tokenizer(val_tokens, is_split_into_words=True, return_offsets_mapping=True,
                          padding=True, truncation=True)
train_encodings = tokenizer(train_tokens, is_split_into_words=True, return_offsets_mapping=True,
                            padding=True, truncation=True)


def encode_tags(tags, encodings):
    _labels = id_tag.convertTags2Id(tags)
    encoded_labels = []
    for doc_labels, doc_offset in zip(_labels, encodings.offset_mapping):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset), dtype=int) * -100
        arr_offset = np.array(doc_offset)

        # set labels whose first offset position is 0 and the second is not 0
        try:
            doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = doc_labels
        except:
            print(doc_labels)

        encoded_labels.append(doc_enc_labels.tolist())

    return encoded_labels


val_labels = encode_tags(val_labels, val_encodings)
train_labels = encode_tags(train_labels, train_encodings)


class WNUTDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, _labels):
        self.encodings = encodings
        self.labels = _labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def format_time(elapsed_time):
    # Round to the nearest second.
    elapsed_rounded = int(round(elapsed_time))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(_preds, _labels):
    pred_flat = np.argmax(_preds, 1).flatten()
    labels_flat = _labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


train_encodings.pop("offset_mapping")  # we don't want to pass this to the model
val_encodings.pop("offset_mapping")
train_dataset = WNUTDataset(train_encodings, train_labels)
val_dataset = WNUTDataset(val_encodings, val_labels)

torch.cuda.empty_cache()
gc.collect()

model = BertForTokenClassification.from_pretrained('SZTAKI-HLT/hubert-base-cc', num_labels=len(id_tag.label_list),
                                                   output_attentions=False,
                                                   output_hidden_states=False, )
model.to(device)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)

optim = AdamW(model.parameters(), lr=5e-5)

# Number of training epochs (authors recommend between 2 and 4)
epochs = 3

# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_loader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=0, num_training_steps=total_steps)

print()
print("Training started...")

training_stats = []

total_t0 = time.time()
for epoch in range(epochs):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
    print('Training...')
    t0 = time.time()
    total_loss = 0
    model.train()
    for i, batch in enumerate(train_loader):
        if i % 5 == 0 and not i == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(i, len(train_loader), elapsed))

        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        result = model(input_ids,
                       attention_mask=attention_mask,
                       labels=labels)

        loss = result.loss
        logits = result.logits
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_loader)
    training_time = format_time(time.time() - t0)
    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))

    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    print("")
    print("Running Validation...")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0
    # Evaluate data for one epoch
    for batch in val_loader:
        # Unpack this training batch from our dataloader.
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using
        # the `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids
        #   [1]: attention masks
        #   [2]: labels
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # token_type_ids is the same as the "segment ids", which
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            result = model(input_ids,
                           attention_mask=attention_mask,
                           labels=labels)

        # Get the loss and "logits" output by the model. The "logits" are the
        # output values prior to applying an activation function like the
        # softmax.
        loss = result.loss
        logits = result.logits

        # Accumulate the validation loss.
        total_eval_loss += loss.item()

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences, and
        # accumulate it over all batches.
        total_eval_accuracy += flat_accuracy(logits, label_ids)

        # Report the final accuracy for this validation run.
    avg_val_accuracy = total_eval_accuracy / len(val_loader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(val_loader)

    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)

    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

print("")
print("Training complete!")

print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

output_dir = './model_save/'

# Create output directory if needed
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Saving model to %s" % output_dir)

# Save a trained model, configuration and tokenizer using `save_pretrained()`.
# They can then be reloaded using `from_pretrained()`
model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
model_to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

pd.set_option('precision', 2)

df_stats = pd.DataFrame(data=training_stats)

df_stats = df_stats.set_index('epoch')
# Display the table.
print(tabulate(df_stats, headers=["Epoch", "Training Loss", "Valid. Loss", "Training Time", "Valid. Time"]))

# Use plot styling from seaborn.
sns.set(style='darkgrid')

# Increase the plot size and font size.
sns.set(font_scale=1.5)
plt.rcParams["figure.figsize"] = (12, 6)

# Plot the learning curve.
plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")

# Label the plot.
plt.title("Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.xticks([1, 2, 3, 4])

plt.show()
