from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    BertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
import torch
import numpy as np
import time, datetime
import random

from sklearn.metrics import log_loss

from transformers import WEIGHTS_NAME, CONFIG_NAME
import os


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    # Round to the nearest second
    elapsed_rounded = int(round(elapsed))

    # hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


MODEL_GROUPS = dict(
    bert=BertForSequenceClassification,
    distilbert=DistilBertForSequenceClassification
)

optimizers = dict(
    adam=AdamW
)

schedulers = dict(
    get_linear_schedule_with_warmup
)

eval_metrics = dict(
    accuracy=flat_accuracy
)


class SequenceClassifierModel:

    @property
    def device(self):
        if torch.cuda.is_available():
            return torch.device('cuda')
            print(f'There are {torch.cuda.device_count()} GPUs available')
            print('We will use the GPU:', torch.cuda.get_device_name(0))
        else:
            print('No GPU available, using the cpu instead...')
            return torch.device('cpu')

    @property
    def model_params(self):
        return list(self.model.parameters())

    @property
    def total_steps(self):
        return len(self.train_data) * self.epochs

    def __init__(
            self,
            this_project_name,
            tr_model_group,
            optimizer,
            scheduler,
            num_labels,
            train_data,
            val_data,
            tr_model_id='bert-base-uncased',
            epochs=2,
            lr=2e-5,
            seed=100,
            eval_metric='accuracy',
            output_dir='./models/',
            output_attentions=False,
            output_hidden_states=False

    ):
        """

        :param tr_pretrained_model_name_or_path:
        :param optimizer:
        :param scheduler:
        :param num_labels:
        :param train_data:
        :param val_data:
        :param tr_model_id:
        :param epochs:
        :param seed:
        :param eval_metric:
        :param output_dir:
        :param output_attentions:
        :param output_hidden_states:
        """
        self.this_project_name = this_project_name
        self.model = tr_model_group.from_pretrained(
            pretrained_model_name_or_path=tr_model_id,
            num_labels=num_labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )

        self.train_data = train_data
        self.val_data = val_data
        self.num_labels = num_labels
        self.epochs = epochs
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.seed = seed
        self.eval_metric = eval_metrics[eval_metric]
        self.model_output_dir = output_dir
        self.lr = lr
        self.optimizer = AdamW(
            params=self.model_params,
            lr=self.lr
        )
        self.scheduler = scheduler(
            optimizer=self.optimizer,
            num_warmup_steps=5,
            num_training_steps=self.total_steps
        )
        if self.device.type in ('cuda', 'gpu'):
            self.model.cuda()

    def train(self):
        """
        Performs training and evaluation
        :return:
        """

        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        loss_values = []

        train_steps = len(self.train_data) - 1
        eval_steps = len(self.val_data) - 1
        # 1. Iterating over the number of epochs
        for epoch in range(0, self.epochs):
            # =============================
            # Training
            # =============================

            print(f'=============== Epoch {epoch + 1} / {self.epochs} ======')
            print('Training...')

            t0 = time.time()
            total_loss, train_accuracy = self._run_training()

            # Calculate the average loss over the training data
            average_train_loss = total_loss / len(self.train_data)

            loss_values.append(average_train_loss)

            print(" Train Accuracy: {0:.2f}".format(train_accuracy / train_steps))
            print(f'Average training loss: {average_train_loss}')
            print(f'Training epoch took: {format_time(time.time() - t0)}')

            t0 = time.time()

            eval_accuracy = self._run_validation()

            print(f' Accuracy: {eval_accuracy / eval_steps}')
            print(f' Validation took: {format_time(time.time() - t0)}')

        print()
        print('Training Complete')

        self.save_model()

    def _run_training(self):
        # 2. Iterating over the batches in train_data_loader👇🏾
        self.model.train()

        total_loss = 0  # Reset the total loss for this epoch.

        train_accuracy = 0.0

        t1 = time.time()

        for step, batch in enumerate(self.train_data):
            # Progress update every 40 batches
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes
                elapsed = format_time(time.time() - t1)

                # Report progress
                print('Batch {:>5,} of {:>5,}. Elapsed: {:}.'.format(
                    step, len(self.train_data), elapsed)
                )  # Unpack this training batch from our dataloader. ')

                # Copy tensor to GPU
                # batch contains three tensors
                # input_ids, attention_masks and labels

            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_input_labels = batch[2].to(self.device)

            self.model.zero_grad()  # setting batch gradients to 0

            outputs = self.model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                labels=b_input_labels
            )
            # The call to `model` always returns a tuple, so we need to pull the
            # loss value out of the tuple.
            # outputs = (loss, logits)

            loss = outputs[0]

            tlogits = outputs[1].detach().cpu().numpy()
            tlabel_ids = b_input_labels.detach().cpu().numpy()

            tmp_tr_acc = flat_accuracy(tlogits, tlabel_ids)
            train_accuracy += tmp_tr_acc

            # 👇🏾Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            total_loss += loss.item()

            loss.backward()  # backward pass to calculate gradients

            # 👇🏾 cliping grads to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()
            self.scheduler.step()  # Update the learning rate

        return total_loss, train_accuracy

    def _run_validation(self):
        # =====================================
        #              Validation
        # =====================================

        # After the completion of each training epoch, measure your performance
        # on the validation set
        print()
        print('Running Validation...')

        # put the model in evaluation mode -- the dropout layers behave differently
        # during evaluation

        self.model.eval()

        # Tracking variables

        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        # Evaluate data for one epoch

        for batch in self.val_data:
            batch = tuple(t.to(self.device) for t in batch)

            # Unpack inputs from dataloader
            b_input_ids, b_input_mask, b_labels = batch

            # Telling the model not to compute or store gradients,
            # saving memory and speeding up validation
            with torch.no_grad():
                # Foreward pass, calculate logit predictions
                # Only returning logits instead of loss because
                # we haven't provided labels

                # token_type_ids is the same as the segment_ids which
                # indicates which token belongs to sentence 1 or 2 in
                # 2 sentence tasks

                outputs = self.model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask
                )  # labels are not passed here in validation

                # Get the "logits" output by the model. The "logits" are the output
                # values prior to applying an activation function like softmax

                logits = outputs[0]

                # Move logits and labels to CPU

                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                # Calculate the accuracy for this batch of test sentences.
                tmp_eval_accuracy = flat_accuracy(logits, label_ids)
                eval_accuracy += tmp_eval_accuracy

                # track the number of batches
                nb_eval_steps += 1
        return eval_accuracy

        # Report the final accuracy for this validation run

    def save_model(self):

        model_full_path = os.path.join(self.model_output_dir, self.this_project_name)
        if not os.path.exists(model_full_path):
            os.mkdir(model_full_path)

        output_model_file = os.path.join(model_full_path, WEIGHTS_NAME)
        output_config_file = os.path.join(model_full_path, CONFIG_NAME)

        self.model.to('cpu')

        print(self.model_output_dir)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)

        print(f'model saved to {output_model_file}')
