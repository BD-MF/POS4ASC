# -*- coding: utf-8 -*-

import os
import time
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn import metrics

from data_loader import ASCDataLoader

from transformers import AdamW, get_linear_schedule_with_warmup

class Trainer:
    def __init__(self, args):
        self._print_args(args)

    def _print_args(self, args):
        print('>> training arguments:')
        for arg in vars(args):
            print('>> {}: {}'.format(arg, getattr(args, arg)))

    def _train(self, args, model, optimizer, scheduler, train_data_loader, dev_data_loader, state_dict_path):
        best_dev_acc, best_dev_f1 = 0, 0
        best_dev_epoch = 0
        iter_step = 0
        for epoch in range(args.num_train_epochs):
            print('>' * 30 + 'epoch {}'.format(epoch + 1) + '>' * 30)
            for batch in train_data_loader:
                iter_step += 1
                
                model.train()
                optimizer.zero_grad()

                inputs = [
                    batch[col].to(args.device)
                    for col in args.input_fields
                ]
                target = batch['polarity'].to(args.device)
                output = model(inputs)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                if iter_step % args.log_interval == 0:
                    dev_acc, dev_f1 = self._evaluate(args, model, dev_data_loader)
                    print('train loss: {:.4f}, dev acc: {:.4f}, dev f1: {:.4f}'.
                        format(loss.item(), dev_acc, dev_f1))
                    if dev_acc > best_dev_acc:
                        best_dev_acc = dev_acc
                    if dev_f1 > best_dev_f1:
                        print('>> new best model.')
                        best_dev_epoch = epoch
                        best_dev_f1 = dev_f1
                        torch.save(model.state_dict(), state_dict_path)
            
            if epoch - best_dev_epoch >= args.num_patience_epochs:
                print('>> early stop.')
                break
        return best_dev_acc, best_dev_f1

    def _evaluate(self, args, model, data_loader):
        model.eval()
        t_target_all, t_output_all = None, None
        with torch.no_grad():
            for t_batch in data_loader:
                t_inputs = [
                    t_batch[col].to(args.device) for col in args.input_fields
                ]
                t_target = t_batch['polarity'].to(args.device)
                t_output = model(t_inputs)
                if t_target_all is None:
                    t_target_all = t_target
                    t_output_all = t_output
                else:
                    t_target_all = torch.cat((t_target_all, t_target), dim=0)
                    t_output_all = torch.cat((t_output_all, t_output), dim=0)
        acc = metrics.accuracy_score(t_target_all.cpu().numpy(),
                                     torch.argmax(t_output_all, -1).cpu().numpy())
        f1 = metrics.f1_score(t_target_all.cpu().numpy(),
                              torch.argmax(t_output_all, -1).cpu().numpy(),
                              labels=[0, 1, 2],
                              average='macro')
        return acc, f1

    def run(self, args, embedding, train_data, dev_data):
        print('+' * 30 + ' training on {} '.format(args.train_data_name) + '+' * 30)
        for i in range(args.num_repeats):
            print('#' * 30 + ' repeat {} '.format(i + 1) + '#' * 30)

            train_data_loader = ASCDataLoader(
                train_data,
                batch_size=args.batch_size,
                shuffle=True
            )
            dev_data_loader = ASCDataLoader(
                dev_data, 
                batch_size=args.batch_size, 
                shuffle=False
            )

            model = args.model_class(args, embedding).to(args.device)
            
            temp_best_path = os.path.join(args.exp_dir, 'best_ckpt_{}.pt'.format(i))

            if 'bert' in args.model_name:
                no_decay = ['bias', 'LayerNorm.weight']
                grouped_parameters = [
                    {
                        'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                        'weight_decay': args.weight_decay,
                    },
                    {
                        'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
                        'weight_decay': 0.0
                    },
                ]
                optimizer = AdamW(grouped_parameters, lr=args.learning_rate)
                scheduler = get_linear_schedule_with_warmup(optimizer, int(0.05 * args.num_train_epochs * len(train_data_loader)), args.num_train_epochs * len(train_data_loader))
            else:
                optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
                scheduler = None

            self._train(args, model, optimizer, scheduler, train_data_loader, dev_data_loader, temp_best_path)
