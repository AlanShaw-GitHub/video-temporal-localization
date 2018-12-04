import sys
sys.path.append('..')
from absl import app
from dataloader import Loader
from config import config, device
import dataloader
import torch
import math
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
import pickle
import os
from models import Model
import numpy as np


class Trainer(object):
    def __init__(self, config):

        self.params = config
        self.device = device
        self.word2index, self.index2word, self.embeddings = pickle.load(open(config.data_pickle, 'rb'))
        train_dataset = Loader(config, config.p_train_data)
        val_dataset = Loader(config, config.p_val_data)
        test_dataset = Loader(config, config.p_test_data)

        self.model = Model(config, self.embeddings).to(self.device)

        self.train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)
        self.val_loader = DataLoader(dataset=val_dataset, batch_size=config.batch_size, shuffle=False)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size, shuffle=False)

        params = filter(lambda param: param.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.Adam(lr=config.learning_rate, betas=(config.beta1, config.beta2), eps=1e-7, weight_decay=3e-7, params=params)

        # lr = config.learning_rate
        # base_lr = 1.0
        # params = filter(lambda param: param.requires_grad, slef.model.parameters())
        # optimizer = torch.optim.Adam(lr=base_lr, betas=(config.beta1, config.beta2), eps=1e-7, weight_decay=3e-7, params=params)
        # cr = lr / math.log2(config.lr_warm_up_num)
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        #                                         lr_lambda=lambda ee: cr * math.log2(ee + 1) if ee < config.lr_warm_up_num else lr)


        self.model_path = os.path.join(self.params.cache_dir)
        if not os.path.exists(self.model_path):
            print('create path: ', self.model_path)
            os.makedirs(self.model_path)

        self.best_model = None
        self.lr_epoch = 0
        # When iteration starts, queue and thread start to load data from files.


    def train(self):
        print('Trainnning begins......')
        best_epoch_acc = 0
        best_epoch_id = 0

        print('=================================')
        print('Model Params:')
        print(config.flag_values_dict())
        print('=================================')

        self.evaluate(self.train_loader)

        for i_epoch in range(self.params.max_epoches):

            self.model.train()

            t_begin = time.time()
            avg_batch_loss = self.train_one_epoch(i_epoch)
            t_end = time.time()
            print('Epoch %d ends. Average loss %.3f. %.3f seconds/epoch' % (i_epoch, avg_batch_loss, t_end - t_begin))

            if i_epoch % self.params.evaluate_interval == 0 and i_epoch != 0:
                print('=================================')
                print('Overall evaluation')
                # print('=================================')
                # print('train set evaluation')
                # train_acc = self.evaluate(self.train_loader)
                print('=================================')
                print('valid set evaluation')
                valid_acc = self.evaluate(self.train_loader)
                print('=================================')
                print('test set evaluation')
                # test_acc = self.evaluate(self.test_loader)
                print('=================================')
            else:
                print('=================================')
                print('valid set evaluation')
                valid_acc = self.evaluate(self.train_loader)
                print('=================================')

            if valid_acc > best_epoch_acc:
                best_epoch_acc = valid_acc
                best_epoch_id = i_epoch
                print('Saving new best model...')
                timestamp = time.strftime("%m%d%H%M%S", time.localtime())
                self.best_model = self.model_path + timestamp + '.pt'
                torch.save(self.model.state_dict(), self.best_model)
            else:
                if i_epoch - best_epoch_id >= self.params.early_stopping:
                    print('Early stopped. Best loss %.3f at epoch %d' % (best_epoch_acc, best_epoch_id))
                    break

        print('=================================')
        print('Evaluating best model in file', self.best_model, '...')
        if self.best_model is not None:
            self.model.load_state_dict(torch.load(self.best_model))
            self.evaluate(self.test_loader)
        else:
            print('ERROR: No checkpoint available!')

    def train_one_epoch(self, i_epoch):
        loss_sum = 0
        t1 = time.time()
        for i_batch, (frame_vecs, frame_n, ques, ques_n, start_frame, end_frame) in enumerate(self.train_loader):
            frame_vecs = frame_vecs.to(self.device)
            ques = ques.to(self.device)

            # Forward pass
            p1, p2 = self.model(frame_vecs, ques)
            # start_frame = torch.tensor([30] * config.batch_size)
            y1, y2 = start_frame.to(device), end_frame.to(device)
            y1.requires_grad = False
            y2.requires_grad = False
            loss1 = F.nll_loss(p1, y1, reduction='elementwise_mean')
            loss2 = F.nll_loss(p2, y2, reduction='elementwise_mean')
            loss = (loss1 + loss2) / 2

            # Backward and optimize
            self.optimizer.zero_grad()
            loss1.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 1)
            self.optimizer.step()

            self.lr_epoch += 1
            loss_sum += float(loss)

            # if i_batch == 20:
            #     break

            if i_batch % self.params.display_batch_interval == 0 and i_batch != 0:
                p1, p2 = self.model(frame_vecs, ques,True)
                print(p1[0])
                # print(p2[0])
                print(loss1, loss2)
                print('predict',p1[0].argmax(), p2[0].argmax(),p1[0].max(), p2[0].max())
                print('truth',y1[0], y2[0], p1[0][y1[0]], p2[0][y2[0]])
                t2 = time.time()
                print('Epoch %d, Batch %d, loss = %.4f, %.3f seconds/batch' % ( i_epoch, i_batch, loss_sum / i_batch ,(t2 - t1) / self.params.display_batch_interval))
                t1 = t2

            if self.lr_epoch > 0 and self.lr_epoch % 3000 == 0:
                self.adjust_learning_rate()

        avg_batch_loss = loss_sum / i_batch

        return avg_batch_loss

    def evaluate(self, data_loader):
        # IoU_thresh = [0.5,0.7]
        # top1,top5

        IoU_thresh = [0.1, 0.3, 0.5, 0.7]
        all_correct_num_topn_IoU = np.zeros(shape=[1,4],dtype=np.float32)
        mIoU = 0
        all_retrievd = 0.0

        self.model.eval()
        t = time.time()
        for i_batch, (frame_vecs, frame_n, ques, ques_n, start_frame, end_frame) in enumerate(data_loader):
            # if i_batch == 20:
            #     break
            frame_vecs = frame_vecs.to(self.device)
            ques = ques.to(self.device)
            batch_size = len(frame_vecs)

            # Forward pass
            p1, p2 = self.model(frame_vecs, ques)
            y1, y2 = start_frame.to(device), end_frame.to(device)
            for i in range(batch_size):
                predict_windows = [p1[i].argmax(),p2[i].argmax()]
                gt_windows = [y1[i],y2[i]]
                result = self.calculate_IoU(predict_windows, gt_windows)
                mIoU += result
                for j in range(len(IoU_thresh)):
                    if result.__float__() >= IoU_thresh[j]:
                        # print(result.__float__())
                        # print(predict_windows, gt_windows)
                        all_correct_num_topn_IoU[0][j] += 1.0

            all_retrievd += batch_size
        t = time.time() - t
        print('time ', t, t / all_retrievd, all_retrievd)
        avg_correct_num_topn_IoU = all_correct_num_topn_IoU / all_retrievd
        print('=================================')
        print(all_correct_num_topn_IoU, avg_correct_num_topn_IoU)
        print('=================================')

        acc = avg_correct_num_topn_IoU[0,2]

        return acc
        
    def calculate_IoU(self, i0, i1):
        if i0[0] == i0[1]:
            i0[0] -= 0.5
            i0[1] += 0.5
        union = (min(i0[0], i1[0]), max(i0[1], i1[1]))
        inter = (max(i0[0], i1[0]), min(i0[1], i1[1]))
        if union[1].__float__() - union[0].__float__() == 0:
            return 0
        iou = 1.0 * (inter[1].__float__() - inter[0].__float__()) / (union[1].__float__() - union[0].__float__())
        if iou < 0:
            iou = 0
        return iou

    def adjust_learning_rate(self, decay_rate=0.8):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay_rate


def main(_):
    if config.mode == "train":
        trainer = Trainer(config)
        trainer.train()
    elif config.mode == "preprocess":
        dataloader.preprocess(_)
    elif config.mode == "debug":
        trainer = Trainer(config)
        train_dataset = Loader(config, config.p_train_data)
        train_loader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=False)
        data_iter = iter(train_loader)
        frame_vecs, frame_n, ques, ques_n, start_frame, end_frame = data_iter.next()
        frame_vecs = frame_vecs.to(trainer.device)
        ques = ques.to(trainer.device)

        # Forward pass
        p1, p2 = trainer.model(frame_vecs, ques)
        y1, y2 = start_frame.to(trainer.device), end_frame.to(trainer.device)
        print(p1.shape,p2.shape,y1.shape,y2.shape)
        loss1 = F.nll_loss(p1, y1, reduction='elementwise_mean')
        loss2 = F.nll_loss(p2, y2, reduction='elementwise_mean')
        loss = (loss1 + loss2) / 2
        # Backward and optimize
        trainer.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(trainer.model.parameters(), 1)
        trainer.optimizer.step()
        print(loss.item())

    elif config.mode == "test":
        trainer = Trainer(config)
        trainer.evaluate()
    else:
        print("Unknown mode")
        exit(0)


if __name__ == '__main__':
    app.run(main)