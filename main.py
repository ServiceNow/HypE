import numpy as np
import torch
from torch.nn import functional as F
from models import *
import argparse
from dataset import Dataset as dataset
from tester import Tester
import math

class Experiment:

    def __init__(self, model_name, dataset, num_iterations, batch_size, learning_rate, emb_dim, hidden_drop, input_drop, neg_ratio,
                in_channels, out_channels, filt_h, filt_w, stride):
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.emb_dim = emb_dim
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.neg_ratio = neg_ratio
        self.max_arity = 6
        self.dataset = dataset
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.kwargs = {"in_channels":in_channels,"out_channels": out_channels, "filt_h": filt_h, "filt_w": filt_w, "hidden_drop": hidden_drop, "stride": stride, "input_drop":input_drop}
        self.hyperpars = {"model": model_name,"lr":learning_rate,"emb_dim":emb_dim,"out_channels":out_channels,"filt_w":filt_w,"nr":neg_ratio,"stride":stride, "hidden_drop":hidden_drop, "input_drop":input_drop}
        self.stride = stride

    def decompose_predictions(self, targets, predictions, max_length):
        positive_indices = np.where(targets > 0)[0]
        seq = []
        for ind, val in enumerate(positive_indices):
            if(ind == len(positive_indices)-1):
                seq.append(self.padd(predictions[val:], max_length))
            else:
                seq.append(self.padd(predictions[val:positive_indices[ind + 1]], max_length))
        return seq

    def padd(self, a, max_length):
        b = F.pad(a, (0,max_length - len(a)), 'constant', -math.inf)
        return b

    def padd_and_decompose(self, targets, predictions, max_length):
        seq = self.decompose_predictions(targets, predictions, max_length)
        return torch.stack(seq)


    def load_and_test(self):
        print("Testing the %s model..." % self.model_name)

        if(self.model_name == "MDistMult"):
            model = MDistMult(self.dataset, self.emb_dim, **self.kwargs).to(self.device)
        elif(self.model_name == "MCP"):
            model = MCP(self.dataset, self.emb_dim, **self.kwargs).to(self.device)
        elif(self.model_name == "HSimplE"):
            model = HSimplE(self.dataset, self.emb_dim, **self.kwargs).to(self.device)
        elif(self.model_name == "HypE"):
            model = HypE(self.dataset, self.emb_dim, **self.kwargs).to(self.device)
        elif(self.model_name == "MTransH"):
            model = MTransH(self.dataset, self.emb_dim, **self.kwargs).to(self.device)


        model.init()
        ####################
        if self.model_name=='HSimplE':
            print("LOADING HSimplE")
            model.load_state_dict(torch.load('./data/HSimplE-0.01.chkpnt'))
        elif self.model_name=='MTransH':
            print("LOADING MTransH")
            model.load_state_dict(torch.load('./data/MTransH-0.06.chkpnt'))
        else:
            print("NO MODEL FOUND")
        model.eval()
        with torch.no_grad():
            #print("test in iteration " + str(best_itr) + ":")
            tester = Tester(self.dataset, model, "test", self.model_name)
            tester.test()



    def train_and_eval(self):
        print("Training the %s model..." % self.model_name)
        print("Number of training data points: %d" % len(self.dataset.data["train"]))

        if(self.model_name == "MDistMult"):
            model = MDistMult(self.dataset, self.emb_dim, **self.kwargs).to(self.device)
        elif(self.model_name == "MCP"):
            model = MCP(self.dataset, self.emb_dim, **self.kwargs).to(self.device)
        elif(self.model_name == "HSimplE"):
            model = HSimplE(self.dataset, self.emb_dim, **self.kwargs).to(self.device)
        elif(self.model_name == "HypE"):
            model = HypE(self.dataset, self.emb_dim, **self.kwargs).to(self.device)
        elif(self.model_name == "MTransH"):
            model = MTransH(self.dataset, self.emb_dim, **self.kwargs).to(self.device)


        model.init()

        opt = torch.optim.Adagrad(model.parameters(), lr=self.learning_rate)
        loss_layer = torch.nn.CrossEntropyLoss()
        print("Starting training...")
        best_mrr = 0
        for it in range(1, self.num_iterations+1):
            last_batch = False
            model.train()
            losses = 0
            while not last_batch:
                r, e1, e2, e3, e4, e5, e6, targets, ms, bs = self.dataset.next_batch(self.batch_size, neg_ratio=self.neg_ratio, device=self.device)
                last_batch = self.dataset.was_last_batch()
                opt.zero_grad()
                number_of_positive = len(np.where(targets > 0)[0])
                if(self.model_name == "HypE" or self.model_name == "HypE_DM"):
                    predictions = model.forward(r, e1, e2, e3, e4, e5, e6, ms, bs)
                elif(self.model_name == "MTransH"):
                    predictions = model.forward(r, e1, e2, e3, e4, e5, e6, ms)
                else:
                    predictions = model.forward(r, e1, e2, e3, e4, e5, e6)
                predictions = self.padd_and_decompose(targets, predictions, self.neg_ratio*self.max_arity)
                targets = torch.zeros(number_of_positive).long().to(self.device)
                loss = loss_layer(predictions, targets)
                loss.backward()
                opt.step()
                losses += loss.item()

            print("iteration#: " + str(it) + ", loss: " + str(losses))

            if(it % 100 == 0):
                model.eval()
                with torch.no_grad():
                    print("validation:")
                    tester = Tester(self.dataset, model, "valid", self.model_name)
                    mrr = tester.test()
                    if(mrr > best_mrr):
                        best_mrr = mrr
                        best_model = model
                        best_itr = it


        best_model.eval()
        with torch.no_grad():
            print("test in iteration " + str(best_itr) + ":")
            tester = Tester(self.dataset, best_model, "test", self.model_name)
            tester.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, default="HypE_DM")
    parser.add_argument('-dataset', type=str, default="JF17K")
    parser.add_argument('-lr', type=float)
    parser.add_argument('-nr', type=int, default=10)
    parser.add_argument('-out_channels', type=int, default=2)
    parser.add_argument('-filt_w', type=int, default=2)
    parser.add_argument('-emb_dim', type=int, default=200)
    parser.add_argument('-hidden_drop', type=float, default=0.2)
    parser.add_argument('-input_drop', type=float, default=0.2)
    parser.add_argument('-stride', type=int, default=2)
    args = parser.parse_args()

    dataset = dataset(args.dataset)
    experiment = Experiment(args.model, dataset, num_iterations=500, batch_size=128, learning_rate=args.lr, emb_dim=args.emb_dim,
                            hidden_drop=args.hidden_drop, input_drop=args.input_drop, neg_ratio=args.nr, in_channels=1, out_channels=args.out_channels, filt_h=1, filt_w=args.filt_w, stride=args.stride)
    experiment.load_and_test()
    #experiment.train_and_eval()
