import os
import json
import datetime
import numpy as np
import torch
from torch.nn import functional as F
from models import *
import argparse
from dataset import Dataset as dataset
from tester import Tester
import math

SAVE_DIR = 'output'

class Experiment:

    def __init__(self, model_name, dataset, num_iterations, batch_size, learning_rate, emb_dim, max_arity, hidden_drop,
                 input_drop, neg_ratio, in_channels, out_channels, filt_h, filt_w, stride, pretrained):
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.emb_dim = emb_dim
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.neg_ratio = neg_ratio
        self.max_arity = max_arity
        self.dataset = dataset
        self.pretrained = pretrained
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.kwargs = {"in_channels":in_channels,"out_channels": out_channels, "filt_h": filt_h, "filt_w": filt_w, "hidden_drop": hidden_drop, "stride": stride, "input_drop":input_drop}
        self.hyperpars = {"model": model_name,"lr":learning_rate,"emb_dim":emb_dim,"out_channels":out_channels,"filt_w":filt_w,"nr":neg_ratio,"stride":stride, "hidden_drop":hidden_drop, "input_drop":input_drop}
        self.stride = stride
        self.output_dir = self.create_output_dir()
        self.measure = None
        self.measure_by_arity = None
        # Load the right model
        self.load_model()

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


    def load_model(self):
        """ If a pretrained model is provided, then it will be loaded. """
        print("Initializing the model ...")
        if(self.model_name == "MDistMult"):
            self.model = MDistMult(self.dataset, self.emb_dim, **self.kwargs).to(self.device)
        elif(self.model_name == "MCP"):
            self.model = MCP(self.dataset, self.emb_dim, **self.kwargs).to(self.device)
        elif(self.model_name == "HSimplE"):
            self.model = HSimplE(self.dataset, self.emb_dim, self.max_arity, **self.kwargs).to(self.device)
        elif(self.model_name == "HypE"):
            self.model = HypE(self.dataset, self.emb_dim, **self.kwargs).to(self.device)
        elif(self.model_name == "MTransH"):
            self.model = MTransH(self.dataset, self.emb_dim, **self.kwargs).to(self.device)
        else:
            raise Exception("!!!! No mode called {} found !!!!".format(self.model_name))

        # Load the pretrained model
        if self.pretrained is not None:
            print("Loading the pretrained model at {}".format(self.pretrained))
            self.model.load_state_dict(torch.load(self.pretrained))


    def test_and_eval(self):
        print("Testing the {} model on {}...".format(self.model_name, self.dataset.name))
        self.model.eval()
        with torch.no_grad():
            tester = Tester(self.dataset, self.model, "test", self.model_name)
            test_by_arity = self.dataset.name.startswith('JF17K')
            print("STARTS WITH", test_by_arity, self.model_name, self.dataset.name.startswith('JF17K'))
            self.measure, self.measure_by_arity = tester.test(test_by_arity)


    def train_and_eval(self):
        print("Training the %s model..." % self.model_name)
        print("Number of training data points: %d" % len(self.dataset.data["train"]))

        best_model = None
        self.model.init()

        self.opt = torch.optim.Adagrad(self.model.parameters(), lr=self.learning_rate)
        loss_layer = torch.nn.CrossEntropyLoss()
        print("Starting training...")
        best_mrr = 0
        #for it in range(1, self.num_iterations+1):
        for it in range(1, self.num_iterations+1):
            last_batch = False
            self.model.train()
            losses = 0
            while not last_batch:
                r, e1, e2, e3, e4, e5, e6, targets, ms, bs = self.dataset.next_batch(self.batch_size, neg_ratio=self.neg_ratio, device=self.device)
                last_batch = self.dataset.was_last_batch()
                self.opt.zero_grad()
                number_of_positive = len(np.where(targets > 0)[0])
                if(self.model_name == "HypE" or self.model_name == "HypE_DM"):
                    predictions = self.model.forward(r, e1, e2, e3, e4, e5, e6, ms, bs)
                elif(self.model_name == "MTransH"):
                    predictions = self.model.forward(r, e1, e2, e3, e4, e5, e6, ms)
                else:
                    predictions = self.model.forward(r, e1, e2, e3, e4, e5, e6)
                predictions = self.padd_and_decompose(targets, predictions, self.neg_ratio*self.max_arity)
                targets = torch.zeros(number_of_positive).long().to(self.device)
                loss = loss_layer(predictions, targets)
                loss.backward()
                self.opt.step()
                losses += loss.item()

            print("iteration#: " + str(it) + ", loss: " + str(losses))

            if(it % 100 == 0):
                self.model.eval()
                with torch.no_grad():
                    print("validation:")
                    tester = Tester(self.dataset, self.model, "valid", self.model_name)
                    measure_valid, _ = tester.test()
                    mrr = measure_valid.mrr["fil"]
                    if(mrr > best_mrr):
                        best_mrr = mrr
                        best_model = self.model
                        self.best_itr = it
                        # Save the model at checkpoint
                        self.save_model(it)


        if best_model is None:
            best_model = self.model
            self.best_itr = it
        best_model.eval()
        with torch.no_grad():
            print("test in iteration " + str(self.best_itr) + ":")
            tester = Tester(self.dataset, best_model, "test", self.model_name)
            test_by_arity = self.dataset.name.startswith('JF17K')
            self.measure, self.measure_by_arity = tester.test(test_by_arity)

        # Save the model at checkpoint
        print("Saving model at {}".format(self.output_dir))
        self.save_model(it)


    def create_output_dir(self, output_dir=None):
        if output_dir is None:
            time = datetime.datetime.now()
            model_name = '{}_{}_{}'.format(self.model_name, self.dataset.name, time.strftime("%Y%m%d-%H%M%S"))
            output_dir = os.path.join(SAVE_DIR, self.model_name, model_name)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        return output_dir


    def save_model(self, itr=None):
            """
            Save the model state to the output folder
            """
            model_name = 'model_{}itr.chkpnt'.format(itr) if itr else self.model_name+'.chkpnt'
            opt_name = 'opt_{}itr.chkpnt'.format(itr) if itr else self.model_name+'.chkpnt'
            measure_name = 'measure_{}itr.json'.format(itr) if itr else self.model_name+'.json'
            torch.save(self.model.state_dict(), os.path.join(self.output_dir, model_name))
            torch.save(self.opt.state_dict(), os.path.join(self.output_dir, opt_name))
            if self.measure is not None:
                measure_dict = vars(self.measure)
                measure_dict["best_iteration"] = self.best_itr
                with open(os.path.join(self.output_dir, measure_name), 'w') as f:
                        json.dump(measure_dict, f, indent=4, sort_keys=True)
            if self.measure_by_arity is not None:
                H = {}
                measure_by_arity_name = 'measure_{}itr_by_arity.json'.format(itr) if itr else self.model_name+'.json'
                for key in self.measure_by_arity:
                    H[key] = vars(self.measure_by_arity[key])
                with open(os.path.join(self.output_dir, measure_by_arity_name), 'w') as f:
                        json.dump(H, f, indent=4, sort_keys=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, default="HSimplE")
    parser.add_argument('-dataset', type=str, default="JF17K")
    parser.add_argument('-lr', type=float, default=0.01)
    parser.add_argument('-nr', type=int, default=10)
    parser.add_argument('-out_channels', type=int, default=2)
    parser.add_argument('-filt_w', type=int, default=2)
    parser.add_argument('-emb_dim', type=int, default=200)
    parser.add_argument('-hidden_drop', type=float, default=0.2)
    parser.add_argument('-input_drop', type=float, default=0.2)
    parser.add_argument('-stride', type=int, default=2)
    parser.add_argument('-num_iterations', type=int, default=500)
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-max_arity', type=int, default=6)
    parser.add_argument("-test", action="store_true")
    parser.add_argument('-pretrained', type=str, default=None, help="A path to a trained model, which will be loaded if a value provided.")
    args = parser.parse_args()

    dataset = dataset(args.dataset)
    experiment = Experiment(args.model, dataset, args.num_iterations, batch_size=args.batch_size, learning_rate=args.lr,
                            emb_dim=args.emb_dim, max_arity=args.max_arity, hidden_drop=args.hidden_drop,
                            input_drop=args.input_drop, neg_ratio=args.nr, in_channels=1, out_channels=args.out_channels,
                            filt_h=1, filt_w=args.filt_w, stride=args.stride, pretrained=args.pretrained)
    if args.test:
        experiment.test_and_eval()
    else:
        print("********************EXPRIMENT****", experiment.model_name)
        experiment.train_and_eval()
