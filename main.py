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

DEFAULT_SAVE_DIR = 'output'
DEFAULT_MAX_ARITY = 6

class Experiment:

    def __init__(self, model_name, dataset, num_iterations, batch_size, learning_rate, emb_dim, max_arity, hidden_drop,
                 input_drop, neg_ratio, in_channels, out_channels, filt_h, filt_w, stride, pretrained, no_test_by_arity):
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.emb_dim = emb_dim
        self.batch_size = batch_size
        self.neg_ratio = neg_ratio
        self.max_arity = max_arity
        self.dataset = dataset
        self.pretrained = pretrained
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.kwargs = {"in_channels":in_channels,"out_channels": out_channels, "filt_h": filt_h, "filt_w": filt_w,
                       "hidden_drop": hidden_drop, "stride": stride, "input_drop":input_drop}
        self.hyperpars = {"model": model_name,"lr":learning_rate,"emb_dim":emb_dim,"out_channels":out_channels,
                          "filt_w":filt_w,"nr":neg_ratio,"stride":stride, "hidden_drop":hidden_drop, "input_drop":input_drop}


        self.num_iterations = num_iterations
        self.stride = stride
        # Create an output dir unless a pretrained directory is given
        self.output_dir = self.create_output_dir(pretrained)
        self.measure = None
        self.measure_by_arity = None
        self.test_by_arity = not no_test_by_arity
        # Load the specified model and initialize based on given checkpoint (if any)
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
            self.model = HSimplE(self.dataset, self.emb_dim, **self.kwargs).to(self.device)
        elif(self.model_name == "HypE"):
            self.model = HypE(self.dataset, self.emb_dim, **self.kwargs).to(self.device)
        elif(self.model_name == "MTransH"):
            self.model = MTransH(self.dataset, self.emb_dim, **self.kwargs).to(self.device)
        else:
            raise Exception("!!!! No mode called {} found !!!!".format(self.model_name))

        # Load the pretrained model
        self.opt = torch.optim.Adagrad(self.model.parameters(), lr=self.learning_rate)
        if self.pretrained is not None:
            print("Loading the pretrained model at {}".format(self.pretrained))
            self.model.load_state_dict(torch.load(self.pretrained))
            # Construct the name of the optimizer file based on the pretrained model path
            opt_path = os.path.join(os.path.dirname(self.pretrained), os.path.basename(self.pretrained).replace('model','opt'))
            # If an optimizer exists (needed for training, but not testing), then load it.
            if os.path.exists(opt_path):
                self.opt.load_state_dict(torch.load(opt_path))
            else:
                print("*** NO OPTIMIZER FOUND. SKIPPING. ****")

        else:
            # Initilize the model
            self.model.init()


    def test_and_eval(self):
        print("Testing the {} model on {}...".format(self.model_name, self.dataset.name))
        self.model.eval()
        with torch.no_grad():
            tester = Tester(self.dataset, self.model, "test", self.model_name)
            self.measure, self.measure_by_arity = tester.test(self.test_by_arity)


    def train_and_eval(self):
        # If the number of iterations is the same as the current iteration, exit.
        if (self.model.cur_itr.data >= self.num_iterations):
            print("*************")
            print("Number of iterations is the same as that in the pretrained model.")
            print("Nothing left to train. Exiting.")
            print("*************")
            return

        print("Training the {} model...".format(self.model_name))
        print("Number of training data points: {}".format(len(self.dataset.data["train"])))

        best_model = None

        loss_layer = torch.nn.CrossEntropyLoss()
        print("Starting training...")
        best_mrr = 0
        for it in range(self.model.cur_itr.data, self.num_iterations+1):
            last_batch = False
            self.model.train()
            self.model.cur_itr.data += 1
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

            print("Iteration#: {}, loss: {}".format(it, losses))

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
                        self.save_model(it, "valid")


        if best_model is None:
            best_model = self.model
            self.best_itr = it
        best_model.eval()
        with torch.no_grad():
            print("test in iteration {}:".format(self.best_itr))
            tester = Tester(self.dataset, best_model, "test", self.model_name)
            self.measure, self.measure_by_arity = tester.test(self.test_by_arity)

        # Save the model at checkpoint
        print("Saving model at {}".format(self.output_dir))
        self.save_model(it, "test")


    def create_output_dir(self, output_dir=None):
        """
        If an output dir is given, then make sure it exists. Otherwise, create one based on time stamp.
        """
        if output_dir is None:
            time = datetime.datetime.now()
            model_name = '{}_{}_{}'.format(self.model_name, self.dataset.name, time.strftime("%Y%m%d-%H%M%S"))
            output_dir = os.path.join(DEFAULT_SAVE_DIR, self.model_name, model_name)
        else:
            output_dir = os.path.dirname(self.pretrained)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        return output_dir


    def save_model(self, itr=None, test_or_valid='test'):
            """
            Save the model state to the output folder
            """
            model_name = 'model_{}itr.chkpnt'.format(itr) if itr else self.model_name+'.chkpnt'
            opt_name = 'opt_{}itr.chkpnt'.format(itr) if itr else self.model_name+'.chkpnt'
            measure_name = '{}_measure_{}itr.json'.format(test_or_valid, itr) if itr else '{}.json'.format(self.model_name)

            torch.save(self.model.state_dict(), os.path.join(self.output_dir, model_name))
            torch.save(self.opt.state_dict(), os.path.join(self.output_dir, opt_name))
            if self.measure is not None:
                measure_dict = vars(self.measure)
                measure_dict["best_iteration"] = self.best_itr
                with open(os.path.join(self.output_dir, measure_name), 'w') as f:
                        json.dump(measure_dict, f, indent=4, sort_keys=True)
            # Note that measure_by_arity is only computed at test time (not validation)
            if (self.test_by_arity) and (self.measure_by_arity is not None):
                H = {}
                H["best_iteration"] = self.best_itr
                measure_by_arity_name = '{}_measure_{}itr_by_arity.json'.format(test_or_valid, itr) if itr else '{}.json'.format(self.model_name)
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
    parser.add_argument('-out_channels', type=int, default=6)
    parser.add_argument('-in_channels', type=int, default=1)
    parser.add_argument('-filt_w', type=int, default=1)
    parser.add_argument('-filt_h', type=int, default=1)
    parser.add_argument('-emb_dim', type=int, default=200)
    parser.add_argument('-hidden_drop', type=float, default=0.2)
    parser.add_argument('-input_drop', type=float, default=0.2)
    parser.add_argument('-stride', type=int, default=2)
    parser.add_argument('-num_iterations', type=int, default=1000)
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument("-test", action="store_true")
    parser.add_argument("-no_test_by_arity", action="store_true")
    parser.add_argument('-pretrained', type=str, default=None, help="A path to a trained model, which will be loaded if provided.")
    args = parser.parse_args()

    dataset = dataset(args.dataset)
    experiment = Experiment(args.model, dataset, args.num_iterations, batch_size=args.batch_size, learning_rate=args.lr,
                            emb_dim=args.emb_dim, max_arity=DEFAULT_MAX_ARITY, hidden_drop=args.hidden_drop,
                            input_drop=args.input_drop, neg_ratio=args.nr, in_channels=args.in_channels, out_channels=args.out_channels,
                            filt_h=args.filt_h, filt_w=args.filt_w, stride=args.stride, pretrained=args.pretrained, no_test_by_arity=args.no_test_by_arity)

    if args.test:
        print("************** START OF TESTING ********************", experiment.model_name)
        if args.pretrained is None:
            raise Exception("You must provide a trained model to test!")
        experiment.test_and_eval()
    else:
        print("************** START OF TRAINING ********************", experiment.model_name)
        experiment.train_and_eval()
