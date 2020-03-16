import os
import glob
import re
import json
import datetime
import numpy as np
import torch
from torch.nn import functional as F
from models import *
import argparse
from dataset import Dataset
from tester import Tester
import math

DEFAULT_SAVE_DIR = 'output'
DEFAULT_MAX_ARITY = 6

class Experiment:
    def __init__(self, args):
        self.model_name = args.model
        self.learning_rate = args.lr
        self.emb_dim = args.emb_dim
        self.batch_size = args.batch_size
        self.neg_ratio = args.nr
        self.max_arity = DEFAULT_MAX_ARITY
        self.pretrained = args.pretrained
        self.test = args.test
        self.output_dir = args.output_dir
        self.restartable = args.restartable
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.kwargs = {"in_channels":args.in_channels,"out_channels":args.out_channels, "filt_h":args.filt_h, "filt_w":args.filt_w,
                       "hidden_drop":args.hidden_drop, "stride":args.stride, "input_drop":args.input_drop}
        self.hyperpars = {"model":args.model,"lr":args.lr,"emb_dim":args.emb_dim,"out_channels":args.out_channels,
                          "filt_w":args.filt_w,"nr":args.nr,"stride":args.stride, "hidden_drop":args.hidden_drop, "input_drop":args.input_drop}

        # Load the dataset
        self.dataset = Dataset(args.dataset, DEFAULT_MAX_ARITY)

        self.num_iterations = args.num_iterations
        # Create an output dir unless one is given
        self.output_dir = self.create_output_dir(args.output_dir)
        self.measure = None
        self.measure_by_arity = None
        self.test_by_arity = not args.no_test_by_arity
        self.best_model = None
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

    def get_model_from_name(self, model_name):
        """
        Instantiate a model object given the model name
        """
        model = None
        if(model_name == "MDistMult"):
            model = MDistMult(self.dataset, self.emb_dim, **self.kwargs).to(self.device)
        elif(model_name == "MCP"):
            model = MCP(self.dataset, self.emb_dim, **self.kwargs).to(self.device)
        elif(model_name == "HSimplE"):
            model = HSimplE(self.dataset, self.emb_dim, **self.kwargs).to(self.device)
        elif(model_name == "HypE"):
            model = HypE(self.dataset, self.emb_dim, **self.kwargs).to(self.device)
        elif(model_name == "MTransH"):
            model = MTransH(self.dataset, self.emb_dim, **self.kwargs).to(self.device)
        else:
            raise Exception("!!!! No mode called {} found !!!!".format(self.model_name))
        return model


    def load_last_saved_model(self, output_dir):
        """
        Find the last saved model in the output_dir and load it.
        Load also the best_model
        If no model found in the output_dir or if the dir does not exists
        initialize the model randomly.
        Sets self.model, self.best_model
        """
        model_found = False
        # If the output_dir contains a model, then it will be loaded
        # Pick the latest saved model
        try:
            # Get the list of checkpoint files and strip the full path
            model_list = [os.path.basename(x) for x in glob.glob(os.path.join(self.output_dir, 'model_*.chkpnt'))]
            # Sort the files based on the iteration number
            model_list_sorted = sorted(model_list, key=lambda f: int(re.match(r'model_(\d+)itr.chkpnt', f).groups(0)[0]))
        except:
            print("*** NO SAVED MODEL FOUND in {}. LOADING FROM SCRATCH. ****".format(self.output_dir))
            # Initilize the model
            self.model.init()
        else: # if no exceptions, then run the following
            # If there are saved models
            if len(model_list_sorted) > 0:
                # Pick the most recent model
                self.pretrained = os.path.join(self.output_dir, model_list_sorted[-1])
                # Construct the name of the optimizer file based on the pretrained model path
                opt_path = os.path.join(os.path.dirname(self.pretrained), os.path.basename(self.pretrained).replace('model','opt'))
                if os.path.exists(opt_path):
                    self.model.load_state_dict(torch.load(self.pretrained))
                    self.opt.load_state_dict(torch.load(opt_path))
                    model_found = True

                    try:
                        # Load the best model
                        best_model_path = os.path.join(self.output_dir, "best_model.chkpnt")
                        self.best_model = self.get_model_from_name(self.model_name)
                        self.best_model.load_state_dict(torch.load(best_model_path))
                        print("Loading the model {} with best MRR {}.".format(self.pretrained, self.best_model.best_mrr))
                    except:
                        print("*** NO BEST MODEL FOUND in {}. ****".format(self.output_dir))
                        # Set the best model to None
                        self.best_model = None

        if not model_found:
            print("*** NO MODEL/OPTIMIZER FOUND in {}. LOADING FROM SCRATCH. ****".format(self.output_dir))
            # Initilize the model
            self.model.init()


    def load_model(self):
        """ If a pretrained model is provided, then it will be loaded. """
        """ If the output_dir contains a (half-)trained model, then it will be loaded """
        """ Priority is given to pretrained model. """

        print("Initializing the model ...")
        self.model = self.get_model_from_name(self.model_name)

        # Load the pretrained model
        self.opt = torch.optim.Adagrad(self.model.parameters(), lr=self.learning_rate)

        if self.pretrained is not None:
            print("Loading the pretrained model at {} for testing".format(self.pretrained))
            self.model.load_state_dict(torch.load(self.pretrained))
            # Construct the name of the optimizer file based on the pretrained model path
            opt_path = os.path.join(os.path.dirname(self.pretrained), os.path.basename(self.pretrained).replace('model','opt'))
            # If an optimizer exists (needed for training, but not testing), then load it.
            if os.path.exists(opt_path):
                self.opt.load_state_dict(torch.load(opt_path))
            elif not self.test:
                # if no optimizer found and 'test` is not set, then raise an error
                # (we need a pretrained optimizer in order to continue training
                raise Exception("*** NO OPTIMIZER FOUND. SKIPPING. ****")
        elif self.restartable and os.path.isdir(self.output_dir):
            # If the output_dir contains a model, then it will be loaded
            self.load_last_saved_model(self.output_dir)
        else:
            # Initilize the model
            self.model.init()


    def test_and_eval(self):
        print("Testing the {} model on {}...".format(self.model_name, self.dataset.name))
        self.model.eval()
        with torch.no_grad():
            tester = Tester(self.dataset, self.model, "test", self.model_name)
            self.measure, self.measure_by_arity = tester.test(self.test_by_arity)
            # Save the result of the tests
            self.save_model(self.model.cur_itr, "test")


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


        loss_layer = torch.nn.CrossEntropyLoss()
        print("Starting training at iteration ... {}".format(self.model.cur_itr.data))
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
                if(self.model_name == "HypE"):
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

            # Evaluate the model every 100th iteration or if it is the last iteration
            if (it % 100 == 0) or (it == self.num_iterations):
                self.model.eval()
                with torch.no_grad():
                    print("validation:")
                    tester = Tester(self.dataset, self.model, "valid", self.model_name)
                    measure_valid, _ = tester.test()
                    mrr = measure_valid.mrr["fil"]
                    # This is the best model we have so far if
                    # no "best_model" exists yes, or if this MRR is better than what we had before
                    is_best_model = (self.best_model is None) or (mrr > self.best_model.best_mrr)
                    if is_best_model:
                        self.best_model = self.model
                        # Update the best_mrr value
                        self.best_model.best_mrr.data = torch.from_numpy(np.array([mrr]))
                        self.best_model.best_itr.data = torch.from_numpy(np.array([it]))
                    # Save the model at checkpoint
                    self.save_model(it, "valid", is_best_model=is_best_model)


        #if self.best_model is None:
        #    self.best_model = self.model

        self.best_model.eval()
        with torch.no_grad():
            print("testing best model at iteration {} .... ".format(self.best_model.best_itr))
            tester = Tester(self.dataset, self.best_model, "test", self.model_name)
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

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print("Created output directory {}".format(output_dir))
        return output_dir


    def save_model(self, itr=0, test_or_valid='test', is_best_model=False):
            """
            Save the model state to the output folder.
            If is_best_model is True, then save the model also as best_model.chkpnt
            """
            if is_best_model:
                torch.save(self.model.state_dict(), os.path.join(self.output_dir, 'best_model.chkpnt'))
                print("######## Saving the BEST MODEL")

            model_name = 'model_{}itr.chkpnt'.format(itr)
            #model_name = 'model_{}itr.chkpnt'.format(itr) if itr else '{}.chkpnt'.format(self.model_name)
            opt_name = 'opt_{}itr.chkpnt'.format(itr) if itr else '{}.chkpnt'.format(self.model_name)
            measure_name = '{}_measure_{}itr.json'.format(test_or_valid, itr) if itr else '{}.json'.format(self.model_name)
            print("######## Saving the model {}".format(os.path.join(self.output_dir, model_name)))

            torch.save(self.model.state_dict(), os.path.join(self.output_dir, model_name))
            torch.save(self.opt.state_dict(), os.path.join(self.output_dir, opt_name))
            if self.measure is not None:
                measure_dict = vars(self.measure)
                # If a best model exists
                if self.best_model:
                    measure_dict["best_iteration"] = self.best_model.best_itr.cpu().item()
                    measure_dict["best_mrr"] = self.best_model.best_mrr.cpu().item()
                with open(os.path.join(self.output_dir, measure_name), 'w') as f:
                        json.dump(measure_dict, f, indent=4, sort_keys=True)
            # Note that measure_by_arity is only computed at test time (not validation)
            if (self.test_by_arity) and (self.measure_by_arity is not None):
                H = {}
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
    parser.add_argument('-pretrained', type=str, default=None, help="A path to a trained model (.chkpnt file), which will be loaded if provided.")
    parser.add_argument('-output_dir', type=str, default=None, help="A path to the directory where the model will be saved and/or loaded from.")
    parser.add_argument('-restartable', action="store_true", help="If restartable is set, then you must specify an output_dir")
    args = parser.parse_args()

    if args.restartable and (args.output_dir is None):
            parser.error("-restarable requires -output_dir.")

    experiment = Experiment(args)

    if args.test:
        print("************** START OF TESTING ********************", experiment.model_name)
        if args.pretrained is None:
            raise Exception("You must provide a trained model to test!")
        experiment.test_and_eval()
    else:
        print("************** START OF TRAINING ********************", experiment.model_name)
        experiment.train_and_eval()
