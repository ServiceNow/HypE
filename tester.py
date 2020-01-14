import torch
from dataset import Dataset
import numpy as np
from measure import Measure
from os import listdir
from os.path import isfile, join

class Tester:
    def __init__(self, dataset, model, valid_or_test, model_name):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.model.eval()
        self.dataset = dataset
        self.model_name = model_name
        self.valid_or_test = valid_or_test
        self.measure = Measure()
        self.all_facts_as_set_of_tuples = set(self.allFactsAsTuples())

    def get_rank(self, sim_scores):#assuming the test fact is the first one
        return (sim_scores >= sim_scores[0]).sum()

    def create_queries(self, fact, position):
        r, e1, e2, e3, e4, e5, e6 = fact

        if position == 1:
            return [(r, i, e2, e3, e4, e5, e6) for i in range(1, self.dataset.num_ent())]
        elif position == 2:
            return [(r, e1, i, e3, e4, e5, e6) for i in range(1, self.dataset.num_ent())]
        elif position == 3:
            return [(r, e1, e2, i, e4, e5, e6) for i in range(1, self.dataset.num_ent())]
        elif position == 4:
            return [(r, e1, e2, e3, i, e5, e6) for i in range(1, self.dataset.num_ent())]
        elif position == 5:
            return [(r, e1, e2, e3, e4, i, e6) for i in range(1, self.dataset.num_ent())]
        elif position == 6:
            return [(r, e1, e2, e3, e4, e5, i) for i in range(1, self.dataset.num_ent())]

    def add_fact_and_shred(self, fact, queries, raw_or_fil):
        if raw_or_fil == "raw":
            result = [tuple(fact)] + queries
        elif raw_or_fil == "fil":
            result = [tuple(fact)] + list(set(queries) - self.all_facts_as_set_of_tuples)
        return self.shred_facts(result)


    def test(self, test_by_arity=False):
        """
        Evaluate the given dataset and print results, either by arity or all at once
        """
        settings = ["raw", "fil"]
        normalizer = 0
        self.measure_by_arity = {}
        self.meaddsure = Measure()

        # If the dataset is JF17K and we have test data by arity, then
        # compute test accuracies by arity and show also global result
        if test_by_arity:
        #if (self.valid_or_test == 'test' and self.dataset.data.get('test_2', None) is not None):
            # Iterate over test sets by arity
            for cur_arity in range(2,self.dataset.max_arity+1):
                # Reset the normalizer by arity
                test_by_arity = "test_{}".format(cur_arity)
                # If the dataset does not exit, continue
                if not (test_by_arity in vars(self.dataset)['data'].keys()):
                    print("{} does not exist. Skipping.".format(test_by_arity))
                    continue

                print("**** Evaluating arity {} having {} samples".format(cur_arity, len(self.dataset.data[test_by_arity])))
                # Evaluate the test data for arity cur_arity
                current_measure, normalizer_by_arity =  self.eval_dataset(self.dataset.data[test_by_arity])

                # Sum before normalizing current_measure
                normalizer += normalizer_by_arity
                self.measure += current_measure

                # Normalize the values for the current arity and save to dict
                current_measure.normalize(normalizer_by_arity)
                self.measure_by_arity[test_by_arity] = current_measure

        else:
            # Evaluate the test data for arity cur_arity
            current_measure, normalizer =  self.eval_dataset(self.dataset.data[self.valid_or_test])
            self.measure = current_measure

        self.measure.normalize(normalizer)

        for arity in self.measure_by_arity:
            print("Results for arity {}".format(arity[5:]))
            self.measure_by_arity[arity].print_()
        print("Results for ALL ARITIES in {} set".format(self.valid_or_test))
        self.measure.print_()
        return self.measure, self.measure_by_arity

    def eval_dataset(self, dataset):
        """
        Evaluate the given dataset with the given model.
        """
        # Reset normalization parameter
        settings = ["raw", "fil"]
        normalizer = 0
        # Contains the measure values for the given dataset (e.g. test for arity 2)
        current_rank = Measure()
        for i, fact in enumerate(dataset):
            arity = self.dataset.max_arity - (fact == 0).sum()
            for j in range(1, arity + 1):
                normalizer += 1
                queries = self.create_queries(fact, j)
                for raw_or_fil in settings:
                    r, e1, e2, e3, e4, e5, e6 = self.add_fact_and_shred(fact, queries, raw_or_fil)
                    if(self.model_name == "HypE"):
                        ms = np.zeros((len(r),6))
                        bs = np.ones((len(r), 6))

                        ms[:, 0:arity] = 1
                        bs[:, 0:arity] = 0

                        ms = torch.tensor(ms).float().to(self.device)
                        bs = torch.tensor(bs).float().to(self.device)
                        sim_scores = self.model(r, e1, e2, e3, e4, e5, e6, ms, bs).cpu().data.numpy()
                    elif(self.model_name == "MTransH"):
                        ms = np.zeros((len(r),6))
                        ms[:, 0:arity] = 1
                        ms = torch.tensor(ms).float().to(self.device)
                        sim_scores = self.model(r, e1, e2, e3, e4, e5, e6, ms).cpu().data.numpy()
                    else:
                        sim_scores = self.model(r, e1, e2, e3, e4, e5, e6).cpu().data.numpy()

                    # Get the rank and update the measures
                    rank = self.get_rank(sim_scores)
                    current_rank.update(rank, raw_or_fil)
                    # self.measure.update(rank, raw_or_fil)

            if i%1000 == 0:
                print("--- Testing sample {}".format(i))

        return current_rank, normalizer

    def shred_facts(self, tuples):
        r  = [tuples[i][0] for i in range(len(tuples))]
        e1 = [tuples[i][1] for i in range(len(tuples))]
        e2 = [tuples[i][2] for i in range(len(tuples))]
        e3 = [tuples[i][3] for i in range(len(tuples))]
        e4 = [tuples[i][4] for i in range(len(tuples))]
        e5 = [tuples[i][5] for i in range(len(tuples))]
        e6 = [tuples[i][6] for i in range(len(tuples))]
        return torch.LongTensor(r).to(self.device), torch.LongTensor(e1).to(self.device), torch.LongTensor(e2).to(self.device), torch.LongTensor(e3).to(self.device), torch.LongTensor(e4).to(self.device), torch.LongTensor(e5).to(self.device), torch.LongTensor(e6).to(self.device)

    def allFactsAsTuples(self):
        tuples = []
        for spl in self.dataset.data:
            for fact in self.dataset.data[spl]:
                tuples.append(tuple(fact))
        return tuples


