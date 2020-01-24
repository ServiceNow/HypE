import torch
import random
import logging
import itertools
from torch.utils.data import DataLoader, Dataset, RandomSampler
from tqdm import tqdm, trange

logger = logging.getLogger(__name__)


class Txt2GrphDataset(Dataset):
    def __init__(self, corpus_path, tokenizer, seq_len, encoding="utf-8", corpus_lines=None, on_memory=True):
        self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.on_memory = on_memory
        self.corpus_path = corpus_path
        self.encoding = encoding
        self.current_rel = 0  # to avoid random sentence from same doc

        # for loading samples directly from file
        self.sample_counter = 0  # used to keep track of full epochs on file
        self.line_buffer = None  # keep second sentence of a pair in memory and use as first sentence in next pair

        # for loading samples in memory
        self.current_random_doc = 0
        self.num_rels = 0
        self.sample_to_doc = []  # map sample index to doc and line

        self.all_rel_data = {}
        self.num_lines = 0
        with open(corpus_path, "r", encoding=encoding) as f:
            for line in tqdm(f, desc="Loading Dataset", total=corpus_lines):
                line = line.strip()
                assert line != '', "There should be no empty lines in your dataset."
                items = line.split('\t')
                rel = items[0]
                entities = items[1:]
                if rel not in self.all_rel_data.keys():
                    self.all_rel_data[rel] = []
                    self.num_rels += 1
                sample = {"doc_id": rel, "line": len(self.all_rel_data[rel])}
                self.sample_to_doc.append(sample)
                self.all_rel_data[rel].append(" ".join(entities))
                self.num_lines = self.num_lines + 1

        # remove last added sample because there won't be a subsequent line anymore in the sample_to_doc
        self.rels = list(self.all_rel_data.keys())
        for rel in self.rels:
            sample2remove = {"doc_id": rel, "line": len(self.all_rel_data[rel])-1}
            self.sample_to_doc.remove(sample2remove)

    def __len__(self):
        return self.num_lines - self.num_rels - 1

    def __getitem__(self, item):
        cur_id = self.sample_counter
        self.sample_counter += 1

        r, t1, t2, is_same_relation = self.random_entities(item)

        # tokenize
        relation = self.special_tokenize(r)
        tokens_a = self.special_tokenize(t1)
        tokens_b = self.special_tokenize(t2)

        tokens_a = relation + tokens_a
        tokens_b = relation + tokens_b

        # combine to one sample
        cur_example = InputExample(guid=cur_id, tokens_a=tokens_a, tokens_b=tokens_b, is_next=is_same_relation)

        # transform sample to features
        cur_features = convert_example_to_features(cur_example, self.seq_len, self.tokenizer)

        cur_tensors = (torch.tensor(cur_features.input_ids),
                       torch.tensor(cur_features.input_mask),
                       torch.tensor(cur_features.segment_ids),
                       torch.tensor(cur_features.lm_label_ids),
                       torch.tensor(cur_features.is_next))
        return cur_tensors

    def special_tokenize(self, tokens):
        """ The graph based datasets contain a lot of punctuation that would be tokenized but should not
        :param tokens:
        :return:
        """
        new_tokens = []
        for t in tokens.split():
            list = self.tokenizer.tokenize(t)
            new_tokens.append(list)
        return new_tokens

    def random_entities(self, index):
        """
        Get one sample from corpus consisting of two sets of entities. With prob. 50% these are two sets of entities
        from the same relation. With 50% the second set will be from a random other relation.
        :param index: int, index of sample.
        :return: (str, str, int), sentence 1, sentence 2, isNextSentence Label
        """
        r, t1, t2 = self.get_corpus_line(index)
        if random.random() > 0.5:
            label = 0
        else:
            t2 = self.get_random_line()
            label = 1

        assert len(t1) > 0
        assert len(t2) > 0
        return r, t1, t2, label

    def get_corpus_line(self, item):
        """
        Get one sample from corpus consisting of a pair of two subsequent lines from the same doc.
        :param item: int, index of sample.
        :return: (str, str), two subsequent sentences from corpus
        """
        t1 = ""
        t2 = ""
        assert item < self.num_lines
        if self.all_rel_data:
            sample = self.sample_to_doc[item]
            t1 = self.all_rel_data[sample["doc_id"]][sample["line"]]
            t2 = random.choice(self.all_rel_data[sample["doc_id"]])
            # used later to avoid random nextSentence from same doc
            self.current_rel = sample["doc_id"]
            return self.current_rel, t1, t2

        assert t1 != ""
        assert t2 != ""
        return t1, t2

    def get_random_line(self):
        """
        Get random line from another document for nextSentence task.
        :return: str, content of one line
        """
        # Similar to original tf repo: This outer loop should rarely go for more than one iteration for large
        # corpora. However, just to be careful, we try to make sure that
        # the random document is not the same as the document we're processing.
        for _ in range(10):
            self.current_rand_rel = random.choice(self.rels)
            rand_lines = self.all_rel_data[self.current_rand_rel]
            line = rand_lines[random.randrange(len(rand_lines))]
            #check if our picked random line is really from another relation like we want it to be
            if self.current_rand_rel != self.current_rel:
                break
        return line  # add current relation identifiers to other seq of relation

    def get_next_line(self):
        """ Gets next line of random_file and starts over when reaching end of file"""
        try:
            line = next(self.random_file).strip()
            #keep track of which document we are currently looking at to later avoid having the same doc as t1
            if line == "":
                self.current_random_doc = self.current_random_doc + 1
                line = next(self.random_file).strip()
        except StopIteration:
            self.random_file.close()
            self.random_file = open(self.corpus_path, "r", encoding=self.encoding)
            line = next(self.random_file).strip()
        return line


class InputExample(object):
    """A single training/test example for the language model."""

    def __init__(self, guid, tokens_a, tokens_b=None, is_next=None, lm_labels=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            tokens_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            tokens_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.tokens_a = tokens_a
        self.tokens_b = tokens_b
        self.is_next = is_next  # nextSentence
        self.lm_labels = lm_labels  # masked words for language model


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, is_next, lm_label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.is_next = is_next
        self.lm_label_ids = lm_label_ids


def random_word(tokens, tokenizer):
    """
    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    :param tokens: list of str, tokenized sentence.
    :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
    :return: (list of str, list of int), masked tokens and related labels for LM prediction
    """
    output_label = []

    # mask just one entity or relation
    pos = random.randint(0, len(tokens) - 1)
    masked_tokens = tokens
    masked_tokens[pos] = ["[MASK]"] * (len(tokens[pos]))
    masked_merged_tokens = list(itertools.chain(*masked_tokens))
    all_merged_tokens = list(itertools.chain(*tokens))

    for mtoken, atoken in zip(masked_merged_tokens, all_merged_tokens):
        # masked token
        if mtoken == "[MASK]":
            prob = random.random()

            # 10% randomly change token to random token
            if prob < 0.1:
                mtoken = random.choice(list(tokenizer.vocab.items()))[0]
            # append current token to output (we will predict these later)
            try:
                output_label.append(tokenizer.vocab[atoken])
            except KeyError:
                # For unknown words (should not occur with BPE vocab)
                output_label.append(tokenizer.vocab["[UNK]"])
                logger.warning("Cannot find token '{}' in vocab. Using [UNK] insetad".format(mtoken))
        # not masked token
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)
    assert len(masked_merged_tokens) == len(output_label)
    return masked_merged_tokens, output_label


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        sum_a = sum([len(t) for t in tokens_a])
        sum_b = sum([len(t) for t in tokens_b])
        total_length = sum_a + sum_b
        if total_length <= max_length:
            logger.warning("Total tokens is execeeding max token length. This should not happen.")
            break
        if len(sum_a) > len(sum_b):
            tokens_a[-1].pop()
        else:
            tokens_b[-1].pop()


def convert_example_to_features(example, max_seq_length, tokenizer):
    """
    Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample with
    IDs, LM labels, input_mask, CLS and SEP tokens etc.
    :param example: InputExample, containing sentence input as strings and is_next label
    :param max_seq_length: int, maximum length of sequence.
    :param tokenizer: Tokenizer
    :return: InputFeatures, containing all inputs and labels of one sample as IDs (as used for model training)
    """
    tokens_a = example.tokens_a
    tokens_b = example.tokens_b
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

    tokens_a, t1_label = random_word(tokens_a, tokenizer)
    tokens_b, t2_label = random_word(tokens_b, tokenizer)
    # concatenate lm labels and account for CLS, SEP, SEP
    lm_label_ids = ([-1] + t1_label + [-1] + t2_label + [-1])

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambigiously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    assert len(tokens_b) > 0
    for token in tokens_b:
        tokens.append(token)
        segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        lm_label_ids.append(-1)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(lm_label_ids) == max_seq_length

    if example.guid < 5:
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
        logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        logger.info("LM label: %s " % (lm_label_ids))
        logger.info("Is next sentence label: %s " % (example.is_next))

    features = InputFeatures(input_ids=input_ids,
                             input_mask=input_mask,
                             segment_ids=segment_ids,
                             lm_label_ids=lm_label_ids,
                             is_next=example.is_next)
    return features