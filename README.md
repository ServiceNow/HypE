# HypE


Summary
=======

This is the implementation of the model proposed in [Knowledge Hypergraphs: Extending Knowledge Graphs Beyond Binary Relations](https://arxiv.org/abs/1906.00137) for knowledge hypergraph embedding and also all the baselines for this task. It can be also used to learn `HypE` models for any input model. The software can be also used as a framework to implement new knowledge hypergraph embedding models.

## Dependencies

* `Python` version 3.6.6
* `Numpy` version 1.16.2
* `PyTorch` version 0.4.0

## Usage

To run HypE or any of the baselines you should define the following parameters:

`model`: name of the model

`dataset`: The dataset you want to run this model on

`lr`: learning rate

`nr`: number of negative examples per positive example per arity

`out_channels`: number of out channels for convolution filters in HypE

`filt_w`: width of convolutional weight filters in HypE

`stride`: stride of convolutional weight filters in HypE

`emb_dim`: embedding dimension

`input_drop`: drop out rate for input layer of all models

`hidden_drop`: drop out rate for hidden layer of all models

* Run `python main.py -model model -dataset dataset -lr lr -nr nr -out_channels out_channels -filt_w filt_w -stride stride -emb_dim emb_dim -hidden_drop hidden_drop -input_drop input_drop`
