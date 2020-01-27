
Summary
=======

This repo contains the implementation of the model proposed in `Knowledge Hypergraphs: Extending Knowledge Graphs Beyond Binary Relations` for knowledge hypergraph embedding, as well as the code for most of the baselines in the paper. 
The code can be also used to train a `HypE` models for any input graph. 


_Note however that the code is designed to handle graphs with arity at most 6 (which is the case for JF17K)._

The software can be also used as a framework to implement new knowledge hypergraph embedding models.

## Dependencies

* `Python` version 3.7
* `Numpy` version 1.17
* `PyTorch` version 1.4.0

## Docker
We recommend running the code inside a `Docker` container. 
To do so, you will first need to have [Docker installed](https://docs.docker.com/).
You can then compile the image with:
```console
docker build -t hype-image:latest  .
```

and run using (replace the path to your local repo):
```console
docker run --rm -it -v {HypE-code-path}:/eai/project --user `id -u`:`id -g` hype-image /bin/bash
```

## Usage

To train HypE or any of the baselines you should define the parameters relevant to the given model.
The default values for most of these parameters are the ones that were used to obtain the results in the paper.

- `model`: The name of the model. Valid options are `HypE`, `HSimplE`, `MTransH`, `DistMult`, `MCP`.

- `dataset`: The dataset you want to run this model on (`JF17K` is included in this repo).

- `batch_size`: The training batch size.

- `num_iterations`: The total number of training iterations.

- `lr`: The learning rate.

- `nr`: number of negative examples per positive example for each arity.

- `out_channels`: number of out channels for convolution filters in HypE.

- `filt_w`: width of convolutional weight filters in HypE.

- `stride`: stride of convolutional weight filters in HypE.

- `emb_dim`: embedding dimension.

- `input_drop`: drop out rate for input layer of all models.

- `hidden_drop`: drop out rate for hidden layer of all models.

- `test`: when set, this will test a trained model on the test dataset. If this option is present, then you must specify the path to a trained model using `-pretrained` argument.

- `pretrained`: The path to a pretrained model file. If this path exists, then the code will load a pretrained model before starting the train or test process.
The filename is expected to have the form `model_*.chkpnt`. The directory containing this file is expected to also contain the optimizer as `opt_*.chkpnt`, since it will also be loaded. 


## Training `HypE` and `HSimplE` 
You can train by running the following from within Docker (the values provided below are the ones used to obtain the results in the paper):
```console
python main.py -model HypE -dataset JF17K -num_iterations 1000 -batch_size 128 -lr 0.1  -filt_w 1 -out_channels 6 -stride 2 -emb_dim 200 -nr 10
```
```console
python main.py -model HSimplE -dataset JF17K -num_iterations 1000 -batch_size 128 -lr 0.01 -emb_dim 200 -nr 10
```

## Testing a pretrained model
You can test a pretrained model by running the following:
```console
python main.py -model HSimplE -dataset JF17K -pretrained output/my_pretrained_model.chkpnt -test
```


## Baselines

The baselines implemented in this package are `m-DistMult`, `m-CP`, and `m-TransH`. You can train them by running the following:

```console
python main.py -model MTransH -dataset JF17K -num_iterations 1000 -batch_size 128 -lr 0.06 -emb_dim 200 -nr 10
```
```console
python main.py -model MCP -dataset JF17K -num_iterations 1000 -batch_size 128 -lr 0.02 -emb_dim 34 -nr 10
```
```console
python main.py -model MDitMult -dataset JF17K -num_iterations 1000 -batch_size 128 -lr 0.02 -emb_dim 200 -nr 10
```




Contact
=======

Bahare Fatemi

Computer Science Department

The University of British Columbia

201-2366 Main Mall, Vancouver, BC, Canada (V6T 1Z4)  

<bfatemi@cs.ubc.ca>


License
=======

Licensed under the GNU General Public License Version 3.0.
<https://www.gnu.org/licenses/gpl-3.0.en.html>

