# Multimodal Causal Representation Learning


## Installation

This project was developed with Python 3.10 and PyTorch 1.13.1. The
dependencies can be installed as follows:

```bash
# install dependencies (preferably, inside your conda/virtual environment)
$ pip install -r requirements.txt

Created environment: mm_contra_learn

# test if pytorch was installed with cuda support; should not raise an error
$ python -c "import torch; assert torch.cuda.device_count() > 0, 'No cuda support'"
```

## Numerical Simulation

```bash
# train a model on data with style-change probability 0.75 and with statistical and 
# causal dependencies, and save it to the directory "models/mlp_example"
$ python main_mlp.py --model-id "mlp_example"     \
                     --style-change-prob 0.75     \
                     --statistical-dependence     \
                     --content-dependent-style

# evaluate the trained model
$ python main_mlp.py --model-id "mlp_example" --load-args --evaluate
```

## Image/Text Experiment

```bash
# download and extract the dataset
$ wget https://zenodo.org/record/7678231/files/m3di.tar.gz
$ tar -xzf m3di.tar.gz

# train a model with encoding size 4 and save it to the directory "models/imgtxt_example"
$ python main_imgtxt.py --datapath "m3di" --model-id "imgtxt_example" --encoding-size 4

# evaluate the trained model
$ python main_imgtxt.py --datapath "m3di" --model-id "imgtxt_example" --load-args --evaluate
```

## BibTeX
This code is adapted from the code provided by the following paper:

```bibtex
@article{daunhawer2023multimodal,
  author = {
    Daunhawer, Imant and
    Bizeul, Alice and
    Palumbo, Emanuele and
    Marx, Alexander and
    Vogt, Julia E.
  },
  title = {Identifiability Results for Multimodal Contrastive Learning},
  booktitle = {International Conference on Learning Representations},
  year = {2023}
}
```

## Acknowledgements

This project builds on the following resources. Please cite them appropriately.
- https://github.com/imantdaunhawer/multimodal-contrastive-learning <3
- https://github.com/ysharma1126/ssl_identifiability <3
- https://github.com/brendel-group/cl-ica <3
