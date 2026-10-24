<h1 align="center">TEVI: Text-Conditioned Editing of Visual Representations via Sparse Autoencoders for Improved Vision-Language Alignment</h1>

<div align="center">
<a href="https://swetamahajan.github.io">Sweta Mahajan*</a>,
<a href="https://sukrutrao.github.io">Sukrut Rao*</a>,
<a href="https://jiahao000.github.io">Jiahao Xie</a>,
<a href="https://www.coli.uni-saarland.de/~koller">Alexander Koller</a>,
<a href="https://people.mpi-inf.mpg.de/~schiele">Bernt Schiele</a>

<h3 align="center">
EMNLP Findings 2026
</h3>
</div>

<h3 align="center">
<a href="https://arxiv.org/abs/2606.07451">Paper</a>
|
<a href="https://github.com/neuroexplicit-saar/TEVI">Code</a>
</h3>
</p>

This repository contains the code to train and evaluate CG-SAE and TEVI on the synthetic MAD
dataset. The code for the experiments on real data will follow.

## Installation

Create the environment and install the dependencies and the code of this repository:

```bash
conda create -n tevi python=3.10 -y
conda activate tevi
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1
pip install -e ./open_clip
pip install -e ./openai-topk-sparse-autoencoders
pip install -e .
```

## Data

Generate MAD from MNIST. The first command writes the training set and the second the test
set, both to `/path/to/morphoMNIST_0`:

```bash
python MAD/prepare.py --out_dir /path/to/morphoMNIST --root /path/to/mnist --seed 0
python MAD/prepare.py --out_dir /path/to/morphoMNIST --root /path/to/mnist --seed 0 --test
```

## Training and evaluation

Copy `configs/paths.example.env` to `configs/paths.env` and set the paths. The steps below
run in this order.

Train the CLIP model on MAD:

```bash
bash experiments/mad/01_train_clip.sh
```

Save the CLIP features of the images and captions, which the next steps use:

```bash
bash experiments/mad/02_save_features.sh
```

Train the CG-SAE and the SAE with a learnt decoder:

```bash
bash experiments/mad/03_train_sae.sh cgsae
bash experiments/mad/03_train_sae.sh tevi
```

Train the text-conditioned mask on each SAE, for five seeds. The mask on the SAE with a learnt
decoder is TEVI:

```bash
bash experiments/mad/04_train_mask.sh cgsae
bash experiments/mad/04_train_mask.sh tevi
```

Evaluate the masks. For each attribute, this prints the accuracy when the caption names the
attribute and when it does not, averaged over the five seeds:

```bash
bash experiments/mad/05_eval_seed_tables.sh cgsae
bash experiments/mad/05_eval_seed_tables.sh tevi
```

## Third-party code

This repository includes modified copies of [OpenCLIP](https://github.com/mlfoundations/open_clip),
[OpenAI's sparse autoencoder](https://github.com/openai/sparse_autoencoder) and
[Morpho-MNIST](https://github.com/dccastro/Morpho-MNIST), which are released under the MIT
licence.

## Citation

```bibtex
@inproceedings{mahajan2026tevi,
  title     = {{TEVI}: Text-Conditioned Editing of Visual Representations via Sparse Autoencoders for Improved Vision-Language Alignment},
  author    = {Mahajan, Sweta and Rao, Sukrut and Xie, Jiahao and Koller, Alexander and Schiele, Bernt},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2026},
  year      = {2026}
}
```
