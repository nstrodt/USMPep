# USMPep: Universal Sequence Models for Major Histocompatibility Complex Binding Afﬁnity Prediction 
**USMPep** is a simple recurrent neural network for MHC binding afﬁnity prediction. It is competitive with state-of-the-art tools for a single model trained from scratch, while ensembling multiple regressors and language model pretraining can slightly improve its performance.
In our paper we report the excellent predictive performance of **USMPep** on several benchmark datasets.

For a detailed description of technical details and experimental results, please refer to our paper:

[USMPep: Universal Sequence Models for Major Histocompatibility Complex Binding Afﬁnity Prediction](https://doi.org/10.1186/s12859-020-03631-1)

Johanna Vielhaben, Markus Wenzel, Wojciech Samek, and Nils Strodthoff

 	@article{Vielhaben:2020USMPep,
	author = {Vielhaben, Johanna and Wenzel, Markus and Samek, Wojciech and Strodthoff, Nils},
	title = {{USMPep: Universal Sequence Models for Major Histocompatibility Complex Binding Affinity Prediction}},
	journal = {BMC Bioinformatics},
	year = {2020},
	month={Jul},
	volume = {21},
	number = {1},
	pages={279},
	issn={1471-2105},
	doi = {10.1186/s12859-020-03631-1},
	url= {https://doi.org/10.1186/s12859-020-03631-1}
	}

This is the accompanying code repository where we also provide a pretrained language model and predictions of our models on the test datasets discussed in our paper.

We present an extended version of **USMPep**, that we evaluated on a recent SARS-CoV-2 dataset, in our paper:

[Predicting the Binding of SARS-CoV-2 Peptides to the Major Histocompatibility Complex with Recurrent Neural Networks](https://arxiv.org/abs/2104.08237)

Johanna Vielhaben, Markus Wenzel, Eva Weicken, Nils Strodthoff

	@misc{Vielhaben:2021USMPep,
	      title={Predicting the Binding of SARS-CoV-2 Peptides to the Major Histocompatibility Complex with Recurrent Neural Networks}, 
	      author={Johanna Vielhaben and Markus Wenzel and Eva Weicken and Nils Strodthoff},
	      year={2021},
	      eprint={2104.08237},
	      archivePrefix={arXiv},
	      primaryClass={q-bio.QM}
	}

**USMPep** builds on the [UDSMProt](https://github.com/nstrodt/UDSMProt)-framework: [Universal Deep Sequence Models for Protein Classification](https://doi.org/10.1101/704874)
## Dependencies
for training/evaluation: `pytorch` `fastai` `fire` 

for dataset creation: `numpy` `pandas` `scikit-learn` `biopython` `sentencepiece` `lxml`

## Installation
We recommend using conda as Python package and environment manager.
Either install the environment using the provided `proteomics.yml` by running `conda env create -f proteomics.yml` or follow the steps below:
1. Create conda environment: `conda create -n proteomics` and `conda activate proteomics`
2. Install pytorch: `conda install pytorch -c pytorch`
3. Install fastai: `conda install -c fastai fastai=1.0.52`
4. Install fire: `conda install fire -c conda-forge`
5. Install scikit-learn: `conda install scikit-learn`
6. Install Biopython: `conda install biopython -c conda-forge`
7. Install sentencepiece: `pip install sentencepiece`
8. Install lxml: `conda install lxml`

Optionally (for support of threshold 0.4 clusters) install [cd-hit](`https://github.com/weizhongli/cdhit`) and add `cd-hit` to the default searchpath.

## Usage
See the [USMPep User Guide](./code/USMPep_UserGuide.ipynb) for extensive usage information.

A second [User Guide](./code/USMPep_SARS-CoV-2.ipynb) provides usage information for the extended version of **USMPep**.

## Binding Affinity Predictions
We provide peptide binding affinity predictions for our tools, see `git-data`-folder and the corresponding [readme file](./git_data/README.md) for details.
