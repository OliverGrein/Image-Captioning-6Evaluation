# Image Captioning Evaluation

## What this project is about

//ToDo

## Quickstart

To get started, create a new environment and install the requirements:

```bash
python -m venv .venv
source venv/bin/activate
pip install -r requirements.txt
```

Then, pull the data from huggingface using the bootstrap script, which will put the data inside the `data` folder.:

```bash
python scripts/bootstrap.py
```

If you want to source the data and create the embeddings yourself, please follow the steps in the [Data](#data) section.

## Data

### Getting Data

To get the datasets, please run any combination of the following commands:

```bash
python scripts/download/download_flickr8k.py
python scripts/download/download_mscoco.py
python scripts/download/download_pascal50s.py
```

### Getting Embeddings

To create embeddings yourself, first you need to create a `.env` file with the following variables (depending on which api you want to use).

#### OpenAI

```env
OPENAI_API_KEY=<your-openai-api-key>
```

To get the OpenAI embeddings for the datasets, run the following commands, after completing the step [Getting Data](#getting-data):

```bash
python scripts/embeddings/get_openai_embeddings.py
```

#### VoyageAI

```env
VOYAGE_API_KEY=<your-voyageai-api-key>
```

To get the VoyageAI embeddings for the datasets, run the following commands, after completing the step [Getting Data](#getting-data):

```bash
python scripts/embeddings/get_voyageai_embeddings.py
```

#### Cohere

```env
COHERE_API_KEY=<your-cohere-api-key>
```

To get the Cohere embeddings for the datasets, run the following commands, after completing the step [Getting Data](#getting-data):

```bash
python scripts/embeddings/get_cohere_embeddings.py
```

#### VertexAI

**Please make sure you followed [Getting Data](#getting-data) before following this**

To use VertextAI please register the service for you project. Followingly you'll need to install the [gc-cli](https://cloud.google.com/sdk/docs/install?hl=de) and consequently run `gcloud auth applicaiton-default login`

```bash
python scripts/embeddings/get_vertexai_embeddings.py
```

## References

### PASCAL-50S

Cyrus Rashtchian, Peter Young, Micah Hodosh, and Julia Hockenmaier. Collecting Image Annotations Using Amazon's Mechanical Turk. In Proceedings of the NAACL HLT 2010 Workshop on Creating Speech and Language Data with Amazon's Mechanical Turk.

### Flickr-8k

M. Hodosh, P. Young and J. Hockenmaier (2013) "Framing Image Description as a Ranking Task: Data, Models and Evaluation Metrics", Journal of Artifical Intellegence Research, Volume 47, pages 853-899
<http://www.jair.org/papers/paper3994.html>

### MS-COCO

@article{DBLP:journals/corr/LinMBHPRDZ14,
author    = {Tsung{-}Yi Lin and Michael Maire and Serge J. Belongie and Lubomir D. Bourdev and Ross B. Girshick and James Hays and Pietro Perona and Deva Ramanan and Piotr Doll{'{a}}r and C. Lawrence Zitnick},
title     = {Microsoft {COCO:} Common Objects in Context},
journal   = {CoRR},
volume    = {abs/1405.0312},
year      = {2014},
url       = {<http://arxiv.org/abs/1405.0312>},
archivePrefix = {arXiv},
eprint    = {1405.0312},
timestamp = {Mon, 13 Aug 2018 16:48:13 +0200},
biburl    = {<https://dblp.org/rec/bib/journals/corr/LinMBHPRDZ14>},
bibsource = {dblp computer science bibliography, <https://dblp.org>}
}
