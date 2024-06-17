# Image Captioning Evaluation

## Getting Data

To get the datasets, please run any combination of the following commands:

```bash
python scripts/download_flickr8k.py
python scripts/download_mscoco.py
python scripts/download_pascal50s.py
```

## Getting Embeddings

### OpenAI

To get the OpenAI embeddings for the datasets, run the following commands, after completing the step "Getting Data":
```bash
python scripts/get_embeddings_openai.py
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
