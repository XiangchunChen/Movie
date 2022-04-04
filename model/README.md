# Recommendation Model

We use a Deep learning model to analyze the users's interests.

## Environment

**Requirements: **python 3.6, pip3, unzip

Create a new virtual environment on that directory

```shell
python3 -m .venv ./venv 
source ./venv/bin/activate 
pip install -r requirements.txt
```

## Run

1. donwload dataset

```bash
wget https://files.grouplens.org/datasets/movielens/ml-latest-small.zip -O dataset.zip
unzip dataset.zip
export DATA_PATH=/path/to/your/dataset
```

2. traint your model & predict one

```shell
python3 dnmf.py
```

and you will see the below output.

```
User#1 would give *4.4 to the Movie#10
```

see you serialized model at `./dnmf_model_final.h5`

