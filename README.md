####  Environment

```python
pip install -r requirement.txt
```

####  Dataset

download movielens-1m and movielens-100k

then run

```
cd data/movielens_1m/ood_generate_dataset_tiny_10_30u30i/
python preprocess.py
```

####  Train

```
cd code/scripts/
python duet_with_log.sh {dataset} {model} {cuda}
```

example:

```
python duet_with_log.sh movielens_1m din_gau 0
```

