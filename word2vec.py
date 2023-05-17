import pandas as pd, numpy as np
from gensim.models import Word2Vec
from cuml.neighbors import NearestNeighbors


def train(args):
    df = []
    for p in args.input.split(","):
        df.append(pd.read_parquet(p))
    df = pd.concat(df, ignore_index=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")