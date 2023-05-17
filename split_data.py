import pandas as pd, numpy as np
import random
import os
from pathlib import Path


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    print(f"set random seed {seed}")


def create_samples(df):
    def cut_seq(df):
        sess_cnt = df.groupby("session")['aid'].count().reset_index()
        sess_cnt.columns = ['session', 'length']

        def _cut(x):
            if x == 1:
                return 1
            else:
                return np.random.randint(1, x)

        sess_cnt['cut'] = sess_cnt['length'].apply(_cut)
        sess_only_one = sess_cnt[sess_cnt.length == 1].session.tolist()
        return sess_cnt, sess_only_one
    df_cut, sess_only_one = cut_seq(df)
    print(f"{len(sess_only_one)} contains only one action, will dropped")
    df = df[~df.session.isin(sess_only_one)].reset_index(drop=True)

    df["n"] = 1
    df["n"] = df.groupby("session")['n'].cumsum()
    df = df.merge(df_cut[['session', 'cut']], on='session', how='left')

    test_input = df.loc[df.n <= df.cut].reset_index(drop=True)
    del test_input['n']
    del test_input['cut']
    test_label = df.loc[df.n > df.cut].reset_index(drop=True)

    test_label = test_label.groupby(['session', 'type'])['aid'].apply(list)
    test_label = test_label.reset_index().rename(columns={'aid': 'labels'})
    test_label.loc[test_label.type == 0, 'labels'] = test_label.loc[test_label.type == 0, 'labels'].str[:1]
    print(f"inputs shape {test_input.shape}, labels shape {test_label.shape}")
    return test_input, test_label, sess_only_one


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--save', type=str)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--sample', type=float, default=1)
    parser.add_argument('--submission', action='store_true', default=False)
    args = parser.parse_args()
    print(f"=============SPLIT DATA==============")
    seed_everything(args.seed)
    dir = Path(args.save)
    dir.mkdir(exist_ok=True, parents=True)

    df = pd.read_parquet(args.input)

    if args.submission:
        print("create samples for submission")
        sess_start_ts = df.groupby("session")['ts'].min()
        timeline = ["2022-08-01 00:00:00", "2022-08-08 00:00:00", "2022-08-15 00:00:00", "2022-08-22 00:00:00", "2022-08-29 00:00:00"]
        train_input = []
        train_label = []
        sess_only_one = []
        for start, end in zip(timeline[:-1], timeline[1:]):
            use_sess = sess_start_ts[(sess_start_ts >= start) & (sess_start_ts < end)].index.tolist()
            sub_df = df.loc[df.session.isin(use_sess)].reset_index(drop=True)
            sub_df = sub_df.loc[sub_df.ts < end].reset_index(drop=True)
            print(f"train sub {sub_df.shape} from {sub_df.ts.min()} to {sub_df.ts.max()}, contains {sub_df.session.nunique()} sess")
            sub_input, sub_label, _sess_only_one = create_samples(sub_df)
            train_input.append(sub_input)
            train_label.append(sub_label)
            sess_only_one.extend(_sess_only_one)
        train_input = pd.concat(train_input, ignore_index=True)
        train_label = pd.concat(train_label, ignore_index=True)
        no_train = df.loc[df.session.isin(sess_only_one)].reset_index(drop=True)
        train_input.to_parquet(dir / 'train_input.parquet')
        train_label.to_parquet(dir / 'train_label.parquet')
        no_train.to_parquet(dir / 'no_train.parquet')
    else:
        print("create samples for validation")
        sess_only_one = []
        sess_start_ts = df.groupby("session")['ts'].min()
        val_sess = sess_start_ts[sess_start_ts >= "2022-08-22 00:00:00"].index.tolist()
        train = df.loc[~df.session.isin(val_sess)].reset_index(drop=True)
        train = train.loc[train.ts < "2022-08-22 00:00:00"].reset_index(drop=True)
        val = df.loc[df.session.isin(val_sess)].reset_index(drop=True)
        print(f"train all shape {train.shape} from {train.ts.min()} to {train.ts.max()}, contains {train.session.nunique()} sess")
        print(f"val all shape {val.shape} from {val.ts.min()} to {val.ts.max()}, contains {val.session.nunique()} sess")
        if args.sample < 1:
            print(f"use {args.sample} sampled train data")
            sess = train.session.unique()
            sess = np.random.choice(sess, int(args.sample * len(sess)))
            train = train.loc[train.session.isin(sess)].reset_index(drop=True)
            print(f"train sampled shape {train.shape} from {train.ts.min()} to {train.ts.max()}, contains {train.session.nunique()} sess")
        train.to_parquet(dir/"train.parquet")
        val.to_parquet(dir / "valid.parquet")

        timeline = ["2022-08-01 00:00:00", "2022-08-08 00:00:00", "2022-08-15 00:00:00", "2022-08-22 00:00:00"]
        train_input = []
        train_label = []
        for start, end in zip(timeline[:-1], timeline[1:]):
            use_sess = sess_start_ts[(sess_start_ts >= start) & (sess_start_ts < end)].index.tolist()
            sub_df = train.loc[train.session.isin(use_sess)].reset_index(drop=True)
            sub_df = sub_df.loc[sub_df.ts < end].reset_index(drop=True)
            print(f"train sub {sub_df.shape} from {sub_df.ts.min()} to {sub_df.ts.max()}, contains {sub_df.session.nunique()} sess")
            sub_input, sub_label, _sess_only_one = create_samples(sub_df)
            train_input.append(sub_input)
            train_label.append(sub_label)
            sess_only_one.extend(_sess_only_one)

        no_train = df.loc[df.session.isin(sess_only_one)].reset_index(drop=True)
        train_input = pd.concat(train_input, ignore_index=True)
        train_label = pd.concat(train_label, ignore_index=True)
        train_input.to_parquet(dir / 'train_input.parquet')
        train_label.to_parquet(dir / 'train_label.parquet')
        no_train.to_parquet(dir / 'no_train.parquet')

        val_input, val_label, _ = create_samples(val)
        val_input.to_parquet(dir / 'valid_input.parquet')
        val_label.to_parquet(dir / 'valid_label.parquet')
