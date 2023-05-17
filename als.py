import pandas as pd, numpy as np
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares # tend to overfit


def decode_candidates(data, act):
    # data contains session, aid cols
    session = []
    aid = []
    for _, row in data.iterrows():
        aid.extend(row.aid)
        session.extend([row.session] * len(row.aid))

    res = pd.DataFrame({"session": session, "aid": aid}).astype("int32")
    res['type'] = act
    res['type'] = res['type'].astype('int8')
    return res


def get_score(pred_df, gt_df):
    score = 0
    score_dict = {}
    weights = {'clicks': 0.10, 'carts': 0.30, 'orders': 0.60}
    for type_id, type_name in enumerate(['clicks', 'carts', 'orders']):
        sub = pred_df.loc[pred_df.type == type_id].groupby("session")['aid'].apply(list).reset_index()
        sub.columns = ['session', 'aid']

        test_labels = gt_df.loc[gt_df['type'] == type_id].copy()
        test_labels = test_labels.merge(sub, how='left', on=['session'])

        test_labels = test_labels.dropna()
        test_labels['hits'] = test_labels.apply(lambda df: len(set(df.aid).intersection(set(df.labels))),
                                                axis=1)
        test_labels['gt_count'] = test_labels.labels.str.len().clip(0, 20)
        recall = test_labels['hits'].sum() / test_labels['gt_count'].sum()
        score += weights[type_name] * recall
        score_dict[type_name] = recall
        print(f'{type_name} recall = {recall:.4f}')
    print(f'Overall Recall = {score:.4f}')
    return score, score_dict



if __name__ == "__main__":

    import argparse
    import gc
    from pathlib import Path
    from tqdm import tqdm
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--input')
    parser.add_argument('--label', default=None)
    parser.add_argument('--save', default=None)
    parser.add_argument('--k', default=20, type=int)
    parser.add_argument('--factors', default=300, type=int)
    parser.add_argument('--seed', default=22, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--regularization', default=0.01, type=float)
    parser.add_argument('--iterations', default=10000, type=int) 
    parser.add_argument("--train", action='store_true', default=False)
    parser.add_argument("--pred", action='store_true', default=False)
    args = parser.parse_args()
    als = AlternatingLeastSquares(args.factors, args.lr, args.regularization, iterations=args.iterations,
                                      random_state=args.seed)

    if args.train:
        print("===========ALS train===========")
        train_df = pd.concat([pd.read_parquet(path) for path in args.input.split(",")], ignore_index=True)
        train_df['score'] = 1
        train_df = train_df.drop_duplicates(['session', 'aid']).reset_index(drop=True)
        train_coo = coo_matrix((
            train_df['score'].values,
            (
                train_df['session'], # row
                train_df['aid'], # col
            )
        ))
        del train_df; gc.collect()

        train_csr = train_coo.tocsr()
        del train_coo; gc.collect()
        als.fit(train_csr)
        from pathlib import Path
        Path(args.model).parent.mkdir(exist_ok=True, parents=True)
        als.save(args.model)

    if args.pred:
        print(f"===========ALS predict {args.input}===========")
        Path(args.save).parent.mkdir(exist_ok=True, parents=True)
        val_input = pd.read_parquet(args.input)
        als = als.load(args.model)
        val_users = val_input['session'].unique()
        batch_size = 1000000
        num_steps = int(np.ceil(len(val_users) / batch_size))

        val_pred = []
        for step in tqdm(range(num_steps)):
            _val_pred, _val_pred_scores = als.recommend(val_users[step*batch_size: (step+1)*batch_size], None,
                                                        N=args.k, filter_already_liked_items=False)
            val_pred.append(_val_pred)
        val_pred = np.concatenate(val_pred, axis=0)
        pred_df = pd.DataFrame()
        pred_df['session'] = val_users
        pred_df['aid'] = val_pred.tolist()
        
        print("decoding candidates...")
        click_df = decode_candidates(pred_df, 0)
        print("decoding done")
        cart_df = click_df.copy()
        cart_df['type'] = 1
        order_df = click_df.copy()
        order_df['type'] = 2
        pred_df = pd.concat([click_df, cart_df, order_df], ignore_index=True)
        pred_df.to_parquet(args.save)

        if args.label:
            val_label = pd.read_parquet(args.label)
            get_score(pred_df, val_label)
