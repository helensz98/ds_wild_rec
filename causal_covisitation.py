# ref https://www.kaggle.com/code/vslaykovsky/co-visitation-matrix
# https://www.kaggle.com/code/radek1/co-visitation-matrix-simplified-imprvd-logic
# https://www.kaggle.com/code/cdeotte/candidate-rerank-model-lb-0-575#Step-2---ReRank-(choose-20)-using-handcrafted-rules
# Note: the author of the third notebook used test data for training and achieved lb 0.575
# (competition holder said ok) but I don't think it's good practice so no test in train for me.  

import pandas as pd, numpy as np
import os, gc
from collections import Counter
import cudf, itertools

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


def create_causal_cart_order_covisitation(data, causal=5, time_window=24 * 60 * 60, tail=30, k=20, type_weight={0:1, 1:6, 2:3}):
    num_aid_part = 4
    size_aid_part = 1.86e6 / num_aid_part  # 1855603
    num_session_part = 100
    size_session_part = (data.session.max() + 1e4) / num_session_part
    res = []
    for aid_step in range(num_aid_part):
        tmp = None
        for session_step in range(num_session_part):
            df = cudf.DataFrame(
                data.loc[(data.session >= session_step * size_session_part) & (
                            data.session < (session_step + 1) * size_session_part)])
            df = df.sort_values(['session', 'ts'], ascending=[True, False])
            # don't use tail of session
            df = df.reset_index(drop=True)
            df['n'] = df.groupby('session').cumcount()
            # create pairs
            df = df.merge(df, on='session')
            df = df.loc[((df.n_x - df.n_y) < 0) & ((df.n_x - df.n_y) >= -causal) & (df.aid_x != df.aid_y)]
            # memory management compute in parts
            df = df.loc[(df.aid_x >= aid_step * size_aid_part) & (df.aid_x < (aid_step + 1) * size_aid_part)]
            # assign weights
            df = df[['session', 'aid_x', 'aid_y', 'type_y']].drop_duplicates(['session', 'aid_x', 'aid_y'])
            df['wgt'] = df.type_y.map(type_weight)
            df = df[['aid_x', 'aid_y', 'wgt']]
            df.wgt = df.wgt.astype('float32')
            df = df.groupby(['aid_x', 'aid_y']).wgt.sum()
            # combine inner chunks
            if tmp is None:
                tmp = df
            else:
                tmp = tmp.add(df, fill_value=0)
            print(f"aid {aid_step}/{num_aid_part} session {session_step}/{num_session_part}", end='\r')
        del df
        gc.collect()
        # convert matrix to dictionary
        tmp = tmp.reset_index()
        tmp = tmp.sort_values(['aid_x', 'wgt'], ascending=[True, False])

        tmp = tmp.reset_index(drop=True)
        tmp['n'] = tmp.groupby('aid_x').aid_y.cumcount()
        tmp = tmp.loc[tmp.n < k].drop('n', axis=1)
        # save part to disk (convert to pandas first uses less memory)
        tmp = tmp.to_pandas()
        res.append(tmp)
    res = pd.concat(res, ignore_index=True)
    return res

def create_causal_buy2buy_covisitation(data, causal=5, time_window=14 * 24 * 60 * 60, tail=30, k=20):
    num_aid_part = 4
    size_aid_part = 1.86e6 / num_aid_part

    num_session_part = 100
    size_session_part = (data.session.max() + 1e4) / num_session_part
    res = []

    for aid_step in range(num_aid_part):
        tmp = None
        for session_step in range(num_session_part):
            df = cudf.DataFrame(
                data.loc[(data.session >= session_step * size_session_part) & (
                        data.session < (session_step + 1) * size_session_part)])
            df = df.loc[df['type'].isin([1, 2])]  # ONLY WANT CARTS AND ORDERS
            df = df.sort_values(['session', 'ts'], ascending=[True, False])
            df = df.reset_index(drop=True)
            df['n'] = df.groupby('session').cumcount()
            # create pairs
            df = df.merge(df, on='session')
            df = df.loc[((df.n_x - df.n_y) < 0) & ((df.n_x - df.n_y) >= -causal) & (df.aid_x != df.aid_y)]
            # memory management compute in parts
            df = df.loc[(df.aid_x >= aid_step * size_aid_part) & (df.aid_x < (aid_step + 1) * size_aid_part)]
            # assign weights
            df = df[['session', 'aid_x', 'aid_y', 'type_y']].drop_duplicates(['session', 'aid_x', 'aid_y'])
            df['wgt'] = 1
            df = df[['aid_x', 'aid_y', 'wgt']]
            df.wgt = df.wgt.astype('float32')
            df = df.groupby(['aid_x', 'aid_y']).wgt.sum()
            # combine inner chunks
            if tmp is None:
                tmp = df
            else:
                tmp = tmp.add(df, fill_value=0)
            print(f"aid {aid_step}/{num_aid_part} session {session_step}/{num_session_part}", end='\r')
        # convert matrix to dictionary
        tmp = tmp.reset_index()
        tmp = tmp.sort_values(['aid_x', 'wgt'], ascending=[True, False])
        tmp = tmp.reset_index(drop=True)
        tmp['n'] = tmp.groupby('aid_x').aid_y.cumcount()
        tmp = tmp.loc[tmp.n < k].drop('n', axis=1)
        # save part to disk (convert to pandas first uses less memory)
        tmp = tmp.to_pandas()
        res.append(tmp)
    res = pd.concat(res, ignore_index=True)
    return res

def create_causal_click_covisitation(data, causal=5, time_window=24 * 60 * 60, tail=30, k=20):
    num_aid_part = 4
    size_aid_part = 1.86e6 / num_aid_part

    num_session_part = 100
    size_session_part = (data.session.max() + 1e4) / num_session_part
    res = []

    for aid_step in range(num_aid_part):
        tmp = None
        for session_step in range(num_session_part):
            df = cudf.DataFrame(
                data.loc[(data.session >= session_step * size_session_part) & (
                        data.session < (session_step + 1) * size_session_part)])
            df = df.sort_values(['session', 'ts'], ascending=[True, False])
            # don't use tail of session
            df = df.reset_index(drop=True)
            df['n'] = df.groupby('session').cumcount()
            # create pairs
            df = df.merge(df, on='session')
            df = df.loc[((df.n_x - df.n_y) < 0) & ((df.n_x - df.n_y) >= -causal) & (df.aid_x != df.aid_y)]
            # memory management compute in parts
            df = df.loc[(df.aid_x >= aid_step * size_aid_part) & (df.aid_x < (aid_step + 1) * size_aid_part)]
            # assign WEIGHTS
            df = df[['session', 'aid_x', 'aid_y', 'ts_x', 'ts_y']].drop_duplicates(['session', 'aid_x', 'aid_y'])
            df['wgt'] = 1 + 3 * (df.ts_x - 1659304800) / (1662328791 - 1659304800)
            df = df[['aid_x', 'aid_y', 'wgt']]
            df.wgt = df.wgt.astype('float32')
            df = df.groupby(['aid_x', 'aid_y']).wgt.sum()
            # combine inner chunks
            if tmp is None:
                tmp = df
            else:
                tmp = tmp.add(df, fill_value=0)
            print(f"aid {aid_step}/{num_aid_part} session {session_step}/{num_session_part}", end='\r')
        # convert matrix to dictionary
        tmp = tmp.reset_index()
        tmp = tmp.sort_values(['aid_x', 'wgt'], ascending=[True, False])
        # save top k
        tmp = tmp.reset_index(drop=True)
        tmp['n'] = tmp.groupby('aid_x').aid_y.cumcount()
        tmp = tmp.loc[tmp.n < k].drop('n', axis=1)
        # save part to disk (convert to pandas first uses less memory)
        tmp = tmp.to_pandas()
        res.append(tmp)
    res = pd.concat(res, ignore_index=True)
    return res

# single
def suggest_aids(cov, fill, df):
    top_cov = cov_map[cov]
    top_fill = top_clicks if fill=="top_clicks" else top_orders
    # user history aids and types
    aids=df.aid.tolist()
    types = df.type.tolist()
    type_weight_multipliers = {0: 1, 1: 6, 2: 3}
    unique_aids = list(dict.fromkeys(aids[::-1]))
    if len(unique_aids)>=20:
        weights=np.logspace(0.1,1,len(aids),base=2, endpoint=True)-1
        aids_temp = Counter() 
        # rerank based on repeat items and type of items
        for aid,w,t in zip(aids,weights,types): 
            aids_temp[aid] += w * type_weight_multipliers[t]
        sorted_aids = [k for k,v in aids_temp.most_common(20)]
        return sorted_aids
    aids2 = list(itertools.chain(*[top_cov[aid] for aid in unique_aids if aid in top_cov]))
    top_aids2 = [aid2 for aid2, cnt in Counter(aids2).most_common(K) if aid2 not in unique_aids]
    result = unique_aids + top_aids2[:20 - len(unique_aids)]
    return result + list(top_fill)[:20-len(result)]

# rerank
def suggest_clicks(df):
    # user history aids and types
    aids=df.aid.tolist()
    types = df.type.tolist()
    type_weight_multipliers = {0: 1, 1: 6, 2: 3}
    unique_aids = list(dict.fromkeys(aids[::-1]))
    if len(unique_aids)>=20:
        weights=np.logspace(0.1,1,len(aids),base=2, endpoint=True)-1
        aids_temp = Counter() 
        # rerank based on repeat item and type of items
        for aid,w,t in zip(aids,weights,types): 
            aids_temp[aid] += w * type_weight_multipliers[t]
        sorted_aids = [k for k,v in aids_temp.most_common(20)]
        return sorted_aids
    aids2 = list(itertools.chain(*[top_click[aid] for aid in unique_aids if aid in top_click]))
    top_aids2 = [aid2 for aid2, cnt in Counter(aids2).most_common(K) if aid2 not in unique_aids]
    result = unique_aids + top_aids2[:20 - len(unique_aids)]
    return result + list(top_clicks)[:20-len(result)]

# rerank
def suggest_buys(df):
    # user history aids and types
    aids=df.aid.tolist()
    types = df.type.tolist()
    type_weight_multipliers = {0: 1, 1: 6, 2: 3}
    # unique aids and unique buys
    unique_aids = list(dict.fromkeys(aids[::-1] ))
    df = df.loc[(df['type']==1)|(df['type']==2)]
    unique_buys = list(dict.fromkeys( df.aid.tolist()[::-1] ))
    # rerank candidates using weights
    if len(unique_aids)>=20:
        weights=np.logspace(0.5,1,len(aids),base=2, endpoint=True)-1
        aids_temp = Counter() 
        # rerank based on repeat items and type of items
        for aid,w,t in zip(aids,weights,types): 
            aids_temp[aid] += w * type_weight_multipliers[t]
        # rerank candidates using "BUY2BUY" co-visitation matrix
        aids3 = list(itertools.chain(*[top_buy2buy[aid] for aid in unique_buys if aid in top_buy2buy]))
        for aid in aids3:
            aids_temp[aid] += 0.1
        sorted_aids = [k for k,v in aids_temp.most_common(20)]
        return sorted_aids
    aids2 = list(itertools.chain(*[top_cart_order[aid] for aid in unique_aids if aid in top_cart_order]))
    # use "BUY2BUY" co-visitation matrix
    aids3 = list(itertools.chain(*[top_buy2buy[aid] for aid in unique_buys if aid in top_buy2buy]))
    # rerank candidates
    top_aids2 = [aid2 for aid2, cnt in Counter(aids2+aids3).most_common(K) if aid2 not in unique_aids]
    result = unique_aids + top_aids2[:20 - len(unique_aids)]
    return result + list(top_orders)[:20-len(result)]


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


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--model", type=str)
    parser.add_argument("--save", default=None)
    parser.add_argument("--k", default=20, type=int)
    parser.add_argument("--causal", default=5, type=int)
    parser.add_argument("--label", default=None, type=str)
    parser.add_argument("--train", action='store_true', default=False)
    parser.add_argument("--pred", action='store_true', default=False)
    parser.add_argument("--single", action='store_true', default=False)
    args = parser.parse_args()
    CAUSAL = 5

    if args.train:
        print("===========Train causal covisitation ============")
        save_dir = Path(args.model)
        save_dir.mkdir(exist_ok=True, parents=True)
        data = []
        for p in args.input.split(","):
            data.append(pd.read_parquet(p))
        data = pd.concat(data, ignore_index=True)
        data['ts'] = data['ts'].astype(int) / 1e9
        print("create causal cart order covisitation")
        top_cart_order = create_causal_cart_order_covisitation(data, CAUSAL)
        print("create causal buy2buy covisitation")
        top_buy2buy = create_causal_buy2buy_covisitation(data, CAUSAL)
        print("create causal click covisitation")
        top_click = create_causal_click_covisitation(data, CAUSAL)

        top_click.to_parquet(save_dir / "causal_top_click.parquet")
        top_buy2buy.to_parquet(save_dir / "causal_top_buy2buy.parquet")
        top_cart_order.to_parquet(save_dir / "causal_top_cart_order.parquet")

    if args.pred:
        print(f"===========causal covisitation predict {args.input}===========")
        save = Path(args.save)
        save.parent.mkdir(exist_ok=True, parents=True)
        K = args.k
        print(f"generate {K} candidates")
        print(f"load covisitation")
        top_click = pd.read_parquet(os.path.join(args.model, "causal_top_click.parquet"))  # top_20_clicks
        top_buy2buy = pd.read_parquet(os.path.join(args.model, "causal_top_buy2buy.parquet"))  # top_20_buy2buy
        top_cart_order = pd.read_parquet(os.path.join(args.model, "causal_top_cart_order.parquet")) # top_20_buys

        top_click = top_click.groupby("aid_x")['aid_y'].apply(list).to_dict()
        top_buy2buy = top_buy2buy.groupby("aid_x")['aid_y'].apply(list).to_dict()
        top_cart_order = top_cart_order.groupby("aid_x")['aid_y'].apply(list).to_dict()

        test_df = pd.read_parquet(args.input)
        test_df['ts'] = test_df['ts'].astype(int) / 1e9
        # top clicks and orders in test
        top_clicks = test_df.loc[test_df['type']=='clicks','aid'].value_counts().index.values[:20]
        top_orders = test_df.loc[test_df['type']=='orders','aid'].value_counts().index.values[:20]
        
        if args.single:
            print(f"===========single mode prediction===========")
            cov_map = {"top_click":top_click, "top_buy2buy":top_buy2buy, "top_cart_order":top_cart_order}
            
            print(f"suggest click candidates")
            pred_df_click = test_df.sort_values(["session", "ts"]).groupby(["session"]).apply(
                lambda x: suggest_aids("top_click", "top_orders", x)
            )
            
            decode_click = pred_df_click.reset_index()
            decode_click.columns = ['session', 'aid']

            click_only_click_cand = decode_candidates(decode_click, act=0)
            click_only_cart_cand = decode_candidates(decode_click, act=1)
            click_only_order_cand = decode_candidates(decode_click, act=2)

            click_only_cand = pd.concat([click_only_click_cand, click_only_cart_cand, click_only_order_cand])
            click_only_cand.to_parquet(str(save)+"/causal_click_only.parquet")
            print(f"click covisitation candidates saved in {str(save)}/causal_click_only")
            
            print(f"suggest buy2buy candidates")
            pred_df_buy2buy = test_df.sort_values(["session", "ts"]).groupby(["session"]).apply(
                lambda x: suggest_aids("top_buy2buy", "top_orders", x)
            )
            
            decode_buy2buy = pred_df_buy2buy.reset_index()
            decode_buy2buy.columns = ['session', 'aid']

            buy2buy_only_click_cand = decode_candidates(decode_buy2buy, act=0)
            buy2buy_only_cart_cand = decode_candidates(decode_buy2buy, act=1)
            buy2buy_only_order_cand = decode_candidates(decode_buy2buy, act=2)

            buy2buy_only_cand = pd.concat([buy2buy_only_click_cand, buy2buy_only_cart_cand, buy2buy_only_order_cand])
            buy2buy_only_cand.to_parquet(str(save)+"/causal_buy2buy_only.parquet")
            print(f"buy2buy covisitation candidates saved in {str(save)}/causal_buy2buy_only")
            
            print(f"suggest cart_order candidates")
            pred_df_cart_order = test_df.sort_values(["session", "ts"]).groupby(["session"]).apply(
                lambda x: suggest_aids("top_cart_order", "top_orders", x)
            )
            
            decode_cart_order = pred_df_cart_order.reset_index()
            decode_cart_order.columns = ['session', 'aid']

            cart_order_only_click_cand = decode_candidates(decode_cart_order, act=0)
            cart_order_only_cart_cand = decode_candidates(decode_cart_order, act=1)
            cart_order_only_order_cand = decode_candidates(decode_cart_order, act=2)

            cart_order_only_cand = pd.concat([cart_order_only_click_cand, cart_order_only_cart_cand, cart_order_only_order_cand])
            cart_order_only_cand.to_parquet(str(save)+"/causal_cart_order_only.parquet")
            print(f"cart_order covisitation candidates saved in {str(save)}/cuasal_cart_order_only")
            
        else:
            print(f"suggest click candidates")
            pred_df_clicks = test_df.sort_values(["session", "ts"]).groupby(["session"]).apply(
                lambda x: suggest_clicks(x)
            )
            print(f"suggest order candidates")
            pred_df_buys = test_df.sort_values(["session", "ts"]).groupby(["session"]).apply(
                lambda x: suggest_buys(x)
            )

            decode_clicks = pred_df_clicks.reset_index()
            decode_clicks.columns = ['session', 'aid']

            decode_buys = pred_df_buys.reset_index()
            decode_buys.columns = ['session', 'aid']

            click_cand = decode_candidates(decode_clicks, act=0)
            cart_cand = decode_candidates(decode_buys, act=1)
            order_cand = decode_candidates(decode_buys, act=2)

            cand = pd.concat([click_cand, cart_cand, order_cand])
            cand.to_parquet(str(save))
            print(f"saved in {str(save)}")

            if args.label:
                label = pd.read_parquet(args.label)
                get_score(cand, label)
