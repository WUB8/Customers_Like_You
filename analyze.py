"""ML1-ML7: SVD recommender, held-out evaluation, PCA comparison, segments.
Run AFTER python build_matrix.py: python analyze.py
All evaluation uses held-out customer-product pairs not seen at fit time.
"""
from __future__ import annotations
import json
import time
import threading
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import psutil

BASE = Path(__file__).resolve().parent
DATA = BASE / 'data/processed'
RESULTS = BASE / 'results'
FIGURES = BASE / 'figures'
SEED = 42
K_OPTIONS = [10, 20, 50, 100, 200]
TOP_N = 10
CHOSEN_K = 50   # fixed pre-evaluation for modest storage, then check error/variance plot

plt.rcParams.update({'figure.dpi': 140, 'axes.spines.top': False,
                     'axes.spines.right': False, 'font.size': 10,
                     'savefig.bbox': 'tight'})


def measure(fit_fn):
    """Approximate peak process RSS during fit, sampled every 30 milliseconds."""
    process = psutil.Process(os.getpid())
    before = process.memory_info().rss
    peak = [before]
    stop = threading.Event()
    def sample():
        while not stop.wait(0.03):
            try: peak[0] = max(peak[0], process.memory_info().rss)
            except psutil.Error: break
    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    start = time.perf_counter()
    try:
        result = fit_fn()
    finally:
        elapsed = time.perf_counter() - start
        peak[0] = max(peak[0],process.memory_info().rss)
        stop.set()
        thread.join()
    return result, elapsed, peak[0] / 1e6


def reconstruct_error(A, model, method):
    """Exact rank-k projection residual norm without materializing dense A_k."""
    X = model.transform(A)
    norm2 = float(A.multiply(A).sum())
    if method == 'PCA':
        mu = np.asarray(A.mean(axis=0)).ravel()
        centered_norm2 = norm2 - A.shape[0]*float(mu @ mu)
        error2 = centered_norm2 - float(np.square(X).sum())
    else:
        error2 = norm2 - float(np.square(X).sum())
    return float(np.sqrt(max(0.0, error2)/norm2))


def make_embeddings(model):
    """Cosine-normalized product factor vectors, rows represent products."""
    return normalize(model.components_.T.astype(np.float32), norm='l2', axis=1)


def rank_customers(A, product_vectors, customers, topn=10):
    """Scoring by similarity to previously purchased products, excluding seen products."""
    customers = np.array(customers, dtype=np.int32)
    pred = np.empty((len(customers), topn), dtype=np.int32)
    start = time.perf_counter()
    for b in range(0, len(customers), 128):
        batch = customers[b:b+128]
        # sum of item-item cosine similarities: A @ V @ V.T
        profiles = (A[batch] @ product_vectors).astype(np.float32)
        scores = profiles @ product_vectors.T
        known = A[batch].tocsr()
        for t, _customer in enumerate(batch):
            indices = known.indices[known.indptr[t]:known.indptr[t+1]]
            scores[t, indices] = -np.inf
        candidate = np.argpartition(-scores, topn-1, axis=1)[:, :topn]
        weights = np.take_along_axis(scores, candidate, axis=1)
        order = np.argsort(-weights, axis=1, kind='stable')
        pred[b:b+len(batch)] = np.take_along_axis(candidate, order, axis=1)
    return pred, time.perf_counter()-start


def baseline_rank(A, customers, best_sellers_order, topn=10):
    pred = np.empty((len(customers), topn), dtype=np.int32)
    for ix, cust_idx in enumerate(customers):
        already = set(A.indices[A.indptr[cust_idx]:A.indptr[cust_idx+1]])
        suggestions = [int(x) for x in best_sellers_order if int(x) not in already][:topn]
        pred[ix,:] = suggestions
    return pred


def hit_rate_at_10(pred, customers, truth):
    hits = [bool(set(pred[i]) & truth[int(cust)]) for i,cust in enumerate(customers)]
    return float(np.mean(hits)), int(sum(hits))


def label_segments(A, labels, countries, descriptions):
    """Summarize distinct marketing groups; never include real customer identifiers."""
    overall = np.asarray(A.sum(axis=0)).ravel()
    frequency = overall / A.shape[0]
    segment_info = []
    for group in sorted(np.unique(labels)):
        members = np.flatnonzero(labels == group)
        purchases = np.asarray(A[members].sum(axis=0)).ravel()
        percent = purchases / len(members)
        lift = (percent + .01) / (frequency + .01)
        eligible = purchases >= max(8, int(.045 * len(members)))
        distinct_rank = np.argsort(-np.where(
            eligible, percent * np.log2(np.maximum(lift, 1.0)), -1
        ))
        distinctive = [j for j in distinct_rank if eligible[j]][:3]
        if not distinctive:
            distinctive = list(np.argsort(-purchases)[:3])
        popular = np.argsort(-purchases)[:3]
        counts = countries.iloc[members]['Country'].value_counts()
        typical_breadth = float(np.median(np.diff(A[members].indptr)))
        segment_info.append({
            'group': int(group) + 1,
            'customers': len(members),
            'share_percent': float(100 * len(members) / A.shape[0]),
            'distinctive_products': ' | '.join(descriptions.get(int(j), 'Unknown') for j in distinctive),
            'most_bought_products': ' | '.join(descriptions.get(int(j), 'Unknown') for j in popular),
            'top_countries': ', '.join(f'{c} ({ct / len(members):.0%})' for c, ct in counts.head(3).items()),
            'median_distinct_products': typical_breadth,
        })

    # Label groups by actual shopping breadth, plus the product tendencies in this dataset.
    # Sorting the breadth instead of cluster number keeps these segment names interpretable.
    breadth_order = sorted(range(len(segment_info)),
                           key=lambda i: segment_info[i]['median_distinct_products'])
    names_by_breadth = [
        'Occasional mixed-gift shoppers',
        'Home decor regulars',
        'Lunch-bag and partyware shoppers',
        'Broad-catalog gift shoppers',
    ]
    for rank, idx in enumerate(breadth_order):
        segment_info[idx]['segment_name'] = names_by_breadth[rank]
    return pd.DataFrame(segment_info), [entry['segment_name'] for entry in segment_info]


def product_neighbours(A, V, descriptions, topn=10):
    """Examples of nearest products using cosine similarity between factor vectors."""
    rows = []
    popularity = np.asarray(A.sum(axis=0)).ravel()
    for source in np.argsort(-popularity)[:5]:
        scores = V @ V[int(source)]
        scores[int(source)] = -np.inf
        ranked = np.argsort(-scores)[:topn]
        rows.append({
            'source_product': descriptions.get(int(source), 'Unknown'),
            'suggested_products': ' | '.join(
                descriptions.get(int(j), 'Unknown') for j in ranked[:5]),
            'top_similarity': float(scores[ranked[0]]),
        })
    return pd.DataFrame(rows)


def main():
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    A = sparse.load_npz(DATA/'train_matrix.npz').tocsr()
    heldout = pd.read_csv(DATA/'test_holdout.csv')
    truth = heldout.groupby('cust_idx')['prod_idx'].agg(set).to_dict()
    eligible = np.array(sorted(truth), dtype=np.int32)
    n, m = A.shape
    print('Training matrix', A.shape, 'held-out customers',len(eligible),flush=True)
    if m <= max(K_OPTIONS):
        raise ValueError('Insufficient columns for k=200')
    best_sellers = np.argsort(-np.asarray(A.sum(axis=0)).ravel(), kind='stable')
    baseline, baseline_sec = measure(lambda: baseline_rank(A, eligible, best_sellers, TOP_N))[:2]
    baseline_rate, baseline_hits = hit_rate_at_10(baseline,eligible,truth)
    curve = []
    selected = None
    print('Fitting SVD for k:',K_OPTIONS,flush=True)
    for k in K_OPTIONS:
        def task():
            model = TruncatedSVD(n_components=k, n_iter=5, random_state=SEED)
            model.fit(A)
            return model
        model, fit_sec, memory_MB = measure(task)
        err = reconstruct_error(A,model,'SVD')
        info = {'method':'Truncated SVD', 'k':k, 'fit_seconds':fit_sec,
                'observed_peak_RSS_MB':memory_MB, 'relative_reconstruction_error':err,
                'variance_explained_fraction':float(model.explained_variance_ratio_.sum())}
        curve.append(info)
        print('  ',info,flush=True)
        if k == CHOSEN_K:
            selected = model
            svd_fit, svd_peak, svd_err = fit_sec, memory_MB, err
        else:
            del model
    pd.DataFrame(curve).to_csv(RESULTS/'svd_k_curve.csv',index=False)
    V_svd = make_embeddings(selected)
    prediction, score_sec = rank_customers(A,V_svd,eligible,TOP_N)
    svd_rate, svd_hits = hit_rate_at_10(prediction, eligible, truth)
    print('SVD hit@10',svd_rate,'baseline',baseline_rate,'scoring seconds',score_sec,flush=True)

    # Every product's top 10 neighbors are computed in blocks, representing nightly product-page refresh.
    start = time.perf_counter()
    product_top10 = np.empty((m,TOP_N),dtype=np.int32)
    for b in range(0,m,128):
        end = min(m,b+128)
        scores = V_svd[b:end] @ V_svd.T
        scores[np.arange(end-b),np.arange(b,end)] = -np.inf
        cand = np.argpartition(-scores,TOP_N-1,axis=1)[:,:TOP_N]
        vals = np.take_along_axis(scores,cand,axis=1)
        product_top10[b:end] = np.take_along_axis(cand,np.argsort(-vals,axis=1),axis=1)
    neighbors_sec = time.perf_counter()-start
    # IDs here are anonymous product position indices, not real customer identifiers.
    np.save(DATA/'product_neighbors_top10.npy',product_top10)

    # ML5 comparison: same TRAIN matrix, same k, same recommender scoring and holdout.
    print('Fitting sparse PCA with arpack...',flush=True)
    def fit_pca():
        model = PCA(n_components=CHOSEN_K, svd_solver='arpack',random_state=SEED)
        model.fit(A)
        return model
    pca, pca_fit, pca_peak = measure(fit_pca)
    pca_err = reconstruct_error(A,pca,'PCA')
    V_pca = make_embeddings(pca)
    pca_preds, pca_score_sec = rank_customers(A,V_pca,eligible,TOP_N)
    pca_rate, pca_hits = hit_rate_at_10(pca_preds,eligible,truth)
    svd_size = float((n+m)*CHOSEN_K*4/1e6)
    pca_size = float(((n+m)*CHOSEN_K+m)*4/1e6)  # includes centred-data mean vector
    comparison = pd.DataFrame([
        {'method':'Truncated SVD','k':CHOSEN_K,'fit_seconds':svd_fit,
         'observed_peak_RSS_MB':svd_peak,'stored_factors_MB_float32':svd_size,
         'variance_explained_fraction':float(selected.explained_variance_ratio_.sum()),
         'relative_reconstruction_error':svd_err, 'hit_rate_at_10':svd_rate,
         'hit_customers':svd_hits,'baseline_hit_rate_at_10':baseline_rate,
         'score_eligible_customers_seconds':score_sec,
         'component_1_meaning':'Raw buying activity / popularity (uncentred)'},
        {'method':'PCA','k':CHOSEN_K,'fit_seconds':pca_fit,
         'observed_peak_RSS_MB':pca_peak,'stored_factors_MB_float32':pca_size,
         'variance_explained_fraction':float(pca.explained_variance_ratio_.sum()),
         'relative_reconstruction_error':pca_err, 'hit_rate_at_10':pca_rate,
         'hit_customers':pca_hits,'baseline_hit_rate_at_10':baseline_rate,
         'score_eligible_customers_seconds':pca_score_sec,
         'component_1_meaning':'Variation relative to average purchase profile (centred)'}
    ])
    comparison.to_csv(RESULTS/'method_comparison.csv',index=False)
    metrics = {'k':CHOSEN_K, 'test_customers':len(eligible),'eligible_heldout_pairs':len(heldout),
               'baseline_hit_rate_at_10':baseline_rate,'svd_hit_rate_at_10':svd_rate,
               'pca_hit_rate_at_10':pca_rate,'svd_fit_seconds':svd_fit,
               'svd_scoring_seconds':score_sec,'product_neighbor_generation_seconds':neighbors_sec,
               'svd_total_nightly_refresh_seconds':svd_fit+neighbors_sec+score_sec,
               'baseline_scoring_seconds':baseline_sec,'svd_error':svd_err,
               'pca_error':pca_err,'dense_float64_MB':n*m*8/1e6,
               'stored_svd_factors_MB_float32':svd_size,
               'compression_vs_dense_float64':n*m*8/((n+m)*CHOSEN_K*4),
               'recommendation_definition':'Sum of cosine similarities to each previously bought product, exclude already-owned',
               'hit_metric':'Fraction of eligible test customers with at least one new future item in their top 10 suggestions',
               'timing_definition':'SVD fit + all product neighbour updates + held-out customer scoring; disk I/O and Excel parsing excluded'}

    descriptions = pd.read_csv(DATA/'product_descriptions.csv').set_index('prod_idx')['Description'].to_dict()
    product_neighbours(A,V_svd,descriptions).to_csv(RESULTS/'recommendation_examples.csv',index=False)

    # ML6: 2D projection and fixed four marketing segments.
    customer_factors = selected.transform(A)
    xy = customer_factors[:,:2]
    # Scale axes to avoid volume swallowing component 2 during K-means.
    xy_standard = StandardScaler().fit_transform(xy)
    clusters = KMeans(n_clusters=4,n_init=15,random_state=SEED).fit_predict(xy_standard)
    countries = pd.read_csv(DATA/'customer_countries.csv').sort_values('cust_idx')
    segment_table, names = label_segments(A,clusters,countries,descriptions)
    segment_table.to_csv(RESULTS/'segments.csv',index=False)
    fig,ax = plt.subplots(figsize=(11,6.5))
    colors = ['#15759a','#f09b54','#7b67b0','#4c9d79']
    for group in range(4):
        idx = clusters==group
        ax.scatter(xy_standard[idx,0],xy_standard[idx,1],s=7,alpha=.45,
                   color=colors[group],label=f'{group+1}. {names[group]} ({idx.sum():,})',rasterized=True)
    ax.set(title='Customer segments from purchase patterns (SVD, k=50)',
           xlabel='Standardized component 1',ylabel='Standardized component 2')
    ax.legend(loc='upper left',bbox_to_anchor=(1.02,1),frameon=False,fontsize=8)
    ax.grid(alpha=.12)
    fig.savefig(FIGURES/'customer_map.png',dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5,4.5))
    ks = [r['k'] for r in curve]
    errors = [r['relative_reconstruction_error'] for r in curve]
    ax.plot(ks,errors,marker='o',color='#007f9e',lw=2.5)
    ax.axvline(CHOSEN_K,ls='--',color='#e18146',label=f'Chosen k = {CHOSEN_K}')
    ax.set(xlabel='Components retained (k)',ylabel='Relative reconstruction error',
           title='Compression quality on training purchases')
    ax.legend(frameon=False)
    ax.grid(alpha=.15)
    fig.savefig(FIGURES/'svd_error_curve.png',dpi=180)
    plt.close(fig)

    fig,ax = plt.subplots(figsize=(7.5,4.6))
    methods = ['Best sellers','Truncated SVD','PCA']
    scores = [baseline_rate,svd_rate,pca_rate]
    bars = ax.bar(methods,[100*x for x in scores], color=['#8d9aa3','#007f9e','#7b67b0'],width=.65)
    for rect,value in zip(bars,scores):
        ax.text(rect.get_x()+rect.get_width()/2,100*value+.8,f'{value:.1%}',ha='center',fontsize=10)
    ax.set(ylabel='Held-out customer hit rate@10 (%)',title=f'Top-10 recommendations on {len(eligible):,} future-purchase customers',
           ylim=(0,min(100,max(scores)*110+5)))
    ax.grid(axis='y',alpha=.12)
    fig.savefig(FIGURES/'hit_rates.png',dpi=180)
    plt.close(fig)

    # ML7 optional sensitivity check: same training support, changed values only.
    train_pairs = pd.read_csv(DATA/'train_pairs.csv.gz')
    count_vals = train_pairs['quantity'].clip(upper=train_pairs['quantity'].quantile(.99)).to_numpy(dtype=np.float32)
    sensitivity = [{'purchase_value':'binary', 'hit_rate_at_10':svd_rate,'chosen_k':CHOSEN_K}]
    for value_name, values in [('count_winsor99', count_vals), ('log1p_quantity',np.log1p(count_vals))]:
        Aw = sparse.csr_matrix((values,(train_pairs.cust_idx,train_pairs.prod_idx)), shape=A.shape)
        normed = TruncatedSVD(n_components=CHOSEN_K,n_iter=5,random_state=SEED).fit(Aw)
        vs = make_embeddings(normed)
        # recommendation still aggregates unique purchased items; compare factor representation only
        rec,_ = rank_customers(A,vs,eligible,TOP_N)
        hit,_ = hit_rate_at_10(rec, eligible, truth)
        sensitivity.append({'purchase_value':value_name,'hit_rate_at_10':hit,'chosen_k':CHOSEN_K})
        print('Sensitivity',value_name,hit,flush=True)
    pd.DataFrame(sensitivity).to_csv(RESULTS/'sensitivity.csv',index=False)
    (RESULTS/'model_metrics.json').write_text(json.dumps(metrics,indent=2))
    print('FINISHED',json.dumps(metrics,indent=2),flush=True)


if __name__ == '__main__':
    main()
