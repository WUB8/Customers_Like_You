from __future__ import annotations
import json
import zipfile
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse

BASE = Path(__file__).resolve().parent
RAW = BASE / 'data/raw/online_retail_II.xlsx'
OUT = BASE / 'data/processed'
RESULTS = BASE / 'results'
SEED = 42
MIN_PRODUCT_CUSTOMERS = 10
NON_PRODUCT_CODES = {'POST', 'POSTAGE', 'M', 'D', 'DOT', 'BANK CHARGES', 'AMAZONFEE', 'S', 'C2', 'B',
                     'CRUK', 'PADS', 'TEST001', 'TEST002', 'ADJUST', 'ADJUST2', 'DCGSSBOY', 'DCGSSGIRL',
                     'DCGS0003', 'DCGS0004', 'GIFT VOUCHER', 'GIFT VOUCHER S', 'GIFT VOUCHER M',
                     'GIFT VOUCHER L', 'GIFT VOUCHER XL'}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    if not RAW.exists():
        # Convenience: the original downloaded UCI ZIP can be placed here or beside this script.
        candidates = [BASE / 'data/raw/online+retail+ii.zip',
                      BASE / 'data/raw/online_retail_ii.zip',
                      BASE / 'online+retail+ii.zip']
        archive = next((p for p in candidates if p.exists()), None)
        if archive is None:
            raise FileNotFoundError(
                'Missing dataset. Put online_retail_II.xlsx in data/raw/, or '
                'put the original online+retail+ii.zip in data/raw/.')
        RAW.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zf:
            with zf.open('online_retail_II.xlsx') as source, RAW.open('wb') as target:
                shutil.copyfileobj(source, target)
        print(f'Extracted {RAW.name} from {archive.name}', flush=True)
    print('Loading both workbook sheets...', flush=True)
    raw_sheets = pd.read_excel(RAW, sheet_name=None)
    profile = []
    print('Sheets:', [(k, len(v)) for k, v in raw_sheets.items()], flush=True)
    for sheet_name, frame in raw_sheets.items():
        for column in frame.columns:
            profile.append({'sheet': sheet_name, 'column': column, 'dtype': str(frame[column].dtype),
                            'row_count': len(frame), 'missing_count': int(frame[column].isna().sum()),
                            'missing_pct': float(frame[column].isna().mean() * 100)})
    pd.DataFrame(profile).to_csv(RESULTS / 'data_profile.csv', index=False)
    raw = pd.concat(raw_sheets.values(), ignore_index=True)
    raw.rename(columns={'InvoiceNo': 'Invoice', 'UnitPrice': 'Price', 'CustomerID': 'Customer ID'}, inplace=True)
    required = ['Invoice', 'StockCode', 'Description', 'Quantity', 'InvoiceDate', 'Price', 'Customer ID', 'Country']
    missing = set(required) - set(raw.columns)
    if missing:
        raise ValueError(f'Missing columns: {missing}. Got: {list(raw.columns)}')
    raw = raw[required].copy()
    df = raw
    log = []

    def filter_rows(name, mask, reason):
        nonlocal df
        before = len(df)
        df = df.loc[mask].copy()
        log.append({'step': name, 'rows_before': before, 'rows_removed': before - len(df),
                    'rows_remaining': len(df), 'reason': reason})
        print(f'{name}: removed {before-len(df):,}; remaining {len(df):,}', flush=True)

    filter_rows('1 cancel invoices', ~df['Invoice'].astype('string').str.upper().str.startswith('C').fillna(False),
                'C-prefixed invoice represents a cancellation, not a positive sale')
    filter_rows('2 non-positive quantity', df['Quantity'].gt(0).fillna(False),
                'Negative/zero quantities are returns or adjustments, not purchases')
    filter_rows('3 non-positive price', df['Price'].gt(0).fillna(False),
                'Zero/negative prices cannot represent paid purchases')
    codes = df['StockCode'].astype('string').str.strip().str.upper()
    valid = codes.str.fullmatch(r'\d{5,6}[A-Z]?').fillna(False) & ~codes.isin(NON_PRODUCT_CODES)
    filter_rows('4 non-product codes', valid,
                'Keep 5-6 digit stock numbers with optional variant letter; reject service/fee codes')
    filter_rows('5 missing customer ID', df['Customer ID'].notna(),
                'Unidentified purchases cannot be linked to a customer; never infer customer IDs')
    filter_rows('6 invalid invoice date', pd.to_datetime(df['InvoiceDate'], errors='coerce').notna(),
                'A valid time is required for a leak-free chronological split')
    pd.DataFrame(log).to_csv(RESULTS / 'cleaning_log.csv', index=False)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['StockCode'] = df['StockCode'].astype(str).str.upper().str.strip()
    df['Customer ID'] = df['Customer ID'].astype('int64')
    df['Description'] = df['Description'].fillna('Unknown item').astype(str).str.strip()
    df['Country'] = df['Country'].fillna('Unknown').astype(str).str.strip()

    # Last two months, measured back from the last observed invoice timestamp.
    cutoff = df['InvoiceDate'].max() - pd.DateOffset(months=2)
    train = df.loc[df['InvoiceDate'] < cutoff].copy()
    test = df.loc[df['InvoiceDate'] >= cutoff].copy()
    print('Date cut:', cutoff, 'train rows:', len(train), 'test rows:', len(test), flush=True)
    # Product popularity filter derived from TRAIN data only (no test leakage).
    product_customers = train.groupby('StockCode')['Customer ID'].nunique()
    valid_products = product_customers[product_customers >= MIN_PRODUCT_CUSTOMERS].index
    train = train.loc[train['StockCode'].isin(valid_products)].copy()
    cust_ids = np.sort(train['Customer ID'].unique())
    prod_ids = np.sort(train['StockCode'].unique())
    cust_to_idx = pd.Series(np.arange(len(cust_ids), dtype=np.int32), index=cust_ids)
    prod_to_idx = pd.Series(np.arange(len(prod_ids), dtype=np.int32), index=prod_ids)
    train['cust_idx'] = train['Customer ID'].map(cust_to_idx).astype(np.int32)
    train['prod_idx'] = train['StockCode'].map(prod_to_idx).astype(np.int32)
    # Keep purchase signal binary: repeat/reorder volume should not drown out affinities.
    train_pairs = train.groupby(['cust_idx', 'prod_idx'], sort=False, as_index=False).agg(quantity=('Quantity', 'sum'))
    A = sparse.csr_matrix((np.ones(len(train_pairs), dtype=np.float32),
                           (train_pairs['cust_idx'].values, train_pairs['prod_idx'].values)),
                          shape=(len(cust_ids), len(prod_ids)), dtype=np.float32)
    A.sum_duplicates()
    A.data[:] = 1.0
    sparse.save_npz(OUT / 'train_matrix.npz', A)
    # Identifiers live only in ignored internal data/processed/, not in figures/memo/slides.
    pd.DataFrame({'index': np.arange(len(cust_ids)), 'Customer ID': cust_ids}).to_csv(OUT/'customer_index.csv', index=False)
    pd.DataFrame({'index': np.arange(len(prod_ids)), 'StockCode': prod_ids}).to_csv(OUT/'product_index.csv', index=False)
    train_pairs.to_csv(OUT/'train_pairs.csv.gz', index=False, compression='gzip')

    # Eligible test products must be unseen by this customer and known to the TRAIN vocabulary.
    test = test.loc[test['Customer ID'].isin(cust_ids) & test['StockCode'].isin(prod_ids)].copy()
    test['cust_idx'] = test['Customer ID'].map(cust_to_idx).astype(np.int32)
    test['prod_idx'] = test['StockCode'].map(prod_to_idx).astype(np.int32)
    holdout = test[['cust_idx','prod_idx']].drop_duplicates()
    already = pd.MultiIndex.from_frame(train_pairs[['cust_idx', 'prod_idx']])
    unseen = ~pd.MultiIndex.from_frame(holdout).isin(already)
    holdout = holdout.loc[unseen].sort_values(['cust_idx','prod_idx'])
    holdout.to_csv(OUT/'test_holdout.csv', index=False)
    # Descriptions are chosen from training only, after any product filtering.
    descriptions = (train.groupby(['prod_idx','Description']).size().rename('n').reset_index()
                    .sort_values(['prod_idx','n','Description'], ascending=[True,False,True])
                    .drop_duplicates('prod_idx').sort_values('prod_idx'))
    descriptions[['prod_idx','Description']].to_csv(OUT/'product_descriptions.csv', index=False)
    countries = (train.groupby(['cust_idx','Country']).size().rename('n').reset_index()
                 .sort_values(['cust_idx','n','Country'],ascending=[True,False,True])
                 .drop_duplicates('cust_idx').sort_values('cust_idx'))
    countries[['cust_idx','Country']].to_csv(OUT/'customer_countries.csv', index=False)
    # Never write real customer identifiers into any externally shared figure or summary.
    Abytes = int(A.data.nbytes + A.indices.nbytes + A.indptr.nbytes)
    dense_bytes = int(A.shape[0] * A.shape[1] * 8)
    stats = {
        'source': 'UCI Online Retail II (Chen, 2012), CC BY 4.0',
        'raw_rows': len(raw), 'sheet_rows': {k:len(v) for k,v in raw_sheets.items()},
        'cleaned_rows':len(df), 'time_cutoff_exclusive_train':str(cutoff),
        'training_rows_before_product_filter': int((df['InvoiceDate'] < cutoff).sum()),
        'training_rows_after_product_filter':len(train),
        'test_rows_before_eligibility_filter':int((df['InvoiceDate'] >= cutoff).sum()),
        'test_eligible_customers':int(holdout['cust_idx'].nunique()),
        'test_heldout_unique_customer_product_pairs':len(holdout),
        'matrix_shape':[int(x) for x in A.shape], 'matrix_nonzeros':int(A.nnz),
        'matrix_sparsity_percent':float(100*(1-A.nnz/(A.shape[0]*A.shape[1]))),
        'dense_float64_MB':dense_bytes/1e6, 'csr_float32_MB':Abytes/1e6,
        'matrix_value':'binary (1 if bought at least once during TRAIN, else 0)',
        'minimum_unique_train_customers_per_product':MIN_PRODUCT_CUSTOMERS,
        'customer_id_handling':'Internal index map saved under gitignored data/processed/ only',
        'random_seed':SEED,
        'cleaning_steps':log
    }
    (RESULTS/'matrix_report.json').write_text(json.dumps(stats, indent=2))
    print('Matrix:', A.shape, 'nnz', A.nnz, 'sparsity %',round(stats['matrix_sparsity_percent'],2), flush=True)
    print('Holdout:', len(holdout), 'unseen customer-product pairs across',holdout['cust_idx'].nunique(),'customers',flush=True)


if __name__ == '__main__':
    main()
