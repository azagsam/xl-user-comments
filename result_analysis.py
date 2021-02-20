import pandas as pd
import numpy as np
import scipy.stats


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return np.round(m, 4), np.round(m-h, 4), np.round(m+h, 4)

data = ['cro-results-complete.csv', 'kristina.csv', 'articles_only.csv']
for exp in data:
    df = pd.read_csv(f'output/{exp}')
    df.drop(['example_num'], inplace=True, axis=1)

    # remove unwanted experiments
    df = df[df['sent_enc'] != 'None']
    df = df[df['method'] != 'lexrank']

    # # Sentence enoder
    # g = df.groupby(['sent_enc'])
    # g['ROUGE-L'].agg(['mean', 'std', 'min', 'max']).sort_values('mean', ascending=False)
    # g['ROUGE-2'].agg(['mean', 'std', 'min', 'max']).sort_values('mean', ascending=False)
    #
    # # Dim reduction
    # g = df.groupby(['dim_red'])
    # g['ROUGE-L'].agg(['mean', 'std', 'min', 'max']).sort_values('mean', ascending=False)
    # g['ROUGE-2'].agg(['mean', 'std', 'min', 'max']).sort_values('mean', ascending=False)
    #
    # # Method
    # g = df.groupby(['method'])
    # g['ROUGE-L'].agg(['mean', 'std', 'min', 'max']).sort_values('mean', ascending=False)
    # g['ROUGE-2'].agg(['mean', 'std', 'min', 'max']).sort_values('mean', ascending=False)

    # graph-based vs. clustering
    d = {'lexrank': 'graph-based',
         'textrank': 'graph-based',
         'gaussianmixture': 'clustering',
         'kmeans': 'clustering',
         'baseline-lead': 'baseline',
         'baseline-random': 'baseline'}

    method_type = [d[m] for m in df['method']]
    df['method_type'] = method_type

    g = df.groupby(['method_type'])
    g['ROUGE-L'].agg(['mean', 'std', 'min', 'max', mean_confidence_interval]).sort_values('mean', ascending=False)
    g['ROUGE-2'].agg(['mean', 'std', 'min', 'max', mean_confidence_interval]).sort_values('mean', ascending=False)

    # write to disk
    print(3 * '\n', exp)
    groups = ['sent_enc', 'dim_red', 'method_type']
    for group in groups:
        g = df.groupby([group])
        print(f'\n{group} - ROUGE-L')
        print(g['ROUGE-L'].agg(['mean', 'std', 'min', 'max', mean_confidence_interval, 'size']).sort_values('mean', ascending=False).round(4))
        # print(f'\n\n{group} - ROUGE-2')
        # print(g['ROUGE-2'].agg(['mean', 'std', 'min', 'max', mean_confidence_interval]).sort_values('mean', ascending=False))
