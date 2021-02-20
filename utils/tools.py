import os

# # Number of files
# root = 'output/Styria'
# for folder in os.listdir(root):
#     file_num = len(list(os.listdir(os.path.join(root, folder))))
#     print(folder, file_num)


# missing combinations
root = '/home/ales/Documents/RSDO/graph-based/output/Styria'
files = {file for file in os.listdir(root) if '_' in file}
combinations = set()
for enc in ['SentenceBERT', 'CMLM', 'LaBSE']:
    for red in ['pca', 'umap', 'None']:
        for method in ['textrank', 'lexrank', 'gaussianmixture', 'kmeans']:
            combinations.add('_'.join([enc, red, method]))
print(combinations - files)

