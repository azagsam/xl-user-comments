import os
from random import randrange

# data
root = '/home/ales/Documents/Extended/datasets/STYRIA/output/articles+comments-workshop-2500-samples/articles_only'

# lead baseline
tgt_folder = 'output/articles+comments-workshop-2500-samples/baseline-lead'
os.makedirs(tgt_folder, exist_ok=True)
for file in os.scandir(root):
    with open(file.path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    with open(os.path.join(tgt_folder, file.name), 'w') as out:
        # write first two sentences
        for s in range(2):
            out.write(lines[s])
            out.write('\n')


# random baseline
tgt_folder = 'output/articles+comments-workshop-2500-samples/baseline-random'
os.makedirs(tgt_folder, exist_ok=True)
for file in os.scandir(root):
    with open(file.path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    with open(os.path.join(tgt_folder, file.name), 'w') as out:
        # write random two sentences
        for _ in range(2):
            s = randrange(len(lines))
            out.write(lines[s])
            out.write('\n')

