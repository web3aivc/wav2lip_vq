import os
import argparse

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", help="Root folder of the dataset",
                    default='data/min_pre/')
args = parser.parse_args()

root_dir = args.root_dir

lines_train = []
lines_val = []
dir_list = os.listdir(root_dir)
for i, dir in tqdm(enumerate(dir_list), total=len(dir_list)):
    if dir.startswith('.'):
        continue

    sub_dir_list = os.listdir(os.path.join(root_dir, dir))

    for sub_dir in sub_dir_list:
        if sub_dir.startswith('.'):
            continue

        line = dir + '/' + sub_dir
        wav_path = os.path.join(root_dir, line, 'audio.wav')
        if not os.path.exists(wav_path):
            continue

        if i % 10 == 0:
            lines_val.append(line)
        else:
            lines_train.append(line)

print(len(lines_train))
print(len(lines_val))

with open('filelists/train.txt', 'w') as f:
    for line in lines_train:
        f.writelines(line + '\n')

with open('filelists/val.txt', 'w') as f:
    for line in lines_val:
        f.writelines(line + '\n')
