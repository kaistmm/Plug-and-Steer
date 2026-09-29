import os
import glob
from argparse import ArgumentParser
from tqdm import tqdm
import pandas as pd


def get_vpath_from_audpath(audpath, df, lrs2_src_dir):
    short_split = audpath.split('/')[-3]
    if short_split == 'tr':
        split = 'train'
    elif short_split == 'cv':
        split = 'val'
    elif short_split == 'tt':
        split = 'test'
    else:
        raise ValueError

    audpath_splitted = os.path.basename(audpath.replace('.wav','')).split('_')
    s1_vname, s1_vnum, s1_snr = audpath_splitted[0:3]
    s2_vname, s2_vnum, s2_snr = audpath_splitted[3:6]
    
    
    result = df[
        (df.iloc[:, 0] == split) & 
        (df.iloc[:, 2] == s1_vname) & 
        (df.iloc[:, 3] == s1_vnum) &
        (df.iloc[:, 4] == s1_snr) &
        (df.iloc[:, 6] == s2_vname) & 
        (df.iloc[:, 7] == s2_vnum) &
        (df.iloc[:, 8] == s2_snr)
    ]
    if len(result)==0:
        import pdb; pdb.set_trace()
    result = result.iloc[0]
    
    s1_vpath = os.path.join(lrs2_src_dir, result[1], result[2], result[3]+'.mp4')
    s2_vpath = os.path.join(lrs2_src_dir, result[5], result[6], result[7]+'.mp4')
    
    if not os.path.exists(s1_vpath) or not os.path.exists(s2_vpath):
        import pdb; pdb.set_trace()
    
    return s1_vpath, s2_vpath


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--lrs2_2mix_dir', required=True, type=str,
                        help="Path to LRS2-2mix")
    parser.add_argument('--lrs2_src_dir', required=True, type=str,
                        help="Path to LRS2 source directory. It should contain videos.")
    args = parser.parse_args()

    lrs2_2mix_path = args.lrs2_2mix_dir
    df = pd.read_csv('./data/lrs2_2mix/mixture_data_list_2mix.csv', header=None, dtype=str)

    train_mix_scp = './data/lrs2_2mix/tr_mix.scp'
    train_s1_scp = './data/lrs2_2mix/tr_s1.scp'
    train_s2_scp = './data/lrs2_2mix/tr_s2.scp'

    cv_mix_scp = './data/lrs2_2mix/cv_mix.scp'
    cv_s1_scp = './data/lrs2_2mix/cv_s1.scp'
    cv_s2_scp = './data/lrs2_2mix/cv_s2.scp'

    test_mix_scp = './data/lrs2_2mix/tt_mix.scp'
    test_s1_scp = './data/lrs2_2mix/tt_s1.scp'
    test_s2_scp = './data/lrs2_2mix/tt_s2.scp'


    train_mix = f'{lrs2_2mix_path}/tr/mix'
    train_s1 = f'{lrs2_2mix_path}/tr/s1'
    train_s2 = f'{lrs2_2mix_path}/tr/s2'

    cv_mix = f'{lrs2_2mix_path}/cv/mix'
    cv_s1 = f'{lrs2_2mix_path}/cv/s1'
    cv_s2 = f'{lrs2_2mix_path}/cv/s2'

    test_mix = f'{lrs2_2mix_path}/tt/mix'
    test_s1 = f'{lrs2_2mix_path}/tt/s1'
    test_s2 = f'{lrs2_2mix_path}/tt/s2'
    
    
    print("Creating scp files...")
    print("1/9")
    tr_vids = {}
    tr_mix = open(train_mix_scp,'w')
    for root, dirs, files in os.walk(train_mix):
        files.sort()
        for file in files:
            tr_mix.write(file+" "+root+'/'+file)
            tr_mix.write('\n')
            tr_vids[file] = get_vpath_from_audpath(root+'/'+file, df, args.lrs2_src_dir)

    
    print("2/9")
    tr_s1 = open(train_s1_scp,'w')
    for root, dirs, files in os.walk(train_s1):
        files.sort()
        for file in files:
            tr_s1.write(file+" "+root+'/'+file)
            vpath = tr_vids[file][0]
            tr_s1.write(" "+vpath)
            tr_s1.write('\n')


    print("3/9")
    tr_s2 = open(train_s2_scp,'w')
    for root, dirs, files in os.walk(train_s2):
        files.sort()
        for file in files:
            tr_s2.write(file+" "+root+'/'+file)
            vpath = tr_vids[file][1]
            tr_s2.write(" "+vpath)
            tr_s2.write('\n')


    print("4/9")
    tt_vids = {}
    tt_mix = open(test_mix_scp,'w')
    for root, dirs, files in os.walk(test_mix):
        files.sort()
        for file in files:
            tt_mix.write(file+" "+root+'/'+file)
            tt_mix.write('\n')
            tt_vids[file] = get_vpath_from_audpath(root+'/'+file, df, args.lrs2_src_dir)



    print("5/9")
    tt_s1 = open(test_s1_scp,'w')
    for root, dirs, files in os.walk(test_s1):
        files.sort()
        for file in files:
            tt_s1.write(file+" "+root+'/'+file)
            vpath = tt_vids[file][0]
            tt_s1.write(" "+vpath)
            tt_s1.write('\n')


    print("6/9")
    tt_s2 = open(test_s2_scp,'w')
    for root, dirs, files in os.walk(test_s2):
        files.sort()
        for file in files:
            tt_s2.write(file+" "+root+'/'+file)
            vpath = tt_vids[file][1]
            tt_s2.write(" "+vpath)
            tt_s2.write('\n')




    print("7/9")
    cv_vids = {}
    cv_mix_file = open(cv_mix_scp,'w')
    for root, dirs, files in os.walk(cv_mix):
        files.sort()
        for file in files:
            cv_mix_file.write(file+" "+root+'/'+file)
            cv_mix_file.write('\n')
            cv_vids[file] = get_vpath_from_audpath(root+'/'+file, df, args.lrs2_src_dir)



    print("8/9")
    cv_s1_file = open(cv_s1_scp,'w')
    for root, dirs, files in os.walk(cv_s1):
        files.sort()
        for file in files:
            cv_s1_file.write(file+" "+root+'/'+file)
            vpath = cv_vids[file][0]
            cv_s1_file.write(" "+vpath)
            cv_s1_file.write('\n')


    print("9/9")
    cv_s2_file = open(cv_s2_scp,'w')
    for root, dirs, files in os.walk(cv_s2):
        files.sort()
        for file in files:
            cv_s2_file.write(file+" "+root+'/'+file)
            vpath = cv_vids[file][1]
            cv_s2_file.write(" "+vpath)
            cv_s2_file.write('\n')
