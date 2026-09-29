import os
import glob
from argparse import ArgumentParser
from tqdm import tqdm

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path','-dp', required=True, type=str,
                        help="Path to LibriMix")
    args = parser.parse_args()

    librimix_path = args.data_path

    train_mix_scp = 'data/librimix/tr_mix.scp'
    train_enh_scp = 'data/librimix/tr_enh.scp'
    train_s1_scp = 'data/librimix/tr_s1.scp'
    train_s2_scp = 'data/librimix/tr_s2.scp'
    train_noise_scp = 'data/librimix/tr_noise.scp'

    cv_mix_scp = 'data/librimix/cv_mix.scp'
    cv_enh_scp = 'data/librimix/cv_enh.scp'
    cv_s1_scp = 'data/librimix/cv_s1.scp'
    cv_s2_scp = 'data/librimix/cv_s2.scp'
    cv_noise_scp = 'data/librimix/cv_noise.scp'

    test_mix_scp = 'data/librimix/tt_mix.scp'
    test_enh_scp = 'data/librimix/tt_enh.scp'
    test_s1_scp = 'data/librimix/tt_s1.scp'
    test_s2_scp = 'data/librimix/tt_s2.scp'
    test_noise_scp = 'data/librimix/tt_noise.scp'


    base_path = os.path.abspath(librimix_path)

    train_mix = f'{librimix_path}/Libri2Mix/wav16k/min/train-100/mix_both'
    train_enh = f'{librimix_path}/Libri2Mix/wav16k/min/train-100/mix_single'
    train_s1 = f'{librimix_path}/Libri2Mix/wav16k/min/train-100/s1'
    train_s2 = f'{librimix_path}/Libri2Mix/wav16k/min/train-100/s2/'
    train_noise = f'{librimix_path}/Libri2Mix/wav16k/min/train-100/noise'

    cv_mix = f'{librimix_path}/Libri2Mix/wav16k/min/dev/mix_both'
    cv_enh = f'{librimix_path}/Libri2Mix/wav16k/min/dev/mix_single'
    cv_s1 = f'{librimix_path}/Libri2Mix/wav16k/min/dev/s1'
    cv_s2 = f'{librimix_path}/Libri2Mix/wav16k/min/dev/s2/'
    cv_noise = f'{librimix_path}/Libri2Mix/wav16k/min/dev/noise'

    test_mix = f'{librimix_path}/Libri2Mix/wav16k/min/test/mix_both'
    test_enh = f'{librimix_path}/Libri2Mix/wav16k/min/test/mix_single'
    test_s1 = f'{librimix_path}/Libri2Mix/wav16k/min/test/s1'
    test_s2 = f'{librimix_path}/Libri2Mix/wav16k/min/test/s2/'
    test_noise = f'{librimix_path}/Libri2Mix/wav16k/min/test/noise'
    
    
    print("Creating scp files...")
    print("1/15")
    tr_mix = open(train_mix_scp,'w')
    for root, dirs, files in tqdm(os.walk(train_mix)):
        files.sort()
        for file in files:
            tr_mix.write(file+" "+root+'/'+file)
            tr_mix.write('\n')

    print("2/15")
    tr_enh = open(train_enh_scp,'w')
    for root, dirs, files in os.walk(train_enh):
        files.sort()
        for file in files:
            tr_enh.write(file+" "+root+'/'+file)
            tr_enh.write('\n')
    
    print("3/15")
    tr_s1 = open(train_s1_scp,'w')
    for root, dirs, files in os.walk(train_s1):
        files.sort()
        for file in files:
            tr_s1.write(file+" "+root+'/'+file)
            tr_s1.write('\n')


    print("4/15")
    tr_s2 = open(train_s2_scp,'w')
    for root, dirs, files in os.walk(train_s2):
        files.sort()
        for file in files:
            tr_s2.write(file+" "+root+'/'+file)
            tr_s2.write('\n')


    print("5/15")
    tr_noise = open(train_noise_scp,'w')
    for root, dirs, files in os.walk(train_noise):
        files.sort()
        for file in files:
            tr_noise.write(file+" "+root+'/'+file)
            tr_noise.write('\n')

    print("6/15")
    tt_mix = open(test_mix_scp,'w')
    for root, dirs, files in os.walk(test_mix):
        files.sort()
        for file in files:
            tt_mix.write(file+" "+root+'/'+file)
            tt_mix.write('\n')

    print("7/15")
    tt_enh = open(test_enh_scp,'w')
    for root, dirs, files in os.walk(test_enh):
        files.sort()
        for file in files:
            tt_enh.write(file+" "+root+'/'+file)
            tt_enh.write('\n')


    print("8/15")
    tt_s1 = open(test_s1_scp,'w')
    for root, dirs, files in os.walk(test_s1):
        files.sort()
        for file in files:
            tt_s1.write(file+" "+root+'/'+file)
            tt_s1.write('\n')


    print("9/15")
    tt_s2 = open(test_s2_scp,'w')
    for root, dirs, files in os.walk(test_s2):
        files.sort()
        for file in files:
            tt_s2.write(file+" "+root+'/'+file)
            tt_s2.write('\n')

    print("10/15")
    tt_noise = open(test_noise_scp,'w')
    for root, dirs, files in os.walk(test_noise):
        files.sort()
        for file in files:
            tt_noise.write(file+" "+root+'/'+file)
            tt_noise.write('\n')



    print("11/15")
    cv_mix_file = open(cv_mix_scp,'w')
    for root, dirs, files in os.walk(cv_mix):
        files.sort()
        for file in files:
            cv_mix_file.write(file+" "+root+'/'+file)
            cv_mix_file.write('\n')

    print("12/15")
    cv_enh_file = open(cv_enh_scp,'w')
    for root, dirs, files in os.walk(cv_enh):
        files.sort()
        for file in files:
            cv_enh_file.write(file+" "+root+'/'+file)
            cv_enh_file.write('\n')

    print("13/15")
    cv_s1_file = open(cv_s1_scp,'w')
    for root, dirs, files in os.walk(cv_s1):
        files.sort()
        for file in files:
            cv_s1_file.write(file+" "+root+'/'+file)
            cv_s1_file.write('\n')


    print("14/15")
    cv_s2_file = open(cv_s2_scp,'w')
    for root, dirs, files in os.walk(cv_s2):
        files.sort()
        for file in files:
            cv_s2_file.write(file+" "+root+'/'+file)
            cv_s2_file.write('\n')


    print("15/15")
    cv_noise_file = open(cv_noise_scp,'w')
    for root, dirs, files in os.walk(cv_noise):
        files.sort()
        for file in files:
            cv_noise_file.write(file+" "+root+'/'+file)
            cv_noise_file.write('\n')
