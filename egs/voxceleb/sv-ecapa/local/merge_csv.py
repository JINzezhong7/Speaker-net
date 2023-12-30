import pandas as pd
import argparse

def merge_csv(path1,path2,new_path):
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)
    merged_df = pd.concat([df1, df2], ignore_index=True)

    merged_df.to_csv(new_path, index=False)

def main(args):
    merge_csv(args.vox1_path,args.vox2_path,args.new_path)


if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--vox1_path',
                        type=str,
                        default="/home/jinzezhong/data/vox1/dev/train.csv",
                        help="the csv path of vox1")
    parser.add_argument('--vox2_path',
                        type=str,
                        default="/home/jinzezhong/data/vox2_dev/train.csv",
                        help="the csv path of vox2")
    parser.add_argument('--new_path',
                        type=str,
                        default="/home/jinzezhong/data/vox1_2_concat/train.csv",
                        help="the csv path of vox1 and vox2 condat")

    args = parser.parse_args()
    print("begin merge")
    main(args)
    print("Success!!")
