
import argparse
import sys
import tqdm



def main():

    parser = argparse.ArgumentParser(description="Script with conditional parameters")
    
    parser.add_argument('--train_log_path', type=str,required=True)
    parser.add_argument('--check_point_strs', type=str)
    args = parser.parse_args()
    datas = [line.strip() for line in open(args.train_log_path,"r").readlines() if "eval_loss" in line]
    evals = []
    for eval in datas[:-1]:
        evals.append(float(eval.split(",")[0].split(": ")[1]))
    evals.append(float(datas[-1].split("=     ")[1]))
    check_points = [int(point.split("-")[-1]) for point in args.check_point_strs.split(",")]
    check_points = sorted(check_points)
    min_index = evals.index(min(evals[:len(check_points)]))
    best_id = check_points[min_index]
    
    #print(min_index)
    print(check_points[min_index])


if __name__ == "__main__":
    main()