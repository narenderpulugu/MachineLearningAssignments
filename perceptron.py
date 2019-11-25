import os
import pandas as pd
import numpy as np
import getopt
import sys
import csv

def Error_calculation_update(target,output):
    Error = target - output
    return Error

def weights_update(Data_replaced_Target, Instance_Set_Columns,learning_rate, weight_list):
    data = Data_replaced_Target[Data_replaced_Target["Error"] != 0]
#    data["Error"] = data["Error"].abs()
    error = data["Error"].abs().sum()
    gradient = np.dot(np.transpose(data["Error"]),data[np.array(Instance_Set_Columns).squeeze()])
    weight_list_ = weight_list + [learning_rate*gradient]
    return weight_list_, error


def output_calc(Instance_frame, weight_list):
    output = np.dot(Instance_frame, weight_list)
    output_ = pd.DataFrame(output.squeeze())
    _output_ = output_.apply(lambda x: [0 if y <= 0.0 else 1 for y in x])
    return _output_


def main(argv):
    # variables to store the argument values from command line
    input_path_file = ""
    output_file_name = ""

    # Read values from command line
    try:
        opts, args = getopt.getopt(argv, "", ["data=", "output="])
    except getopt.GetoptError:
        print("perceptron.py --data <inputfile> --output <output file> ")
        sys.exit(2)
    for opt, arg in opts:
        if opt == "--data":
            input_path_file = arg
        elif opt == "--output":
            output_file_name = arg + ".tsv"

    with open(output_file_name, 'wt') as out_file:
        writer = csv.writer(out_file, delimiter='\t')
        Data = pd.read_csv(input_path_file, delim_whitespace=True, skipinitialspace=True, header = None)
        Instance_Set = Data[Data.columns[1:]]
        Instance_Set.insert(loc=0, column=0,value = pd.DataFrame(np.ones(len(Instance_Set.index))))
        weight_list = pd.DataFrame(np.zeros(len(Instance_Set.columns)))
        Data_replaced = Data.replace({0: {"A": 1, "B": 0}})
        Data_replaced_Target = Data_replaced.rename(columns ={0: "Target"})
        Data_replaced_Target.insert(loc=0, column=0,value = pd.DataFrame(np.ones(len(Instance_Set.index))))
        learning_rate = 1
        Instance_Set_Columns = Instance_Set.columns

        for learning_rate_ in {"constant", "anealing"}:
            row_array = []
            for i in range(0,101):
                if learning_rate_ == "anealing":
                    learning_rate = 0.5
                Data_replaced_Target_updated = Data_replaced_Target
                Data_replaced_Target_updated["output"] = output_calc(Instance_Set,weight_list)
                #Data_replaced_Target["Error"] = Data_replaced_Target["Target"] - Data_replaced_Target["output"]
                Data_replaced_Target_updated["Error"] = Error_calculation_update(Data_replaced_Target_updated["Target"],Data_replaced_Target_updated["output"])

                #data = Data_replaced_Target[Data_replaced_Target["Error"] == 1]

                weight_list, error = weights_update(Data_replaced_Target_updated, Instance_Set_Columns,learning_rate, weight_list)
                print(error)
                row_array.append(error)

            writer.writerow(row_array)


if __name__ == "__main__":
    argv = sys.argv[1:]
    main(argv)



