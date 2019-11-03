# libraries
import csv
import numpy as np
import pandas as pd
import sys
import getopt
import os


# gradient calculation
def gradient(instanceframe, Yattr, weights):
    # error = np.around(np.subtract(np.array(Yattr), np.dot(instanceframe, weights)), 4)
    error = np.subtract(np.array(Yattr), np.dot(instanceframe, weights))
    gradient_matrix = np.dot(instanceframe.transpose(), error)
    return gradient_matrix


# error calculation
def squarederror(instanceframe, Yattr, weights):
    error = np.subtract(np.array(Yattr), np.dot(instanceframe, weights))
    error = np.power(error, 2)
    squared_error_sum = np.sum(error, axis=0)
    return squared_error_sum


# function t0 update weights
def weights_update(weights, gradient_matrix, learning_rate):
    weights = np.add(weights, learning_rate * gradient_matrix)
    return weights


def main(argv):
    # variables to store the argument values from command line
    input_path_file = ""
    learning_rate = 0.0
    threshold = 0.0

    # Read values from command line
    try:
        opts, args = getopt.getopt(argv, "", ["data=", "learningRate=", "threshold="])
        print(opts)
    except getopt.GetoptError:
        print('linearregr.py --data <inputfile> --learningRate <e.g.0.0001> --threshold <e.g.0.0001>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == "--data":
            input_path_file = arg
        elif opt == "--learningRate":
            learning_rate = np.float(arg)
        elif opt == "--threshold":
            threshold = np.float(arg)

        print(input_path_file, learning_rate, threshold)

    # Read the data-set and  split into instance frame and target frame
    UnnamedData = pd.read_csv(input_path_file, header=None)
    instanceframe = UnnamedData.iloc[:, :-1]
    Yattr = UnnamedData.iloc[:, -1]
    instanceframe.insert(loc=0, column=0, value=1, allow_duplicates=True)
    instanceframe.columns = np.arange(1, len(instanceframe.columns) + 1)
    instanceframe.index = np.arange(1, len(instanceframe.index) + 1)
    Yattr = Yattr.to_frame()

    # initialisation of weight matrix with zeros
    weights = np.zeros(len(instanceframe.columns))
    weights = pd.DataFrame(weights)
    iteration = -1

    # output file path w.r.t both relative and absolute path
    output_file_split_list = input_path_file.split("/")
    if len(output_file_split_list) == 1:
        output_file_name = os.getcwd() + "/" + "solution_for_" + output_file_split_list[-1]
    else:
        output_file_name = "/".join(output_file_split_list[:-1]) + "/" + "solution_generated_for_" + output_file_split_list[-1]

    # Writing Csv Files
    with open(output_file_name, 'w') as f1:
        writer = csv.writer(f1, delimiter=',')
        while True:
            iteration += 1

            # To calculate the initial error and print the first Zero initialised weight values
            if iteration == 0:
                error = squarederror(instanceframe, Yattr, weights)
                previous_error = error

                # Added respective error and iteration values to weight array and passed them to the Csv writer object
                weights_flattened = np.round(np.append(np.array(weights), error).flatten(), 4).tolist()
                weights_flattened.insert(0, iteration)
                writer.writerow(weights_flattened)

                print(iteration, weights, error)
                gradient_matrix = gradient(instanceframe, Yattr, weights)
                weights = weights_update(weights, gradient_matrix, learning_rate)
                present_error = squarederror(instanceframe, Yattr, weights)

            # iterative condition to calculate the error and update the weight values
            if iteration >= 1 and abs(present_error - previous_error) > threshold:
                weights_flattened = np.round(np.append(np.array(weights), present_error).flatten(), 4).tolist()
                weights_flattened.insert(0, iteration)
                writer.writerow(weights_flattened)

                print(iteration, weights, present_error)
                previous_error = present_error
                gradient_matrix = gradient(instanceframe, Yattr, weights)
                weights = weights_update(weights, gradient_matrix, learning_rate)
                present_error = squarederror(instanceframe, Yattr, weights)

            # Conditional statement to stop the iteration
            if iteration >= 1 and abs(present_error - previous_error) < threshold:
                iteration = iteration + 1
                weights_flattened = np.round(np.append(np.array(weights), present_error).flatten(), 4).tolist()
                weights_flattened.insert(0, iteration)
                writer.writerow(weights_flattened)
                print(iteration, weights, present_error)

                break


if __name__ == "__main__":
    argv = sys.argv[1:]
    # print(argv)
    main(argv)
