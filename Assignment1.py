import csv
import numpy as np
import pandas as pd
import sys
import getopt


def gradient(instanceframe, Yattr, weights):
    error = np.around(np.subtract(np.array(Yattr), np.dot(instanceframe, weights)), 4)
    gradient_matrix = np.dot(instanceframe.transpose(), error)
    return error, gradient_matrix


def squarederror(error):
    error = np.power(error, 2)
    squared_error_sum = np.sum(error, axis=0)
    return squared_error_sum


def weights_update(weights, gradient_matrix, learning_rate):
    weights = np.around(np.add(weights, learning_rate * gradient_matrix), 4)
    return weights


def main(argv):
    input_path_file = ""
    learning_rate = 0.0
    threshold = 0.0

    try:
        opts, args = getopt.getopt(argv,"", ["data=", "learningRate=", "threshold="])
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

        print(input_path_file,learning_rate, threshold)

    UnnamedData = pd.read_csv(input_path_file, header=None)
    instanceframe = UnnamedData.iloc[:, :-1]
    Yattr = UnnamedData.iloc[:, -1]
    # pd.DataFrame(Yattr)
    instanceframe.insert(loc=0, column=0, value=1, allow_duplicates=True)

    instanceframe.columns = np.arange(1, len(instanceframe.columns) + 1)
    instanceframe.index = np.arange(1, len(instanceframe.index) + 1)
    Yattr = Yattr.to_frame()

    weights = np.zeros(len(instanceframe.columns))
    weights = pd.DataFrame(weights)
    iteration = 0
    previous_error = 0
    with open('assignment.csv', 'w') as f1:
        writer = csv.writer(f1, delimiter=',')
        print(weights)
        writer.writerow(np.array(weights))

        while True:
            iteration += 1
            if iteration == 1:
                error, gradient_matrix = gradient(instanceframe, Yattr, weights)
                weights = weights_update(weights, gradient_matrix, learning_rate)
                writer.writerow(np.array(weights))

                print(weights)
            if iteration == 2:
                previous_error = np.copy(error)
                error, gradient_matrix = gradient(instanceframe, Yattr, weights)
                weights = weights_update(weights, gradient_matrix, learning_rate)
                writer.writerow(np.array(weights))
                print(weights)
            if iteration > 2 and np.sum(error, axis=0) - np.sum(previous_error, axis=0) > threshold:
                previous_error = np.copy(error)
                error, gradient_matrix = gradient(instanceframe, Yattr, weights)
                weights = weights_update(weights, gradient_matrix, learning_rate)
                writer.writerow(np.array(weights))
                print(weights)
            if iteration > 2 and (np.sum(error) - np.sum(previous_error) < threshold):
                break

            print(iteration)


if __name__ == "__main__":
    argv = sys.argv[1:]
    print(argv)
    main(argv)

# previous_error = 0
# error = 0
