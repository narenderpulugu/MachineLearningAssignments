import csv
import numpy as np
import pandas as pd
import sys
import getopt



def gradient(instanceframe, Yattr, weights):
    #error = np.around(np.subtract(np.array(Yattr), np.dot(instanceframe, weights)), 4)
    error = np.subtract(np.array(Yattr), np.dot(instanceframe, weights))
    gradient_matrix = np.dot(instanceframe.transpose(), error)
    return gradient_matrix


def squarederror(instanceframe, Yattr, weights):
    error = np.subtract(np.array(Yattr), np.dot(instanceframe, weights))
    error = np.power(error, 2)
    squared_error_sum = np.sum(error, axis=0)
    return squared_error_sum


def weights_update(weights, gradient_matrix, learning_rate):
    weights = np.add(weights, learning_rate * gradient_matrix)
    return weights


def main():
    input_path_file = "/home/narender/Desktop/MachineLearning/random.csv"
    learning_rate = 0.0001
    threshold = 0.0001

    # input_path_file = ""
    # learning_rate = 0.0
    # threshold = 0.0

    # try:
    #     opts, args = getopt.getopt(argv,"", ["data=", "learningRate=", "threshold="])
    #     print(opts)
    # except getopt.GetoptError:
    #     print('linearregr.py --data <inputfile> --learningRate <e.g.0.0001> --threshold <e.g.0.0001>')
    #     sys.exit(2)
    # for opt, arg in opts:
    #     if opt == "--data":
    #         input_path_file = arg
    #     elif opt == "--learningRate":
    #         learning_rate = np.float(arg)
    #     elif opt == "--threshold":
    #         threshold = np.float(arg)
    #
    #     print(input_path_file,learning_rate, threshold)

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
    iteration = -1

    with open('assignment.csv', 'w') as f1:
        writer = csv.writer(f1, delimiter=',')
        while True:
            iteration += 1

            if iteration == 0:
                error = squarederror(instanceframe, Yattr, weights)
                previous_error = error
                writer.writerow(np.round(np.append(np.array(weights), error).flatten(),4))
                print(iteration, weights, error)
                gradient_matrix = gradient(instanceframe, Yattr, weights)
                weights = weights_update(weights, gradient_matrix, learning_rate)
                present_error = squarederror(instanceframe, Yattr, weights)

            if iteration >= 1 and abs(present_error - previous_error) > threshold:

                writer.writerow(np.round(np.append(np.array(weights), present_error).flatten(),4))
                print(iteration, weights, present_error)
                previous_error = present_error
                gradient_matrix = gradient(instanceframe, Yattr, weights)
                weights = weights_update(weights, gradient_matrix, learning_rate)
                present_error = squarederror(instanceframe, Yattr, weights)

            if iteration >= 1 and abs(present_error - previous_error) < threshold:
                iteration = iteration+1
                writer.writerow(np.round(np.append(np.array(weights), present_error).flatten(),4))
                print(iteration, weights, present_error)
                break


if __name__ == "__main__":
    # argv = sys.argv[1:]
    # print(argv)
    # main(argv)
    main()

# previous_error = 0
# error = 0
