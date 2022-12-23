import numpy as np
from matplotlib import pyplot as plt
from ANN import ANN
from LinReg import LinReg

#Calculate Standard Deviation
def calculate_std_dev(array, avg):
    std_dev = 0.0
    for instance in array:
      std_dev += ((instance - avg) ** 2)
    std_dev = std_dev / (len(train_input) - 1)
    std_dev = np.sqrt(std_dev)
    return std_dev

#Read the file and store the values in a numpy array
def read_file(filename):
    file = open(filename, "r")

    file_input = []
    file_output = []
    for line in file:
        line = line.strip().split("\t")
        file_input.append(line[0])
        file_output.append(line[1])
    file_input = np.array(file_input, dtype="float")
    file_output = np.array(file_output, dtype="float")
    file.close()
    return file_input, file_output


train_input, train_output = read_file("train1.txt")
test_input, test_output = read_file("test1.txt")

#Linear Regression
reg = LinReg(train_input, train_output)
reg.calculate_regression()
reg.plot_line()

#Selecting the minimum hidden unit count
print("Selecting the minimum hidden unit count")
for i in (2,4,8,16,32):
    iterations = 10000
    rate = 0.001
    ann = ANN(i)
    ann.train(train_input, train_output, iterations, rate)
    print(f"For {i} hidden unit:")
    ann.predict(train_input, train_output)

#Selecting the best learning rate
print("Selecting the best learning rate")
for i in (0.01, 0.001, 0.0001):
    iterations = 10000
    rate = i
    ann = ANN(32)
    ann.train(train_input, train_output, iterations, rate)
    print(f"For {i} learning rate:")
    ann.predict(train_input, train_output)

#Selecting the best epoch
print("Selecting the best epoch")
for i in (100, 1000, 10000, 100000, 1000000):
    iterations = i
    rate = 0.001
    ann = ANN(32)
    ann.train(train_input, train_output, iterations, rate)
    print(f"For {i} epoch:")
    ann.predict(train_input, train_output)

#ANN with selected configurations to test the given dataset
iterations = 100000
rate = 0.001  
ann = ANN(32)
ann.train(train_input, train_output, iterations, rate)
ann.predict(train_input, train_output)
ann.plot_graph(train_input, train_output)

ann.predict(test_input, test_output)
ann.plot_graph(test_input, test_output)

#Try different configurations for ANN
losses = np.zeros((10,2))

avg_loss = 0.0
iterations = 1000
rate = 0.001
ann = ANN(2)
ann.train(train_input, train_output, iterations, rate)
print("The train loss for 2 units:")
train_loss = ann.predict(train_input, train_output)
avg_loss = train_loss / len(train_input)
losses[0,0] = avg_loss
std_dev = calculate_std_dev(train_input, avg_loss)
losses[0,1] = std_dev
ann.plot_graph(train_input, train_output)
print("The test loss for 2 units:")
test_loss = ann.predict(test_input, test_output)
avg_loss = test_loss / len(test_input)
losses[1,0] = avg_loss
std_dev = calculate_std_dev(test_input, avg_loss)
losses[1,1] = std_dev

iterations = 10000
rate = 0.0001
ann = ANN(4)
ann.train(train_input, train_output, iterations, rate)
print("The train loss for 4 units:")
train_loss = ann.predict(train_input, train_output)
avg_loss = train_loss / len(train_input)
losses[2,0] = avg_loss
std_dev = calculate_std_dev(train_input, avg_loss)
losses[2,1] = std_dev
ann.plot_graph(train_input, train_output)
print("The test loss for 4 units:")
test_loss = ann.predict(test_input, test_output)
avg_loss = test_loss / len(test_input)
losses[3,0] = avg_loss
std_dev = calculate_std_dev(test_input, avg_loss)
losses[3,1] = std_dev

iterations = 100000
rate = 0.001
ann = ANN(8)
ann.train(train_input, train_output, iterations, rate)
print("The train loss for 8 units:")
train_loss = ann.predict(train_input, train_output)
avg_loss = train_loss / len(train_input)
losses[4,0] = avg_loss
std_dev = calculate_std_dev(train_input, avg_loss)
losses[4,1] = std_dev
ann.plot_graph(train_input, train_output)
print("The test loss for 8 units:")
test_loss = ann.predict(test_input, test_output)
avg_loss = test_loss / len(test_input)
losses[5,0] = avg_loss
std_dev = calculate_std_dev(test_input, avg_loss)
losses[5,1] = std_dev

iterations = 10000
rate = 0.01
ann = ANN(16)
ann.train(train_input, train_output, iterations, rate)
print("The train loss for 16 units:")
train_loss = ann.predict(train_input, train_output)
avg_loss = train_loss / len(train_input)
losses[6,0] = avg_loss
std_dev = calculate_std_dev(train_input, avg_loss)
losses[6,1] = std_dev
ann.plot_graph(train_input, train_output)
print("The test loss for 16 units:")
test_loss = ann.predict(test_input, test_output)
avg_loss = test_loss / len(test_input)
losses[7,0] = avg_loss
std_dev = calculate_std_dev(test_input, avg_loss)
losses[7,1] = std_dev

iterations = 10000
rate = 0.001
ann = ANN(32)
ann.train(train_input, train_output, iterations, rate)
print("The train loss for 32 units:")
train_loss = ann.predict(train_input, train_output)
avg_loss = train_loss / len(train_input)
losses[8,0] = avg_loss
std_dev = calculate_std_dev(train_input, avg_loss)
losses[8,1] = std_dev
ann.plot_graph(train_input, train_output)
print("The test loss for 32 units:")
test_loss = ann.predict(test_input, test_output)
avg_loss = test_loss / len(test_input)
losses[9,0] = avg_loss
std_dev = calculate_std_dev(test_input, avg_loss)
losses[9,1] = std_dev

#Print the losses
print(losses)