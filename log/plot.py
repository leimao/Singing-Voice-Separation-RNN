
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np


def plot_curve(iteration, training_loss, test_loss):

    # Plot the a cross validation curve using Matplotlib

    fig = plt.figure(figsize=(10,4))
    plt.rc('font', weight='bold')
    plt.plot(iteration, training_loss, color = 'r', clip_on = False, label = 'Training')
    plt.plot(iteration, test_loss, color = 'b', clip_on = False, label = 'Validation')
    plt.legend()
    plt.ylabel('Loss', fontsize = 16, fontweight = 'bold')
    plt.xlabel('Iteration', fontsize = 16, fontweight = 'bold')
    plt.xlim(iteration[0], iteration[-1])
    fig.savefig('vs_rnn_validation.png', format = 'png', dpi = 600, bbox_inches = 'tight')
    plt.close()


def main():

    df = pd.read_csv('train_log.csv', header = None)
    iteration = df.iloc[:, 0].as_matrix()
    training_loss = df.iloc[:, 1].as_matrix()
    test_loss = df.iloc[:, 2].as_matrix()
    plot_curve(iteration = iteration, training_loss = training_loss, test_loss = test_loss)

if __name__ == '__main__':
    
    main()