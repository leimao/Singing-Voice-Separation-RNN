
import os
import pandas as pd 
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_curve(step, loss, filename):

    fig = plt.figure(figsize=(12,8))
    plt.plot(step, loss, color = 'r', clip_on = False, label = 'Training Loss')
    plt.legend()
    plt.ylabel('Loss')
    plt.xlabel('Iteration')
    fig.savefig(filename, dpi = 600, bbox_inches = 'tight')
    plt.close()


def plot_training_curve(tensorboard_log, output_directory):

    df = pd.read_csv(tensorboard_log)
    step = df['Step'].as_matrix()
    loss = df['Value'].as_matrix()

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    plot_curve(step = step, loss = loss, filename = os.path.join(output_directory, 'training_loss.png'))

if __name__ == '__main__':
    
    plot_training_curve(tensorboard_log = 'statistics/run_20180516-190554-tag-summaries_loss.csv', output_directory = 'figures')