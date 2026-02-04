print("Viewin")

# Imports
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons
import datetime
# Resolves a weird error
import timeit
import importDataset as importData

s = 'print("startpoint")'

def view():
    allData = importData.importDataset()
    data, labels = importData.splitDataLabels(allData)

    time = np.arange(len(data[0]))
    index = 0

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.3)

    # Buttons
    axprev = plt.axes([0.25, 0.05, 0.12, 0.075])
    axnext = plt.axes([0.45, 0.05, 0.12, 0.075])
    bnext = Button(axnext, 'Next')
    bprev = Button(axprev, 'Back')
 
    axcheck = plt.axes([0.7, 0.05, 0.2, 0.15])
    check = CheckButtons(axcheck, ['Static Y'], [False])  # Default: floatin

    
    # Compute global y-limits for static mode
    global_ymin = np.min(data)
    global_ymax = np.max(data)

    def update_plot():
        ax.clear()  # Clear previous plot
        ax.plot(time, data[index], color='blue', linewidth=2)

        ax.set_xlabel('Sample')
        ax.set_ylabel('V')

        status = "Good" if labels[index] else "Bad"
        ax.set_title(f"Plot {index+1}- {status}")

        
        # Apply y-axis mode
        if check.get_status()[0]:  # Static mode
            ax.set_ylim(global_ymin, global_ymax)
        else:  # Floating mode
            ymin = np.min(data[index])
            ymax = np.max(data[index])
            pad = 0.05 * (ymax - ymin + 1e-12)
            ax.set_ylim(ymin - pad, ymax + pad)

        fig.canvas.draw_idle()

    def next_plot(event):
        nonlocal index
        if index < len(data) - 1:
            index += 1
            update_plot()

    def prev_plot(event):
        nonlocal index
        if index > 0:
            index -= 1
            update_plot()

    bnext.on_clicked(next_plot)
    bprev.on_clicked(prev_plot)

    # Checkbox binding
    def toggle_yaxis(label):
        update_plot()  # Just redraw with new mode
    check.on_clicked(toggle_yaxis)

    # keyboard funcs
    def on_key(event):
        # Normalize key string: event.key could be 'right', 'left', 'home', etc.
        k = event.key
        if k in ('right', 'd'):
            next_plot(event)
        elif k in ('left', 'a'):
            prev_plot(event)
        elif k in ('escape', 'esc'):
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)

    update_plot()  # Initial plot
    plt.show()

def viewFull():
    data = importData.importFullSignal()
    data, labels = importData.splitFullDataLabels(data)

    time = np.arange(len(data[0]))
    index = 0

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.3)

    # Buttons
    axprev = plt.axes([0.25, 0.05, 0.12, 0.075])
    axnext = plt.axes([0.45, 0.05, 0.12, 0.075])
    bnext = Button(axnext, 'Next')
    bprev = Button(axprev, 'Back')
 
    axcheck = plt.axes([0.7, 0.05, 0.2, 0.15])
    check = CheckButtons(axcheck, ['Static Y'], [False])  # Default: floatin

    
    # Compute global y-limits for static mode
    global_ymin = np.min(data)
    global_ymax = np.max(data)

    def update_plot():
        ax.clear()  # Clear previous plot
        ax.plot(time, data[index], color='blue', linewidth=2)

        ax.set_xlabel('Sample')
        ax.set_ylabel('V')

        status = "Good" if labels[index] else "Bad"
        ax.set_title(f"Plot {index+1}- {status}")

        
        # Apply y-axis mode
        if check.get_status()[0]:  # Static mode
            ax.set_ylim(global_ymin, global_ymax)
        else:  # Floating mode
            ymin = np.min(data[index])
            ymax = np.max(data[index])
            pad = 0.05 * (ymax - ymin + 1e-12)
            ax.set_ylim(ymin - pad, ymax + pad)

        fig.canvas.draw_idle()

    def next_plot(event):
        nonlocal index
        if index < len(data) - 1:
            index += 1
            update_plot()

    def prev_plot(event):
        nonlocal index
        if index > 0:
            index -= 1
            update_plot()

    bnext.on_clicked(next_plot)
    bprev.on_clicked(prev_plot)

    # Checkbox binding
    def toggle_yaxis(label):
        update_plot()  # Just redraw with new mode
    check.on_clicked(toggle_yaxis)

    # keyboard funcs
    def on_key(event):
        # Normalize key string: event.key could be 'right', 'left', 'home', etc.
        k = event.key
        if k in ('right', 'd'):
            next_plot(event)
        elif k in ('left', 'a'):
            prev_plot(event)
        elif k in ('escape', 'esc'):
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)

    update_plot()  # Initial plot
    plt.show()

viewFull()
