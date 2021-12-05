import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np


class MainFrame(tk.Frame):
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.plot = DataPlot(self)
        self.plot.draw()
        self.plot.get_tk_widget().pack()
        self.tool_bar = NavigationToolbar2Tk(self.plot, self)
        self.tool_bar.update()
        self.plot.get_tk_widget().pack()

        # button_frame = tk.Label(self)
        # self.param_1_label = tk.Label(button_frame, text = "Parameter 1")
        # self.param_1_box = tk.Entry(button_frame)
        # self.param_2_label = tk.Label(button_frame, text="Parameter 2")
        # self.param_2_box = tk.Entry(button_frame)
        # self.accept_button = tk.Button(button_frame, text = "Accept Paramters")
        #
        #
        # self.param_1_label.grid(row=0, column =0)
        # self.param_1_box.grid(row=1, column =0)
        # self.param_2_label.grid(row=2, column =0)
        # self.param_2_box.grid(row=3, column =0)
        # self.accept_button.grid(row=3, column =1)
        # button_frame.pack()
        self.grid()


class DataPlot(FigureCanvasTkAgg):
    def __init__(self, master):
        self.figure = Figure()
        FigureCanvasTkAgg.__init__(self, self.figure, master=master)
        self.ax = self.figure.add_subplot(1, 1, 1)
        self.ax.set_autoscaley_on(True)  # Y fixed
