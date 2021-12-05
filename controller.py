import argparse
import threading

import matplotlib.pyplot as plt

from model import DSCModel

parser = argparse.ArgumentParser()

parser.add_argument("-f", "--file", type=str, help="DSC file to parsed")
parser.add_argument("-v", "--version", type=bool, help="Current Tgmon version")
args = parser.parse_args()
file = args.file
print_version = args.version
if print_version:
    print("Version 20200421")

temp_heading = "Temperature (°C)"
cp_heading = "Heat Capacity (mJ/°C)"
time_heading = "Time (min)"
modes = "Intial Region"


class GTPMain():
    def __init__(self):

        self.model = DSCModel(file)
        self.model.guess_interest_region()
        self.query_thread = threading.Thread(target=self.run_model)
        self.query_thread.start()
        self.draw_intial_region()
        plt.show()

    def draw_intial_region(self):

        plt.clf()
        plt.title("DSC data with initial guesses")
        plt.plot(self.model.time_data, self.model.cp_data)
        plt.plot(
            [self.model.time_data[self.model.region_start_index], self.model.time_data[self.model.region_end_index]],
            [self.model.cp_data[self.model.region_start_index], self.model.cp_data[self.model.region_end_index]], "g^")

    def run_model(self):
        self.query_initial_regions()

        self.query_interpolations()

        self.query_linear_region()

        self.query_tg_region()

    def query_initial_regions(self):
        print("Region of interst guess is (leave blank to accept guesses) :")
        print("From %5.2f (min)" % self.model.time_data[self.model.region_start_index])
        print("To %5.2f (min)" % self.model.time_data[self.model.region_end_index])
        start = self.handle_input("Time of start (min)->", self.model.time_data)
        if start != "":
            self.model.region_start_index = self.model.most_close_index(start, series=self.model.time_data)
        end = self.handle_input("Time of end (min)->", self.model.time_data)
        if end != "":
            self.model.region_end_index = self.model.most_close_index(end, series=self.model.time_data)
        self.draw_intial_region()
        plt.draw()
        accept = input("Accept region Y/N?")
        if accept.upper() != "Y":
            self.query_initial_regions()
        else:
            self.model.accept_interest_region()
            plt.clf()
            plt.title("Region of interest for " + self.model.imported.name)
            plt.plot(self.model.temp_data, self.model.cp_data, label=self.model.imported.name)
            plt.legend()
            plt.draw()

    def query_interpolations(self):
        print("Interpolation  defaults (leave blank to accept): ")
        print("Start %5.2f (°C)" % self.model.interpolation_start)
        print("End %5.2f (°C)" % self.model.interpolation_end)
        print("Step size %5.3f (°C) " % self.model.interpolation_step_size)
        start = self.handle_input("Interpolation start (°C)->")
        if start != "":
            self.model.interpolation_start = float(start)
        end = self.handle_input("Interpolation end (°C)->")
        if end != "":
            self.model.interpolation_end = float(end)
        step_size = self.handle_input("Interpolation step size (°C)->")
        if step_size != "":
            self.model.interpolation_step_size = float(step_size)
        self.model.interpolate()
        plt.clf()
        plt.title("Interpolated DSC Data")
        plt.plot(self.model.interped[temp_heading], self.model.interped[cp_heading], label=self.model.imported.name)
        plt.legend()
        plt.draw()
        accept = input("Accept interpolation Y/N?")
        if accept.upper() != "Y":
            self.query_interpolations()
        else:
            if self.model.guess_linear_region() == False:
                noLinear = input("Linear Region not found: Procced without guesses Y/N")
                if noLinear.upper() != "Y":
                    self.query_interpolations()
                else:
                    lin_start = self.handle_input("Linear fit start (°C)->", self.model.interped[temp_heading])
                    while (lin_start == ""):
                        lin_start = self.handle_input("Linear fit start (°C)->", self.model.interped[temp_heading])

                    self.model.linear_start_index = self.model.most_close_index(lin_start, series=self.model.interped[
                        temp_heading])
                    end = self.handle_input("Linear fit end (°C)->", self.model.interped[temp_heading])
                    while (end == ""):
                        end = self.handle_input("Linear fit end (°C)->", self.model.interped[temp_heading])
                    if end != "":
                        self.model.linear_end_index = self.model.most_close_index(end, series=self.model.interped[
                            temp_heading])

    def query_linear_region(self):
        plt.title("Select linear region")
        plt.plot([self.model.interped[temp_heading][self.model.linear_start_index],
                  self.model.interped[temp_heading][self.model.linear_end_index]],
                 [self.model.interped[cp_heading][self.model.linear_start_index],
                  self.model.interped[cp_heading][self.model.linear_end_index]], "g^", label="Linear region guesses")
        plt.legend()
        plt.draw()
        print("Linear fit region guesses (leave blank to accept):")
        print("Start %5.2f (°C)" % self.model.interped[temp_heading][self.model.linear_start_index])
        print("End %5.2f (°C)" % self.model.interped[temp_heading][self.model.linear_end_index])
        start = self.handle_input("Linear fit start (°C)->", self.model.interped[temp_heading])
        if start != "":
            self.model.linear_start_index = self.model.most_close_index(start, series=self.model.interped[temp_heading])
        end = self.handle_input("Linear fit end (°C)->", self.model.interped[temp_heading])
        if end != "":
            self.model.linear_end_index = self.model.most_close_index(end, series=self.model.interped[temp_heading])
        plt.clf()
        plt.plot([self.model.interped[temp_heading][self.model.linear_start_index],
                  self.model.interped[temp_heading][self.model.linear_end_index]],
                 [self.model.interped[cp_heading][self.model.linear_start_index],
                  self.model.interped[cp_heading][self.model.linear_end_index]], "g^")
        plt.plot(self.model.interped[temp_heading], self.model.interped[cp_heading])
        plt.draw()

        print("Applying linear fit....")
        self.model.fit_linear_model()
        self.model.print_linear_fit()
        plt.plot(self.model.interped[temp_heading], self.model.apply_lin_model(self.model.interped[temp_heading]),
                 label="Linear Fit")
        plt.legend()
        plt.draw()
        accept = input("Accept linear fit Y/N?")
        if accept.upper() != "Y":
            self.query_linear_region()
        else:
            if self.model.guess_tg_region() == False:
                manual = input("TG region not found. Procced with out guesses? Y/N")
                if manual.upper() != "Y":
                    self.query_interpolations()
                else:
                    start = self.handle_input("TG region start(°C)->", self.model.interped[temp_heading])
                    while start == "":
                        start = self.handle_input("TG region start(°C)->", self.model.interped[temp_heading])
                    self.model.tg_region_start = self.model.most_close_index(start, series=self.model.interped[
                        temp_heading])
                    end = self.handle_input("TG region end (°C)->", self.model.interped[temp_heading])
                    while end == "":
                        end = self.handle_input("TG region end (°C)->", self.model.interped[temp_heading])
                    self.model.tg_region_end = self.model.most_close_index(end,
                                                                           series=self.model.interped[temp_heading])

    def query_tg_region(self):

        plt.clf()
        plt.title("Select Glass Transition Region")
        plt.plot(self.model.interped[temp_heading], self.model.interped[cp_heading], label=self.model.imported.name)

        plt.plot([self.model.interped[temp_heading][self.model.tg_region_start],
                  self.model.interped[temp_heading][self.model.tg_region_end]],
                 [self.model.interped[cp_heading][self.model.tg_region_start],
                  self.model.interped[cp_heading][self.model.tg_region_end]], "g^", label="TG Region Guesses")
        plt.legend()
        plt.draw()
        print("Glass Transistion Region Guess  (leave blank to accept):")
        print("Start %5.2f (°C)" % self.model.interped[temp_heading][self.model.tg_region_start])
        print("End %5.2f (°C)" % self.model.interped[temp_heading][self.model.tg_region_end])
        start = self.handle_input("TG region start(°C)->", self.model.interped[temp_heading])
        if start != "":
            self.model.tg_region_start = self.model.most_close_index(start, series=self.model.interped[temp_heading])
        end = self.handle_input("TG region end (°C)->", self.model.interped[temp_heading])
        if end != "":
            self.model.tg_region_end = self.model.most_close_index(end, series=self.model.interped[temp_heading])
        plt.clf()
        plt.plot([self.model.interped[temp_heading][self.model.tg_region_start],
                  self.model.interped[temp_heading][self.model.tg_region_end]],
                 [self.model.interped[cp_heading][self.model.tg_region_start],
                  self.model.interped[cp_heading][self.model.tg_region_end]], "g^")
        plt.plot(self.model.interped[temp_heading], self.model.interped[cp_heading])
        plt.draw()
        self.query_fit_guesses()
        self.model.fit_tg_model()
        self.model.print_tg_model()
        plt.clf()
        plt.title("Predicted Glass Transition for " + self.model.imported.name)
        model = self.model.apply_model(self.model.gaus_model.x)
        plt.subplot(2, 1, 1)
        plt.plot(self.model.transistion_range[temp_heading], model[0], label="Glass Transition Model")
        plt.plot(self.model.transistion_range[temp_heading], self.model.transistion_range[cp_heading],
                 label="Interpolated Heat Capicity")
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(self.model.transistion_range[temp_heading], model[1], label="Gaussian Model")
        plt.plot(self.model.transistion_range[temp_heading], model[2], label="Cauchy Model")
        plt.legend()
        plt.draw()
        accept = input("Accept glass transition calculations Y/N?")

        if accept.upper() != "Y":
            self.query_tg_region()
        else:
            quit()

    def query_fit_guesses(self):
        print("Enter Enthalpy and Glass Tansistion Guesses (leave blank to accept default)")
        print("Entahlpy guess %5.4f" % self.model.enthalpy_guess)
        print("Tg guess %5.4f" % self.model.tg_guess)
        print("Ratio between enthalpy models guess %1.2f" % self.model.ratio)
        print("Between 1 and -1. 1 = 100% Guassian; -1 = 100% Cauchy; .0 == 50%-%50 split")
        try:
            e_guess = input("Enthalpy guess ->")
            if e_guess != "":
                self.model.enthalpy_guess = float(e_guess)
            tg_guess = input("Tg guess->")
            if tg_guess != "":
                self.model.tg_guess = float(tg_guess)
            ratio_guess = input("Ratio guess->")
            if ratio_guess != "":
                self.model.ratio = float(ratio_guess)
        except ValueError:
            print("Enter valid floating point entry")
            self.query_fit_guesses()

    def handle_input(self, prompt, series=None):
        valid = False
        while not valid:
            ans = input(prompt)
            if ans != "":
                ans = float(ans)
                if series is not None:
                    valid = self.model.validate_input(ans, series)
                else:
                    valid = True
            else:
                valid = True
        return ans


app = GTPMain()
