import processing
import numpy as np
import math
from scipy.optimize import minimize

temp_heading = "Temperature (°C)"
cp_heading = "Heat Capacity (mJ/°C)"
time_heading = "Time (min)"


class DSCModel():
    def __init__(self, file):
        self.imported = processing.import_dsc_data(file)
        self.data = self.imported.data_frame
        self.data[cp_heading] = self.data[cp_heading].divide(self.imported.sample_mass)
        self.cp_data = self.data[cp_heading]
        self.temp_data = self.data[temp_heading]
        self.time_data = self.data[time_heading]
        self.region_start_index = 0
        self.region_end_index = 1
        self.linear_start_index = 0
        self.linear_end_index = 1
        self.enthalpy_guess = 45
        self.tg_guess = 45
        self.interpolation_start = 30
        self.interpolation_end = 160
        self.interpolation_step_size = .25
        self.tg_region_start = 0
        self.tg_region_end = 1
        self.ratio = .0

    def guess_interest_region(self):
        binned, change_regions = processing.bin_first_deriv(self.data[cp_heading])
        self.region_start_index, self.region_end_index = processing.suggest_overall_interest_regions(change_regions)

    def accept_interest_region(self):
        self.data = self.data.loc[self.region_start_index:  self.region_end_index, [temp_heading, cp_heading]]
        self.temp_data = self.data[temp_heading]
        self.cp_data = self.data[cp_heading]

    def set_intial_region(self, x_1, x_2):
        if x_1 < x_2:
            self.region_start_index = x_1
        else:
            self.region_end_index = x_2

    def validate_input(self, x, series):
        if x < series.min() or x > series.max():
            return False
        else:
            return True

    def most_close_index(self, value, series):
        return (series - value).abs().idxmin()

    def interpolate(self):

        steps = int(abs(self.interpolation_end - self.interpolation_start) / self.interpolation_step_size)
        self.interped = processing.interp_temp_cp(self.data, self.interpolation_start, self.interpolation_step_size,
                                                  steps)

    def guess_linear_region(self):
        zeros, self.inteped_zero_regions = processing.bin_first_deriv(self.interped[cp_heading])
        if len(self.inteped_zero_regions) == 0:
            return False
        longest_region = processing.suggest_linear_region(self.inteped_zero_regions)
        self.linear_end_index = longest_region[1]
        self.linear_start_index = longest_region[0]
        return True

    def guess_tg_region(self):
        if len(self.inteped_zero_regions) == 0:
            return False
        tg_suggestion = processing.suggest_tg_region(self.inteped_zero_regions)

        self.tg_region_end = tg_suggestion[1]
        self.tg_region_start = tg_suggestion[0]
        return True

    def fit_linear_model(self):
        fit_range_interp_data = self.interped.loc[self.linear_start_index:  self.linear_end_index,
                                [temp_heading, cp_heading]]

        # mb is a array of [ m ,b ] of y = m * x + b. These the values that will be minimized against to fit a curve
        def objective_function(mb):
            return math.sqrt(
                ((mb[0] * fit_range_interp_data[temp_heading] + mb[1]) - fit_range_interp_data[cp_heading]).pow(
                    2).sum())

        guesses = np.array([1, 1], dtype=float)
        lin_model = minimize(objective_function, guesses)
        self.lin_model_error = lin_model.fun
        self.lin_model_params = lin_model.x

    def apply_lin_model(self, series):
        return (self.lin_model_params[0] * series + self.lin_model_params[1]).to_frame(cp_heading)

    def print_linear_fit(self):
        print("Error %5.5f" % self.lin_model_error)
        print("%5.5f*x + %5.5f" % (self.lin_model_params[0], self.lin_model_params[1]))

    def fit_tg_model(self):
        self.transistion_range = self.interped.loc[self.tg_region_start:  self.tg_region_end,
                                 [temp_heading, cp_heading]]
        self.transistion_cp_linear_model = (
                    self.lin_model_params[0] * self.transistion_range[temp_heading] + self.lin_model_params[
                1]).to_frame(
            cp_heading)

        # Guess the glass transition temp, width, stp
        # Minimize error between tg model and observed cp

        tg_guesses = [self.tg_guess, 1, 1, self.enthalpy_guess, 1, 1, self.ratio]
        self.magic_number = 17.72432

        def model(guesses, minimize=True):

            t_g = guesses[0]
            width = guesses[1]
            stp = guesses[2]
            enthalpy = guesses[3]
            width_2 = guesses[4]
            max = guesses[5]
            ratio = guesses[6]
            gaus = processing.compute_gaussian(self.transistion_range[temp_heading], t_g, width, stp, self.magic_number)
            invs = processing.inverse_cumulative_gaussian(gaus, cp_heading)
            tg_model = self.transistion_cp_linear_model.reset_index()[cp_heading] - invs[cp_heading]
            enthalpy_distro = processing.compute_enthaply_distro(self.transistion_range[temp_heading], enthalpy,
                                                                 width_2, max)
            enthalpy_distro_2 = processing.compute_enthalpy_disro_2(self.transistion_range[temp_heading], enthalpy,
                                                                    width_2, max)
            full_model = processing.model_combonation(enthalpy, self.transistion_range[temp_heading], enthalpy_distro,
                                                      enthalpy_distro_2,
                                                      tg_model, ratio)
            if minimize:
                return np.sqrt(np.sum(np.power(self.transistion_range[cp_heading] - full_model, 2)))
            else:
                return full_model

        self.gaus_model = minimize(model, tg_guesses)

    def print_tg_model(self):
        print("Fitted Parameters")
        print("-----------------")
        print("T g: " + str(self.gaus_model.x[0]))
        print("Width: " + str(self.gaus_model.x[1]))
        print("Stp: " + str(self.gaus_model.x[2]))
        print("Enthalpy: " + str(self.gaus_model.x[3]))
        print("Width: " + str(self.gaus_model.x[4]))
        print("Max: " + str(self.gaus_model.x[5]))
        print("Guassian/Caucy Model Ratios: " + str(self.gaus_model.x[6]))
        print("Error: " + str(self.gaus_model.fun))

    def apply_model(self, guesses):

        t_g = guesses[0]
        width = guesses[1]
        stp = guesses[2]
        enthalpy = guesses[3]
        width_2 = guesses[4]
        max = guesses[5]
        ratio = guesses[6]
        gaus = processing.compute_gaussian(self.transistion_range[temp_heading], t_g, width, stp, self.magic_number)
        invs = processing.inverse_cumulative_gaussian(gaus, cp_heading)
        tg_model = self.transistion_cp_linear_model.reset_index()[cp_heading] - invs[cp_heading]
        enthalpy_distro = processing.compute_enthaply_distro(self.transistion_range[temp_heading], enthalpy, width_2,
                                                             max)
        enthalpy_distro_2 = processing.compute_enthalpy_disro_2(self.transistion_range[temp_heading], enthalpy, width_2,
                                                                max)
        full_model = processing.model_combonation(enthalpy, self.transistion_range[temp_heading], enthalpy_distro,
                                                  enthalpy_distro_2,
                                                  tg_model, ratio)

        return (full_model, enthalpy_distro, enthalpy_distro_2)
