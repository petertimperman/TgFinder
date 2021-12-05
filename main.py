import argparse
from processing import *
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import math

parser = argparse.ArgumentParser()

parser.add_argument("-f", "--file", type=str, help="DSC file to parsed")
parser.add_argument("-start", type=float, help="Temp at start of region of interest")
parser.add_argument("-end", type=float, help="Temp at end of region of interest")
parser.add_argument("-tg", "--tg_guess", type=float, help="DSC file to parsed")
parser.add_argument("-eg", "--enthalpy_guess", type=float, help="Time at start of region of interest")
parser.add_argument("-is", "--interloplation_start", type=float, help="Temperature at start of interpolation region")
parser.add_argument("-ss", "--step_size", type=float, help="Interpolation step size")
parser.add_argument("-s", "--steps", type=int, help="Number of data points to interpolate")
parser.add_argument("-fs", "--fit_start", type=float, help="Start of linear region to fit")
parser.add_argument("-fe", "--fit_end", type=float, help="End of linear region to fit")
parser.add_argument("-ts", "--tg_start_region", type=float, help="Start of tg region to be modeled")
parser.add_argument("-te", "--tg_end_region", type=float, help="End of tg region to be modeled")
parser.add_argument("-v", "--verbose", help="Turn on verbose mode", action="store_true")
args = parser.parse_args()

data_file = args.file
start_temp = args.start
end_temp = args.end
interloplation_start = args.interloplation_start
steps = args.steps
step_size = args.step_size
tg_guess = args.tg_guess
enthalpy_guess = args.enthalpy_guess
fit_start = args.fit_start
fit_end = args.fit_end
tg_start_region = args.tg_start_region
tg_end_region = args.tg_end_region
verbose = args.verbose

temp_heading = "Temperature (°C)"
cp_heading = "Heat Capacity (mJ/°C)"
time_heading = "Time (min)"

dsc_data = import_dsc_data(data_file)

# Filter out relevant region and data

data = select_between_range(dsc_data.data_frame, start_temp, end_temp, time_heading, verbose=verbose)[
    [temp_heading, cp_heading]]
# Correct for mass
data[cp_heading] = data[cp_heading].divide(dsc_data.sample_mass)
# Interpolate the cp values for a temp with an ordered step size

interp_data = interp_temp_cp(data, interloplation_start, step_size, steps, verbose=verbose)
# Fit a linear curve to the interpolated heat capacity
fit_range_interp_data = select_between_range(interp_data, fit_start, fit_end, temp_heading,
                                             verbose=verbose)  # range of data to fit the curve


# mb is a array of [ m ,b ] of y = m * x + b. These the values that will be minimized against to fit a curve
def objective_function(mb):
    return math.sqrt(
        ((mb[0] * fit_range_interp_data[temp_heading] + mb[1]) - fit_range_interp_data[cp_heading]).pow(2).sum())


guesses = np.array([1, 1], dtype=float)
if verbose:
    print("Fitting c_p =m*temp + b from %5.2f to %5.2f..." % (fit_start, fit_end))
lin_model = minimize(objective_function, guesses)
if verbose:
    print("m:%5.6f b:%5.2f" % (lin_model.x[0], lin_model.x[1]))
# plt.plot(interp_data[temp_heading], (lin_model.x[0]*interp_data[temp_heading]+lin_model.x[1]))
# plt.plot(interp_data[temp_heading], interp_data[cp_heading])

transistion_range = select_between_range(interp_data, tg_start_region, tg_end_region, temp_heading, verbose=verbose)

# apply linear model to the transistion range
transistion_cp_linear_model = (lin_model.x[0] * transistion_range[temp_heading] + lin_model.x[1]).to_frame(cp_heading)

# Guess the glass transition temp, width, stp
# Minimize error between tg model and observed cp


tg_guesses = [tg_guess, 1, 1, enthalpy_guess, 1, 1]
magic_number = 17.72432


def model(guesses, minimize=True):
    if verbose == True and minimize == True:
        print(guesses)
    t_g = guesses[0]
    width = guesses[1]
    stp = guesses[2]
    enthalpy = guesses[3]
    width_2 = guesses[4]
    max = guesses[5]
    gaus = compute_gaussian(transistion_range[temp_heading], t_g, width, stp, magic_number)
    # gaus_cumalitve = compute_cumulative_guassuan(gaus, cp_heading, verbose = True)
    invs = inverse_cumulative_gaussian(gaus, cp_heading)
    tg_model = transistion_cp_linear_model.reset_index()[cp_heading] - invs[cp_heading]
    enthalpy_distro = compute_enthaply_distro(transistion_range[temp_heading], enthalpy, width_2, max)
    enthalpy_distro_2 = compute_enthalpy_disro_2(transistion_range[temp_heading], enthalpy, width_2, max)
    full_model = model_combonation(enthalpy, transistion_range[temp_heading], enthalpy_distro, enthalpy_distro_2,
                                   tg_model)
    if minimize:
        return np.sqrt(np.sum(np.power(transistion_range[cp_heading] - full_model, 2)))
    else:
        return full_model


gaus_model = minimize(model, tg_guesses)
print("Fitted Parameters")
print("-----------------")
print("T g: " + str(gaus_model.x[0]))
print("Width: " + str(gaus_model.x[1]))
print("Stp: " + str(gaus_model.x[2]))
print("Enthalpy: " + str(gaus_model.x[3]))
print("Width: " + str(gaus_model.x[4]))
print("Max: " + str(gaus_model.x[5]))
print("Error: " + str(gaus_model.fun))
