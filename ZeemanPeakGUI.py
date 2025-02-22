import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

## This GUI allows for manual identification of Zeeman peaks. Then, using this initial guess, a quadratic
# Curve is fit to identify peak location. The output file is read by Data_vs_model.py

def filter_by_limit(X, Y, min , max):
    X1 = [i for i in X if min <= i <= max]
    Y1 = [Y[i] for i, x in enumerate(X) if min <= x <= max]
    return np.array(X1), np.array(Y1)

data = []
scope = []
spacing = []
spacing_data = []

# Load data
arr = np.loadtxt(f"tek0000ALL.csv",
                 delimiter=",", dtype=str)

data.append(arr[2:, 0].astype(float)) # Append common time
scope.append(arr[2:, 0].astype(float))
for i in range(0, 57):
    if i != 15: # Data 15 was corrupted
        arr = np.loadtxt(f"tek{i:04}ALL.csv",
                     delimiter=",", dtype=str)
        data.append(arr[2:, 1].astype(float))
        scope.append(arr[2:, 2].astype(float))
data = np.array(data).T
scope = np.array(scope).T

window_length = 300 # Savgol filter length
trace = 2
xdat, ydat = data[:, 0], data[:, trace]
ydat = ydat - np.min(ydat)
ydat = savgol_filter(ydat, window_length, 2)
xmin = np.min(xdat); xmax = np.max(xdat)

# Create the figure and axis
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.4)
line, = ax.plot(xdat, ydat, label='Signal')


# Initial positions of vertical lines
x1_init, x2_init, x3_init = xmax, xmax, xmax
vline1 = ax.axvline(x1_init, color='r', linestyle='--', label='Line 1')
vline2 = ax.axvline(x2_init, color='g', linestyle='--', label='Line 2')
vline3 = ax.axvline(x3_init, color='b', linestyle='--', label='Line 3')

vline4 = ax.axvline(xmin, color = 'black')
vline5 = ax.axvline(xmax, color = 'black')




# Define slider axes
ax_slider1 = plt.axes([0.2, 0.15, 0.65, 0.03])
ax_slider2 = plt.axes([0.2, 0.10, 0.65, 0.03])
ax_slider3 = plt.axes([0.2, 0.05, 0.65, 0.03])
ax_slider4 = plt.axes([0.2, 0.20, 0.65, 0.03])
ax_slider5 = plt.axes([0.2, 0.25, 0.65, 0.03])
ax_slider6 = plt.axes([0.2, 0.30, 0.65, 0.03])

# Create sliders
slider1 = Slider(ax_slider1, 'Line 1', xmin, xmax, valinit=x1_init)
slider2 = Slider(ax_slider2, 'Line 2', xmin, xmax, valinit=x2_init)
slider3 = Slider(ax_slider3, 'Line 3', xmin, xmax, valinit=x3_init)
slider4 = Slider(ax_slider4, 'Window', 0, 5000, valinit=500, valstep=1) # Sets window length for quadratic fit
slider5 = Slider(ax_slider5, 'Min x', xmin, xmax, valinit=xmin) # These really dont have any function, they were previously used for scope calibration..
slider6 = Slider(ax_slider6, 'Max x', xmin, xmax, valinit=xmax)


# Update function
def update(val):
    vline1.set_xdata(slider1.val)
    vline2.set_xdata(slider2.val)
    vline3.set_xdata(slider3.val)
    vline4.set_xdata(slider5.val)
    vline5.set_xdata(slider6.val)
    fig.canvas.draw_idle()

slider1.on_changed(update)
slider2.on_changed(update)
slider3.on_changed(update)
slider4.on_changed(update)
slider5.on_changed(update)
slider6.on_changed(update)

# Button to print values
ax_button = plt.axes([0.5, 0.35, 0.2, 0.05])
button = Button(ax_button, 'Fit')
ax_button2 = plt.axes([0.3, 0.35, 0.2, 0.05])
button2 = Button(ax_button2, 'Save/next')

def next(event):
    global trace, vline1, vline2, vline3, vline4, vline5, spacing_data, xdat, ydat, window_length
    spacing_data.append(spacing)
    np.savetxt("spacing_data.csv", spacing_data, delimiter=",")
    print(f'Trace {trace} saved')
    ax.clear()
    trace += 1
    xdat, ydat = data[:, 0], data[:, trace]
    ydat = ydat - np.min(ydat)
    ydat = savgol_filter(ydat, window_length, 2)
    ax.plot(xdat, ydat)

    vline1 = ax.axvline(slider1.val, color='r', linestyle='--', label='Line 1')
    vline2 = ax.axvline(slider2.val, color='g', linestyle='--', label='Line 2')
    vline3 = ax.axvline(slider3.val, color='b', linestyle='--', label='Line 3')

    vline4 = ax.axvline(slider5.val, color='black')
    vline5 = ax.axvline(slider6.val, color='black')

    fig.canvas.draw_idle()

def fit(event):
    global trace, data, scope, spacing, xdat, ydat

    x, y = xdat, ydat
    refined_peaks = []
    window = slider4.val
    if window == 0: # Sometimes for low magnetic fields, the quadratic fit will be finicky, this allows complete manual identification, if needed.
        refined_peaks.append(slider1.val)
        refined_peaks.append(slider2.val)
        refined_peaks.append(slider3.val)

    else:
        def quadratic(x, a, b, c):
            return a * x ** 2 + b * x + c

        peaks = [np.abs(xdat - slider1.val).argmin(), np.abs(xdat - slider2.val).argmin(), np.abs(xdat - slider3.val).argmin()]
        for peak in peaks:
            start = max(0, peak - window)
            end = min(len(x), peak + window + 1)
            x_subset = x[start:end]
            y_subset = y[start:end]

            params = curve_fit(quadratic, x_subset, y_subset)[0]

            a, b, c = params
            refined_x = -b / (2 * a) # Apex of quadratic fit
            ax.plot(x_subset, quadratic(x_subset, a, b, c))
            refined_peaks.append(refined_x)

    for p in refined_peaks:
        ax.axvline(p, color='purple', linestyle='--')
    spacing = [refined_peaks[0], refined_peaks[1], refined_peaks[2]]


    fig.canvas.draw_idle()


button.on_clicked(fit)
button2.on_clicked(next)

plt.legend()
plt.show()