import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit

## GUI to manually adjust peak finding settings and verify correct peaks are being identified

L = .539 # Cavity Length
dL = .001 # Cavity uncertainty
c = 299792458
FC = c / (2*L*1.0003)

# Fit a quadratic curve to the neighborhood around a peak
def refine_peak(x, y, prominence, distance, window = 50):
    def quadratic(x, a, b, c):
        return a * x ** 2 + b * x + c
    peaks, _ = find_peaks(y, prominence=prominence, distance=distance)

    refined_peaks = []
    for peak in peaks:
        start = max(0, peak - window)
        end = min(len(x), peak + window + 1)
        x_subset = x[start:end]
        y_subset = y[start:end]

        params = curve_fit(quadratic, x_subset, y_subset)[0]

        a, b, c = params
        refined_x = -b / (2 * a)
        refined_peaks.append(refined_x)
    return refined_peaks

def filter_by_ranges(arr, range1, range2):
    # Removes all items not in range1 or range 2

    min_value1, max_value1 = range1
    min_value2, max_value2 = range2

    filtered_array = [x for x in arr if (min_value1 <= x <= max_value1) or (min_value2 <= x <= max_value2)]
    return filtered_array

def filter_by_limit(X, Y, min , max):
    X1 = [i for i in X if min <= i <= max]
    Y1 = [Y[i] for i, x in enumerate(X) if min <= x <= max]
    return np.array(X1), np.array(Y1)

# Calculate osclation distance between two peaks
def oscillation_distance(peak_x, x1, x2):
    x1, x2 = sorted([x1, x2])

    # Find indx of left and right peak
    if x1 <= peak_x[0]:
        left_peak_idx = 0
        x1_fraction = -1 * (peak_x[left_peak_idx] - x1) / (peak_x[left_peak_idx + 1] - peak_x[left_peak_idx])
    else:
        left_peak_idx = max(i for i in range(len(peak_x)) if peak_x[i] < x1)
        x1_fraction = (x1 - peak_x[left_peak_idx]) / (peak_x[left_peak_idx + 1] - peak_x[left_peak_idx])
    if x2 >= peak_x[-1]:
        right_peak_idx = -1
        x2_fraction = (x2 - peak_x[right_peak_idx]) / (peak_x[right_peak_idx] - peak_x[right_peak_idx - 1])
        right_peak_idx = len(peak_x) - 1
    else:
        right_peak_idx = max(i for i in range(len(peak_x)) if peak_x[i] < x2)
        x2_fraction = (x2 - peak_x[right_peak_idx]) / (peak_x[right_peak_idx + 1] - peak_x[right_peak_idx])

    # add together
    oscillations = (right_peak_idx - left_peak_idx) + x2_fraction - x1_fraction
    return oscillations




minx1 =.031; maxx1 = .035; minx2 =.076; maxx2 = .08;
scope_min = .02; scope_max = .095

# Initial peaks detection
initial_prominence1 = .0094
initial_distance1 = 10
initial_prominence2 = .01
initial_distance2 = 20

A_E = []
F_C = []
E_F = []
A_B = []
B_C = []
D_E = []

spacingdata = []


data = []
scope = []

# Load osciliscope data
arr = np.loadtxt(f"Data/tek0000ALL.csv",
                 delimiter=",", dtype=str)

data.append(arr[2:, 0].astype(float)) # Append common time
scope.append(arr[2:, 0].astype(float))
for i in range(0, 100):
    arr = np.loadtxt(f"Data/tek{i:04}ALL.csv",
                 delimiter=",", dtype=str)
    data.append(arr[2:, 1].astype(float))
    scope.append(arr[2:, 2].astype(float))
data = np.array(data).T
scope = np.array(scope).T


n = 1
window = 50
x = data[:, 0] - np.min(data[:, 0])
signal1x, signal1y = filter_by_limit(x, data[:, 1], minx1, maxx1) # First
signal2x, signal2y = filter_by_limit(x, data[:, 1], minx2, maxx2)
signal3x, signal3y = filter_by_limit(x, scope[:, 1], scope_min, scope_max) # Scope

fig, axes = plt.subplots(3, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 1, 1]})
plt.subplots_adjust(left=0.3, right=0.95, bottom=0.1, top=0.95, hspace=0.5)

ax1, ax2, ax3 = axes

peaks1 = refine_peak(signal1x, signal1y, prominence=initial_prominence1, distance=initial_distance1, window = window)
peaks2 = refine_peak(signal2x, signal2y, prominence=initial_prominence1, distance=initial_distance1, window = window)
peaks3 = refine_peak(signal3x, signal3y, prominence=initial_prominence2, distance=initial_distance2, window = window)

(line1,) = ax1.plot(signal1x, signal1y, label="Signal 1")
ax1.vlines(peaks1, ymin=np.min(signal1y), ymax=np.max(signal1y), color='r', linestyle='--', label='Peaks')
ax1.legend()

(line2,) = ax2.plot(signal2x, signal2y, label="Signal 1 Duplicate")
ax2.vlines(peaks2, ymin=np.min(signal2y), ymax=np.max(signal2y), color='r', linestyle='--', label='Peaks')
ax2.legend()

(line3,) = ax3.plot(signal3x, signal3y, label="Signal 2")
ax3.vlines(peaks3, ymin=np.min(signal3y), ymax=np.max(signal3y), color='r', linestyle='--', label='Peaks')
ax3.legend()

slider_ax1_prominence = plt.axes([0.05, 0.7, 0.2, 0.03], frameon=True)
slider_ax1_distance = plt.axes([0.05, 0.65, 0.2, 0.03], frameon=True)
slider_ax1_filter = plt.axes([0.05, 0.6, 0.2, 0.03], frameon=True)


slider1_prominence = Slider(slider_ax1_prominence, 'Prominence 1', .005, .015, valinit=initial_prominence1)
slider1_distance = Slider(slider_ax1_distance, 'Distance 1', 1, 100, valinit=initial_distance1, valstep=1)
slider1_filter = Slider(slider_ax1_filter, 'filter 1', 2, 101, valinit=1, valstep = 2)

# Add sliders for Signal 2
slider_ax2_prominence = plt.axes([0.05, 0.4, 0.2, 0.03], frameon=True)
slider_ax2_distance = plt.axes([0.05, 0.35, 0.2, 0.03], frameon=True)
slider_ax2_filter = plt.axes([0.05, 0.3, 0.2, 0.03], frameon=True)

slider2_prominence = Slider(slider_ax2_prominence, 'Prominence 2', 0, .1, valinit=initial_prominence2)
slider2_distance = Slider(slider_ax2_distance, 'Distance 2', 1, 100, valinit=initial_distance2, valstep=1)
slider2_filter = Slider(slider_ax2_filter, 'filter 2', 2, 101, valinit=1, valstep = 2)

# Add a save button
button_ax = plt.axes([0.05, 0.1, 0.2, 0.05])
button2_ax = plt.axes([0.05, 0.2, 0.2, 0.05])
save_button = Button(button_ax, 'Save')
save_button2 = Button(button2_ax, 'Save to CSV')


scope_peaks = []
SAS_peaks = []
# Update function for sliders
def update_signal1(val):
    global SAS_peaks, signal1x, signal1y, signal2x, signal2y
    prominence = slider1_prominence.val
    distance = slider1_distance.val
    filter = slider1_filter.val

    filt_sig1 = savgol_filter(signal1y, filter, 2)
    filt_sig2 = savgol_filter(signal2y, filter, 2)
    peaks1 = refine_peak(signal1x, filt_sig1, prominence=prominence, distance=distance, window=window)
    peaks2 = refine_peak(signal2x, filt_sig2, prominence=prominence, distance=distance, window=window)

    ax1.clear()
    ax1.plot(signal1x, filt_sig1, label="Signal 1")
    ax1.vlines(peaks1, ymin=min(filt_sig1), ymax=max(filt_sig1), color='r', linestyle='--', label='Peaks')
    ax1.legend()

    ax2.clear()
    ax2.plot(signal2x, filt_sig2, label="Signal 1 Duplicate")
    ax2.vlines(peaks2, ymin=min(filt_sig2), ymax=max(filt_sig2), color='r', linestyle='--', label='Peaks')
    ax2.legend()
    fig.canvas.draw_idle()

    SAS_peaks = np.concatenate([np.array(peaks1), np.array(peaks2)])

def update_signal2(val):
    global scope_peaks, data
    prominence = slider2_prominence.val
    distance = slider2_distance.val
    filter = slider2_filter.val

    filt_sig3 = savgol_filter(signal3y, filter, 2)
    peaks = refine_peak(signal3x, filt_sig3, prominence=prominence, distance=distance, window=window)
    ax3.clear()
    ax3.plot(signal3x, filt_sig3, label="Signal 2")
    ax3.vlines(peaks, ymin=min(filt_sig3), ymax=max(filt_sig3), color='r', linestyle='--', label='Peaks')
    ax3.legend()
    #ax3.plot(data[:, 0], data[:, 1]/5, alpha = .2)
    fig.canvas.draw_idle()

    scope_peaks = peaks
    scope_peaks = peaks

# Connect sliders to update functions
slider1_prominence.on_changed(update_signal1)
slider1_distance.on_changed(update_signal1)
slider1_filter.on_changed(update_signal1)

slider2_prominence.on_changed(update_signal2)
slider2_distance.on_changed(update_signal2)
slider2_filter.on_changed(update_signal2)

# Placeholder for save button
def save_csv(event):
    a = np.asarray([B_C, A_B, E_F, F_C, A_E, D_E]).T
    np.savetxt("data.csv", a, delimiter=",")
    print('Data saved')
def save_action(event):
    global n, signal1x, signal1y, signal2x, signal2y, signal3x, signal3y, scope_peaks, SAS_peaks, x
    if len(SAS_peaks) == 10:

        slider1_prominence.set_val(slider1_prominence.val)
        slider1_distance.set_val(slider1_distance.val)
        slider1_filter.set_val(slider1_filter.val)

        slider2_prominence.set_val(slider2_prominence.val)
        slider2_distance.set_val(slider2_distance.val)
        slider2_filter.set_val(slider2_filter.val)

        n += 1
        signal1x, signal1y = filter_by_limit(x, data[:, n], minx1, maxx1)
        signal2x, signal2y = filter_by_limit(x, data[:, n], minx2, maxx2)
        signal3x, signal3y = filter_by_limit(x, scope[:, n], scope_min, scope_max)  # Scope


        print('')
        print(f'Trace n = {n}')
        print(f'B_C {oscillation_distance(scope_peaks, SAS_peaks[5], SAS_peaks[2]) * FC * 1e-6} Mhz')
        print(f'A_B {oscillation_distance(scope_peaks, SAS_peaks[2], SAS_peaks[0]) * FC * 1e-6} Mhz')
        print(f'E_F {oscillation_distance(scope_peaks, SAS_peaks[9], SAS_peaks[7]) * FC * 1e-6} Mhz')
        print(f'F_C {oscillation_distance(scope_peaks, SAS_peaks[9], SAS_peaks[2]) * FC * 1e-6} Mhz')
        print(f'A_E {oscillation_distance(scope_peaks, SAS_peaks[7], SAS_peaks[0]) * FC * 1e-6} Mhz')
        print(f'D_E {oscillation_distance(scope_peaks, SAS_peaks[7], SAS_peaks[6]) * FC * 1e-6} Mhz')

        B_C.append(oscillation_distance(scope_peaks, SAS_peaks[5], SAS_peaks[2]))
        A_B.append(oscillation_distance(scope_peaks, SAS_peaks[2], SAS_peaks[0]))
        E_F.append(oscillation_distance(scope_peaks, SAS_peaks[9], SAS_peaks[7]))
        F_C.append(oscillation_distance(scope_peaks, SAS_peaks[9], SAS_peaks[2]))
        A_E.append(oscillation_distance(scope_peaks, SAS_peaks[7], SAS_peaks[0]))
        D_E.append(oscillation_distance(scope_peaks, SAS_peaks[7], SAS_peaks[6]))

        slider1_prominence.set_val(slider1_prominence.val)
        slider1_distance.set_val(slider1_distance.val)
        slider1_filter.set_val(slider1_filter.val)

        slider2_prominence.set_val(slider2_prominence.val)
        slider2_distance.set_val(slider2_distance.val)
        slider2_filter.set_val(slider2_filter.val)
    else:
        print('error')

save_button.on_clicked(save_action)
save_button2.on_clicked(save_csv)

plt.show()
