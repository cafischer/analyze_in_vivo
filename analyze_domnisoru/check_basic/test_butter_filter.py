import numpy as np
import matplotlib.pyplot as pl
from spiketimes import butter_bandpass_filter, butter_highpass_filter, butter_lowpass_filter
from cell_characteristics import to_idx
from scipy.signal import resample
from grid_cell_stimuli import compute_fft


if __name__ == '__main__':
    dt = 0.1
    baseline = -70
    peak = 10
    width = 100 # TODO 1
    len_trace = 100 # TODO 4000
    middle = int(round(len_trace / 2.))
    v = np.ones(to_idx(len_trace, dt)) * baseline
    t = np.arange(len(v)) * dt

    # TODO
    # v = np.zeros(to_idx(len_trace, dt))
    # v[to_idx(middle-width, dt):to_idx(middle+width, dt)] = baseline

    # v[to_idx(middle-width, dt):to_idx(middle, dt)] = np.linspace(baseline, peak,
    #                                                              to_idx(middle, dt) - to_idx(middle-width, dt))
    # v[to_idx(middle, dt):to_idx(middle+width, dt)] = np.linspace(peak, baseline,
    #                                                              to_idx(middle+width, dt) - to_idx(middle, dt) +1)[1:]


    v_filtered = butter_highpass_filter(v, 10, 1. / dt * 1000., order=5)

    #v_down, t_down = resample(v, 10000, t)
    v_down = v
    t_down = t
    dt_down = t_down[1] - t_down[0]
    print dt_down

    pl.figure()
    pl.plot(t_down, v_down)

    v_fft, freqs = compute_fft(v_down, dt_down/1000.)
    v_fft = np.abs(v_fft)**2
    pl.figure()
    pl.plot(freqs, v_fft)
    pl.xlim(0, None)
    #pl.ylim(0, 10000)

    # v_filtered_down, t_filtered_down = resample(v_filtered, 1000, t)
    v_filtered_down = v_filtered
    t_filtered_down = t

    pl.figure()
    pl.plot(t_filtered_down, v_filtered_down)

    v_fft, freqs = compute_fft(v_filtered_down, (t_filtered_down[1]-t_filtered_down[0]) / 1000.)
    v_fft = np.abs(v_fft) ** 2
    pl.figure()
    pl.plot(freqs, v_fft)
    pl.xlim(0, None)
    #pl.ylim(0, 10000)

    pl.show()
