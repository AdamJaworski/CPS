import matplotlib.pyplot as plot
from math import sin, pi, cos


def add_to_plot(amplitude: int, freq: int, fs: float, time: float, style: str) -> None:
    global plt
    ns = fs * time
    dt = 1 / fs
    n = range(0, int(ns - 1))
    t = [dt * x for x in n]
    x = [amplitude * sin(2  * pi * freq * t_) for t_ in t]
    plot.plot(t, x, style)

