import matplotlib.pyplot as plt
import array
import numpy
import timeit

def MeasureBandwidthList(n):
    a = [1.0] * n
    b = [2.0] * n
    c = [0.0] * n
    t0 = timeit.default_timer()
    for i in range(n):
        c[i] = a[i]
    t1 = timeit.default_timer()
    for i in range(n):
        b[i] = 2.0 * c[i]
    t2 = timeit.default_timer()
    for i in range(n):
        c[i] = a[i] + b[i]
    t3 = timeit.default_timer()
    for i in range(n):
        a[i] = b[i] + 2.0 * c[i]
    t4 = timeit.default_timer()
    r0 = ((2 * 8 * n) / (t1 - t0)) * 1e-6
    r1 = ((3 * 8 * n) / (t2 - t1)) * 1e-6
    r2 = ((2 * 8 * n) / (t3 - t2)) * 1e-6
    r3 = ((3 * 8 * n) / (t4 - t3)) * 1e-6
    return [r0, r1, r2, r3]

def MeasureBandwidthArray(n):
    a = array.array("f", [1.0] * n)
    b = array.array("f", [2.0] * n)
    c = array.array("f", [0.0] * n)
    t0 = timeit.default_timer()
    for i in range(n):
        c[i] = a[i]
    t1 = timeit.default_timer()
    for i in range(n):
        b[i] = 2.0 * c[i]
    t2 = timeit.default_timer()
    for i in range(n):
        c[i] = a[i] + b[i]
    t3 = timeit.default_timer()
    for i in range(n):
        a[i] = b[i] + 2.0 * c[i]
    t4 = timeit.default_timer()
    r0 = ((2 * 4 * n) / (t1 - t0)) * 1e-6
    r1 = ((3 * 4 * n) / (t2 - t1)) * 1e-6
    r2 = ((2 * 4 * n) / (t3 - t2)) * 1e-6
    r3 = ((3 * 4 * n) / (t4 - t3)) * 1e-6
    return [r0, r1, r2, r3]

def MeasureBandwidthNumpy(n):
    a = numpy.full(n, 1.0, dtype="float32")
    b = numpy.full(n, 2.0, dtype="float32")
    c = numpy.full(n, 0.0, dtype="float32")
    t0 = timeit.default_timer()
    numpy.copyto(c, a)
    t1 = timeit.default_timer()
    numpy.multiply(2.0, c, b)
    t2 = timeit.default_timer()
    numpy.add(a, b, c)
    t3 = timeit.default_timer()
    numpy.multiply(2.0, c, a)
    numpy.add(a, b, a)
    t4 = timeit.default_timer()
    r0 = ((2 * 4 * n) / (t1 - t0)) * 1e-6
    r1 = ((3 * 4 * n) / (t2 - t1)) * 1e-6
    r2 = ((2 * 4 * n) / (t3 - t2)) * 1e-6
    r3 = ((3 * 4 * n) / (t4 - t3)) * 1e-6
    return [r0, r1, r2, r3]

if __name__ == "__main__":
    f = []
    f.append((MeasureBandwidthList,  "List"))
    f.append((MeasureBandwidthArray, "Array"))
    f.append((MeasureBandwidthNumpy, "Numpy"))
    fig, ax = plt.subplots(4, figsize=(16, 9), constrained_layout=True, sharex=True, sharey=True)
    fig.supxlabel("Elements")
    fig.supylabel("Bandwidth (MB/s)")
    xticks = range(0, 4000000 + 1, 100000)
    yticks = range(0,   80000 + 1,  20000)
    for v in f:
        y = [v[0](n) for n in xticks]
        ax[0].plot(xticks, [b[0] for b in y], label="Copy "  + v[1])
        ax[1].plot(xticks, [b[1] for b in y], label="Scale " + v[1])
        ax[2].plot(xticks, [b[2] for b in y], label="Sum "   + v[1])
        ax[3].plot(xticks, [b[3] for b in y], label="Triad " + v[1])
    for v in ax:
        v.grid()
        v.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.xlim([xticks[0], xticks[-1]])
    plt.ylim([yticks[0], yticks[-1]])
    plt.xticks(xticks[0:  :2], xticks[0:  :2])
    plt.yticks(yticks[1:-1:1], yticks[1:-1:1])
    plt.show()
