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
    for v in f:
        m = [v[0](4000000) for i in range(10)]
        d = [numpy.mean(m, axis=0), numpy.std(m, axis=0)]
        print("Bandwidth for " + v[1] + ":")
        print("Copy:  {:16.10f} MB/s ± {:16.10f} MB/s".format(d[0][0], d[1][0]))
        print("Scale: {:16.10f} MB/s ± {:16.10f} MB/s".format(d[0][1], d[1][1]))
        print("Sum:   {:16.10f} MB/s ± {:16.10f} MB/s".format(d[0][2], d[1][2]))
        print("Triad: {:16.10f} MB/s ± {:16.10f} MB/s".format(d[0][3], d[1][3]))
