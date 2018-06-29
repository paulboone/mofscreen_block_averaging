#Compare via running:
# %timeit results = msd_straight_forward(r)
# %timeit results_fft = msd_fft(r)

# results = msd_straight_forward(r)
# r.shape

# r = np.cumsum(np.random.choice([-1., 0., 1.], size=(N, 3)), axis=0)
# results = msd_fft(r)

import matplotlib.pyplot as plt
import numpy as np

def msd_straight_forward(r):
    shifts = np.arange(len(r))
    msds = np.zeros(shifts.size)

    for i, shift in enumerate(shifts):
        # gets us all differences of tau = shift
        diffs = r[:-shift if shift else None] - r[shift:]
        sqdist = np.square(diffs).sum(axis=1)
        msds[i] = sqdist.mean()

    return msds

def autocorrFFT(x):
    N=len(x)
    F = np.fft.fft(x, n=2*N)  #2*N because of zero-padding
    PSD = F * F.conjugate()
    res = np.fft.ifft(PSD)
    res= (res[:N]).real   #now we have the autocorrelation in convention B
    n=N*np.ones(N)-np.arange(0,N) #divide res(m) by (N-m)
    return res/n #this is the autocorrelation in convention A


def msd_fft(r):
    print("msd_fft")
    N=len(r)
    D=np.square(r).sum(axis=1)
    D=np.append(D,0)
    # print("prior to autocorrFFT")
    S2=sum([autocorrFFT(r[:, i]) for i in range(r.shape[1])])
    Q=2*D.sum()
    S1=np.zeros(N)
    # print('start')
    for m in range(N):
        Q=Q-D[m-1]-D[N-m]
        S1[m]=Q/(N-m)
        # if m % 10000 == 0:
        #     print(m)
    return S1-2*S2


N = 4000000
reduce_points = 4000
filename = "lammpstrj.npy"
fs_per_row = 10

t = np.arange(0,N)
simple_t = np.mean(t.reshape(-1,reduce_points), axis=1) * fs_per_row / 1e6
data = np.load(filename) # row:molecule:x,y,z
num_rows, num_molecules, num_cols = data.shape
# num_molecules = 10

# per molecule plots
fig = plt.figure(figsize=(7,7*num_molecules))
all_results = np.zeros(int(N / reduce_points))
for m in range(num_molecules):
    d0 = data[:,m,:][:N,:] # m for mth molecule
    results = msd_fft(d0)

    simple_results = np.mean(results.reshape(-1, reduce_points), axis=1)
    all_results += simple_results
    ax = fig.add_subplot(num_molecules, 1, m + 1)
    ax.set_xlabel('tau [fs?]')
    ax.set_ylabel('MSD')
    ax.grid(linestyle='-', color='0.7', zorder=0)
    ax.plot(simple_t, simple_results, zorder=2)

fig.savefig("msd_fft_molecule_plots.png", dpi=288)
all_results /= (6*num_molecules)



# attempt fits across different ranges
# generally for all fits, first 10% and last 50% are thrown away
# different ranges from 0.1-0.5 are tried, and fit with lowest error is selected as the
# "correct" fit and reported as the diffusivity
lin_fit_pairs = [(0.0,1.0), (0.10,0.50), (0.10,0.45), (0.10,0.40), (0.10,0.35), (0.10,0.30)]
fit_results = []
lowest_error = None
lowest_error_pair = None
for pair in lin_fit_pairs:
    # y = at + b
    len(simple_t)
    p1 = int(pair[0]* N / reduce_points)
    p2 = int(pair[1]* N / reduce_points)
    poly, residuals, rank, _, _ = np.polyfit(simple_t[p1:p2], all_results[p1:p2],1, full=True,)
    error = residuals / (p2 - p1)

    # pick best fit
    if not lowest_error or error < lowest_error:
        lowest_error = error
        lowest_error_pair = (p1,p2)
    fit_results.append([(p1,p2), error, poly])

# plot combined data and fits
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('tau [ns]')
ax.set_ylabel('MSD [Ang^2]')
ax.grid(linestyle='-', color='0.7', zorder=0)
ax.plot(simple_t, all_results, zorder=10)

for r in fit_results:
    p, error, poly = r
    zorder = 2
    if p == lowest_error_pair:
        print("Best fit: (%.2f-%.2f; %.2E):" % (*p, error))
        print("D = %2.2f angstrom^2 / ns" % poly[0])
        print("D = %2.3E cm^2 / s" % (poly[0] * 1e-16/1e-9))
        zorder = 20

    ax.plot(simple_t[p[0]:p[1]], np.polyval(poly, simple_t[p[0]:p[1]]), zorder=zorder,
            label="(%.2f-%.2f; %.2E) %2.0ft + %2.0f" % (*p, error, *poly))

ax.legend()
fig.savefig("msd_fft_all_plot.png", dpi=288)
