import matplotlib.pyplot as plt
import numpy as np

def msd_straight_forward(r):
    shifts = np.arange(len(r))
    msds = np.zeros(shifts.size)

    for i, shift in enumerate(shifts):
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
t = np.arange(0,N)
simple_t = np.mean(t.reshape(-1,reduce_points), axis=1)
# row:molecule:x,y,z
data = np.load(filename)
num_rows, num_molecules, num_cols = data.shape

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


all_results /= num_molecules
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('tau [fs?]')
ax.set_ylabel('MSD')
ax.grid(linestyle='-', color='0.7', zorder=0)
ax.plot(simple_t, all_results, zorder=2)
fig.savefig("msd_fft_all_plot.png", dpi=288)


#Compare via running:

# %timeit results = msd_straight_forward(r)
# %timeit results_fft = msd_fft(r)


# results = msd_straight_forward(r)
# r.shape

# r = np.cumsum(np.random.choice([-1., 0., 1.], size=(N, 3)), axis=0)
# results = msd_fft(r)


# ax.set_xlim(0,t_end)
# ax.legend(['C_a'])
# ax.set_title('Concentration vs Time' + subheader)
# fig.savefig("conc-vs-time.png", dpi=288)
