'''Tugas Praktikum Modul 1
ME3202 Analisis Data Cuaca dan Iklim II
Analisis Harmonik Sederhana
Nama  : Nafal Shaquille Muhammad
NIM   : 12821039
'''

# Import Fungsi Harmonik dari File harmonic_M1.py
from harmonic_M1 import *

# Import Library dan Modul
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy as crt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Baca data presipitasi
pr = xr.open_dataarray('../ANDATII/Modul_1/gsmap.1hr.citarum.nc')

# Definisikan lokasi titik sampel di dua kota, dalam kasus ini, kota yang digunakan adalah Bekasi dan Bogor
bks_loc = [106.98, -6.27]
bgr_loc = [106.80, -6.60]

# Mengelompokkan data hujan perbulan kemudian dirata-ratakan, kemudian slice data untuk di dua titik sampel di dua kota

ts_bks = pr.groupby('time.month').mean().sel(lon=bks_loc[0],lat=bks_loc[1], method='nearest')
ts_bgr = pr.groupby('time.month').mean().sel(lon=bgr_loc[0],lat=bgr_loc[1], method='nearest')

# Menghitung fungsi harmonik
mu_bks = ts_bks.mean().values
mu_bgr = ts_bgr.mean().values
c1,theta1,yi1,k = fharmonic(ts_bks-mu_bks) 
c2,theta2,yi2,k = fharmonic(ts_bgr-mu_bgr) 

# Plot grafik fungsi harmonik
# Define waktu lokal (UTC + 7)
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

ts_bks.plot(figsize=(12,3),marker='+',label='Presipitasi') # Garis data presipitasi
plt.plot(yi1[:,0]+mu_bks,'+-',label='K=1') # Harmonik 1
plt.plot(yi1[:,1]+mu_bks,'+-',label='K=2') # Harmonik 2

plt.legend()
plt.xticks(np.linspace(0,11,12),months)
plt.xlabel('Hour (local time)')
plt.title('Plot Grafik Fungsi Harmonik Kota Bekasi')
plt.grid()
plt.savefig('../ANDATII/Modul_1/Tugas_Modul1/Harmonik Bekasi.png')

ts_bgr.plot(figsize=(12,3),marker='+',label='Presipitasi') # Garis data presipitasi
plt.plot(yi2[:,0]+mu_bks,'+-',label='K=1') # Harmonik 1
plt.plot(yi2[:,1]+mu_bks,'+-',label='K=2') # Harmonik 2

plt.legend()
plt.xticks(np.linspace(0,11,12),months)
plt.xlabel('Hour (local time)')
plt.title('Plot Grafik Fungsi Harmonik Kota Bogor')
plt.grid()
plt.savefig('../ANDATII/Modul_1/Tugas_Modul1/Harmonik Bogor.png')


# Perhitungan Normalized Amplitude
n = ts_bks.size
r1 = n/2*c1**2/((n-1)*np.var(ts_bks-mu_bks,ddof=1).values) # Bekasi
r2 = n/2*c2**2/((n-1)*np.var(ts_bgr-mu_bgr,ddof=1).values) # Bogor

# Plot Periodogram untuk melihat kontribusi masing-masing harmonik
plt.figure(figsize=(12,3))

plt.bar(k + 0.00, r1, label = 'Bekasi', width = 0.25)
plt.bar(k + 0.25, r2, label = 'Bogor', width = 0.25)

plt.xticks(np.linspace(1,12,12))
plt.title('Periodogram: k vs R^2')
plt.xlabel('Wave Number (k)')
plt.ylabel('Normalized Amplitude (R^2)')
plt.legend()
plt.grid()
plt.savefig('../ANDATII/Modul_1/Tugas_Modul1/Periodogram.png')

# Distribusi Spasial Amplitude
prMonth = pr.groupby('time.month').mean()

nt,ny,nx = prMonth.shape

# Inisialisasi array untuk menyimpan normalized amplitude di setiap titik
C = np.zeros((int(nt/2),ny,nx))

# Looping titik grid
for iy in range(ny):
    for ix in range(nx):
        # Time-Series
        ts=prMonth[:,iy,ix].values
        n=ts.size
        mu=ts.mean()
        # Harmonic Analysis
        c,th,yi,k=fharmonic(ts-mu)
        # Normalized Amplitude
        r=n/2*c**2/((n-1)*np.var(ts-mu,ddof=1))
        #
        C[:,iy,ix] = r

# Plot Distribusi Spasial Amplitude
lon = prMonth.lon.values
lat = prMonth.lat.values
# Annual (k = 1)
lvl = np.linspace(0.5,1,6)
plot_2d(C[0,:,:],lon,lat,levels=lvl,title='Normalized Amplitude Harmonik ke-1 (k=1)')

# Semi-Annual
lvl = np.linspace(0,0.5,6)
plot_2d(C[1,:,:],lon,lat,levels=lvl,title='Normalized Amplitude Harmonik ke-2 (k=2)')