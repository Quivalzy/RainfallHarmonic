def fharmonic(Y):
    """
    Fungsi Analisa Harmonik dengan menggunakan MLR
    Input: timeseries yang telah dibuang Mean-nya
    Output: C (amplitude), theta (sudut fasa), dan Yi (sinyal harmonik) dari masing-masing k (bilangan gelombang)
    """

    import numpy as np
    from sklearn import linear_model

    # Persamaan C(k)*cos(2*pi*k*t/n - theta(k)) dapat dijabarkan menjadi
    # Persamaan: A(k)*cos(2pi*kt/n) + B(k)*sin(2pi*kt/n)
    # Parameter A dan B dicari dengan memecahkan multiple linear regression
    # Dari persamaan y = A(1)*x1 + B(1)*x2 + A(2)*x3 + B(2)*x4 + .... dst
    # dengan x1=cos(2pi*1*t/n)    ,k=1
    #       x2=sin(2pi*1*t/n)    ,k=1
    #       x3=cos(2pi*2*t/n)    ,k=2
    #       x4=sin(2pi*2*t/n)    ,k=2
    #       x5=sin(2pi*3*t/n)    ,k=3
    #       x6=sin(2pi*3*t/n)    ,k=3
    #       ... dst
    # A dan B selanjutnya digunakan untuk mengestimasi nilai C dan THETA

    n=Y.size
    t=np.linspace(0, n-1, n)
    X=np.empty([0,n])
    for k in range(1,int(np.fix(n/2))+1):
        x_o=np.cos(2*np.pi*k*t/n) # Untuk kasus ganjil
        x_e=np.sin(2*np.pi*k*t/n) # Untuk kasus genap
        X=np.vstack((X,x_o,x_e))

    # Melakukan transpose pada matriks X
    X=X.transpose()

    # Melakukan fitting ke fungsi MLR
    reg=linear_model.LinearRegression().fit(X,Y)

    # Menghitung koefisien regresi
    b=reg.coef_
    A=b[0:n:2] #ganjil
    B=b[1:n:2] #genap

    # Menghitung nilai C dan theta
    C=np.sqrt((A**2)+(B**2))
    theta=t[0:int(np.fix(n/2))]*np.nan
    theta[A>0.0]=np.arctan(B[A>0.0]/A[A>0.0])
    theta[A<0.0]=np.arctan(B[A<0.0]/A[A<0.0])+np.pi
    theta[A==0.0]=np.pi/2

    # Evaluasi persamaan harmoniknya
    yi=np.empty([int(np.fix(n/2)),n])
    yi=np.matrix.transpose(yi)
    for k in range(0,int(np.fix(n/2))):
        yi[:,k-1]=(C[k-1]*np.cos((2*np.pi*k*t/n) - (theta[k-1])));
    k=np.linspace(1,int(np.fix(n/2)),int(np.fix(n/2)))
    return C,theta,yi,k

def plot_2d(da,lon,lat,levels,title,cmap='rainbow',figsize=(10,10)):
  import numpy as np
  import matplotlib.pyplot as plt
  import warnings
  warnings.filterwarnings("ignore")
  import cartopy.crs as ccrs
  import cartopy.io.img_tiles as cimgt
  tiler = cimgt.GoogleTiles()
  zoom = 9

  fig=plt.figure(figsize=figsize)
  ax= plt.axes(projection=ccrs.PlateCarree())
  ax.set_extent((lon[0],lon[-1],lat[0],lat[-1]))
  ax.add_image(tiler, zoom )
  p=ax.contourf(lon,lat,da,
              transform=ccrs.PlateCarree(),
              cmap=cmap,
              levels=levels,
              alpha=0.7)

  ax.set_title(title)
  ax.coastlines(color='black',linewidth=3,alpha=0.5)
  ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
  plt.colorbar(p,orientation='vertical')
  plt.savefig(title)