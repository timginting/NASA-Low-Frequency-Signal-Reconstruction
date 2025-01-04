import os, sys
from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import h5py
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from inc.io import read_sds
from inc.gridding import *
from inc.plotting import *


# Loop though GLM data, extracting gridded products...

debug = False

#ddir = '/media/sciguymp/TRMM/data/trmm/level1/1Z09/'
ddir  = '../data/orbits/1Z09/'
lsdir = '../data/orbits/lis/trmmlis_sc2/'
lrdir = '../data/orbits/lis/trmmlis_sc2/'
pdir = "img/swaths/"
swdir = '../data/orbits/resampled/'

ofile = '../data/images/db_v5_cv2/{year}/trmm_view_db_v5_{date}.h5'

delta = 8
p = None

oc = 0

cd = '1998040'

debug = False

# Prepare subset parameters for resampling to regular grid...
# --------------------------------------------------------------------------------------------------

#Grid Definition
#nx = int(360/0.025)+1
#ny = int(80/0.025)+1
nx = int(360/0.020)+1
ny = int(80/0.020)+1
lons = np.repeat(np.expand_dims(np.linspace(-180,180,nx), axis=0), ny, axis=0).T
lats = np.repeat(np.expand_dims(np.linspace(-40,40,ny), axis=0),nx, axis=0)

# Loop through all available TRMM orbit files...
# --------------------------------------------------------------------------------------------------

for year in sorted(os.listdir(ddir)):
  if "." in year:
    continue
  for month in sorted(os.listdir(ddir+year)):
    if "." in month:
      continue
    for day in sorted(os.listdir(ddir+year+'/'+month)):
      if "." in day:
        continue

      ofname = ofile.replace('{date}',year+month+day).replace('{year}',year)
      
      if cd not in ofname:
        continue
     
      w = None
      ocnt = 0

      for orbit in sorted(os.listdir(ddir+year+'/'+month+'/'+day)):
        if "." in orbit or '02556' in orbit:
          continue

        sodir = swdir + year + '/' + year+month+day + '/'
        if not os.path.exists(sodir):
          os.makedirs(sodir)
        sofile = sodir + 'com_swath_'+year+month+day+'_'+orbit+'.h5'

        if os.path.exists(sofile):
          print(ofname, " exists, skipping...")
          continue


        print(datetime.now(), year, month, day, ocnt, orbit)
        ocnt += 1

        # Read in LIS data...
        # --------------------------------------------------------------------------------------------------

        # APEX data...
        
        lfile = lrdir + year + '/trmmlis_'+year+month+day+'_group.h5'
        print(lfile)
        if not os.path.exists(lfile):
          print(" -- LIS file not found!")
          continue
        
        f = h5py.File(lfile)

        okey = str(int(orbit))
        if okey not in f["grids"]:
          print(" -- LIS grid not found!")
          continue
        
        glon = f["grids"][okey]["lon"][:]
        glat = f["grids"][okey]["lat"][:]
        gfed = f["grids"][okey]["fed"][:]

        #print("APEX_FED")
        #fed_apx = regrid_swath(gfed, glon, glat, lons, lats, debug=debug, ewa=False)                                           

        # 0.02
        nx = int(360/0.02)+1
        ny = int(80/0.02)+1
        gfed = np.zeros((nx, ny))
        lx = (np.round((glon+180)/0.02)).astype('int')
        ly = (np.round((glat+40)/0.02)).astype('int')
        gfed[lx, ly] = f["grids"][okey]["fed"][:]
        glon = np.zeros((nx, ny))
        lx = np.linspace(-180,180,nx)
        for i in range(0, ny):
          glon[:,i] = lx
        glat = np.zeros((nx, ny))
        ly = np.linspace(-40,40,ny)
        for i in range(0, nx):
          glat[i,:] = ly
        fed_apx = gfed
        
        f.close()


        # LIS science data...
        
        lfile = lsdir + year + '/trmmlis_'+year+month+day+'_group.h5'
        if not os.path.exists(lfile):
          print(" -- LIS file not found!")
          continue
        
        f = h5py.File(lfile)

        okey = str(int(orbit))
        if okey not in f["grids"]:
          print(" -- LIS grid not found!")
          break
        
        glon = f["grids"][okey]["lon"][:]
        glat = f["grids"][okey]["lat"][:]
        gfed = f["grids"][okey]["fed"][:]

        #print("SC_FED")
        #fed_sc = regrid_swath(gfed, glon, glat, lons, lats, debug=debug, ewa=False)                                           

        print(datetime.now(), np.amin(glon), np.amax(glon), np.amin(glat), np.amax(glat))

        # 0.02
        nx = int(360/0.02)+1
        ny = int(80/0.02)+1
        gfed = np.zeros((nx, ny))
        lx = (np.round((glon+180)/0.02)).astype('int')
        ly = (np.round((glat+40)/0.02)).astype('int')
        gfed[lx, ly] = f["grids"][okey]["fed"][:]
        glon = np.zeros((nx, ny))
        lx = np.linspace(-180,180,nx)
        for i in range(0, ny):
          glon[:,i] = lx
        glat = np.zeros((nx, ny))
        ly = np.linspace(-40,40,ny)
        for i in range(0, nx):
          glat[i,:] = ly
        fed_sc = gfed

        print(datetime.now())
        
        # LIS viewtimes...
        
        #for key in f:
        #  print(key)
        #for key in f["l2file"]:
        #  print(key, np.shape(f["l2file"][key]), np.amin(f["l2file"][key][:]), np.mean(f["l2file"][key][:]), np.amax(f["l2file"][key][:]))
        #sys.exit()

        #print("LIS_VT")

        llon = np.squeeze(f["l2file"][okey]["viewtime_lon"][:])
        llat = np.squeeze(f["l2file"][okey]["viewtime_lat"][:])
        lisvt = np.squeeze(f["l2file"][okey]["viewtime_effective_obs"][:])

        print(datetime.now(), np.amin(llon), np.amax(llon), np.amin(llat), np.amax(llat))

        # 0.5
        nx = int(360/0.5)+1
        ny = int(180/0.5)+1
        lisvt = np.zeros((nx, ny))
        lx = (np.round((llon-0.25+180)/0.5)).astype('int')
        ly = (np.round((llat-0.25+90)/0.5)).astype('int')
        lisvt[lx, ly] = np.squeeze(f["l2file"][okey]["viewtime_effective_obs"][:])
        llon = np.zeros((nx, ny))
        lx = np.linspace(-180,180,nx)
        for i in range(0, ny):
          llon[:,i] = lx
        llat = np.zeros((nx, ny))
        ly = np.linspace(-90,90,ny)
        for i in range(0, nx):
          llat[i,:] = ly

        lis_vt = regrid_swath(lisvt, llon, llat, lons, lats, debug=debug, ewa=True)                                           

        print(datetime.now())

        f.close()
               
        # Read in TRMM data...
        # --------------------------------------------------------------------------------------------------

        droot = ddir+year+'/'+month+'/'+day+'/'+orbit+'/'

        f = read_sds(droot, debug=debug)


        # Prepare subset parameters for resampling to regular grid...
        # --------------------------------------------------------------------------------------------------

        #print("PRMAXSURFZ")
        tmi = f['TMI']['PCT85'][:]*0+1
        tmi[:,0] = 0
        tmi[:,-1] = 0
        tmi[0,:] = 0
        tmi[-1,:] = 0

        tmiswath = griddata((f['TMI']['LONHI'][:].flatten(),
                             f['TMI']['LATHI'][:].flatten()),tmi.flatten(),(lons,lats),method='linear')
   
        idtmi = np.where(tmiswath > 0.99)
        tmiswath[:] = 0
        tmiswath[idtmi] = 1 
        idntmi = np.where(tmiswath < 0.99)  

        pr = f['PR']['2A25']['NEARSURFZ'][:]*0+1
        pr[:,0] = 0
        pr[:,-1] = 0
        pr[0,:] = 0
        pr[-1,:] = 0

        prswath = griddata((f['PR']['2A25']['LON'][:].flatten(),
                            f['PR']['2A25']['LAT'][:].flatten()),pr.flatten(),(lons,lats),method='linear')

        idpr = np.where(prswath > 0.99)
        prswath[:] = 0
        prswath[idpr] = 1 
        idnpr = np.where(prswath < 0.99)  

        #print(np.shape(prswath))
        #print(np.shape(idtmi))
        #print(np.shape(fed_apx[idtmi]))
        #sys.exit()

        #fed_apx = fed_apx/lis_vt
        #fed_sc = fed_sc/lis_vt

        if not os.path.exists(sodir):
          os.makedirs(sodir)
        so = h5py.File(sofile, "w")
        
        #nx = int(360/0.020)+1
        #ny = int(80/0.020)+1
        #lons = np.repeat(np.expand_dims(np.linspace(-180,180,nx), axis=0), ny, axis=0).T
        #lats = np.repeat(np.expand_dims(np.linspace(-40,40,ny), axis=0),nx, axis=0)

        so.create_dataset("grid_extent_degrees", data=np.array([-180, 180, -40, 40]))
        so.create_dataset("grid_resolution_degrees", data=np.array([0.020, 0.020]))
        so.create_dataset("grid_image_size_pixels", data=np.array([nx, ny]))
        so.create_dataset("orbit_mask", data=np.array(idtmi), compression="gzip", compression_opts=9)
        so.create_dataset("lon", data=lons[idtmi], compression="gzip", compression_opts=9)
        so.create_dataset("lat", data=lats[idtmi], compression="gzip", compression_opts=9)
        so.create_dataset("lis_fed", data=fed_apx[idtmi], compression="gzip", compression_opts=9)
        #so.create_dataset("lis_fed_sc", data=fed_sc[idtmi])
        so.create_dataset("lis_vt", data=lis_vt[idtmi], compression="gzip", compression_opts=9)
        
        del fed_sc, fed_apx, lis_vt


        #print("PCT10")
        h10 = regrid_swath(f['TMI']['H10'][:], 
                             f['TMI']['LONLO'][:],
                             f['TMI']['LATLO'][:], lons, lats, debug=debug)                                           
        v10 = regrid_swath(f['TMI']['V10'][:], 
                             f['TMI']['LONLO'][:],
                             f['TMI']['LATLO'][:], lons, lats, debug=debug)                                           
        pct10 = (1+1.48)*v10-1.48*h10
        pct10[idntmi] = 0

        so.create_dataset("tmi_pct10", data=pct10[idtmi], compression="gzip", compression_opts=9)
        del v10, h10, pct10

        #print("PCT19")
        h19 = regrid_swath(f['TMI']['H19'][:], 
                             f['TMI']['LONLO'][:],
                             f['TMI']['LATLO'][:], lons, lats, debug=debug)                                           
        v19 = regrid_swath(f['TMI']['V19'][:], 
                             f['TMI']['LONLO'][:],
                             f['TMI']['LATLO'][:], lons, lats, debug=debug)                                           
        pct19 = (1+1.36)*v19-1.36*h19
        pct19[idntmi] = 0

        so.create_dataset("tmi_pct19", data=pct19[idtmi], compression="gzip", compression_opts=9)
        del v19, h19, pct19

        #print("PCT37")
        pct37 = regrid_swath(f['TMI']['PCT37'][:], 
                             f['TMI']['LONLO'][:],
                             f['TMI']['LATLO'][:], lons, lats, debug=debug)                                           
        pct37[idntmi] = 0

        so.create_dataset("tmi_pct37", data=pct37[idtmi], compression="gzip", compression_opts=9)
        del pct37

        #print("PCT85")
        pct85 = regrid_swath(f['TMI']['PCT85'][:], 
                             f['TMI']['LONHI'][:],
                             f['TMI']['LATHI'][:], lons, lats, debug=debug)                                           
        pct85[idntmi] = 0

        so.create_dataset("tmi_pct85", data=pct85[idtmi], compression="gzip", compression_opts=9)
        del pct85

        #print("VIRSCH4")
        irch4 = regrid_swath(f['VIRS']['CH4_ALL'][:], 
                             f['VIRS']['LON'][:],
                             f['VIRS']['LAT'][:], lons, lats, debug=debug)                                           
        irch4[idntmi] = 0
        so.create_dataset("virs_irch4", data=irch4[idtmi], compression="gzip", compression_opts=9)
        del irch4

        #print("RAIN")
        tmirain = regrid_swath(f['TMI']['RAIN'][:], 
                              f['TMI']['LONHI'][:],
                              f['TMI']['LATHI'][:], lons, lats, debug=debug)                                           
        tmirain[idntmi] = 0

        so.create_dataset("tmi_rain", data=tmirain[idtmi], compression="gzip", compression_opts=9)
        del tmirain

        """
        fig, axs = plt.subplots(6,2,figsize=(15,30))

        print(np.shape(idpr))
        print(np.shape(lons))
        print(np.shape(lats))
        print(np.shape(tmirain))
        print(np.shape(fed_sc))
        print(np.nanmax(fed_sc))

        #im = np.nanargmax(tmirain[idpr])
        #im = np.nanargmax(fed_sc[idpr])
        #idprs = np.where(prswath == 1)
        im = np.nanargmax(fed_sc[idpr])
        clon = lons[idpr][im]
        clat = lats[idpr][im]
        cval = fed_sc[idpr][im]

        print(clon, clat, cval)

        cdel = 1.5
        cdel = 5
        xrange = [clon - cdel, clon + cdel]
        yrange = [clat - cdel, clat + cdel]
        idg = np.where((lons >= xrange[0]) & 
                       (lons <= xrange[1]) & 
                       (lats >= yrange[0]) & 
                       (lats <= yrange[1]))

        print(np.nanmax(tmirain[idpr]), tmirain[idpr][im])
        print(np.nanmax(fed_sc[idpr]), fed_sc[idpr][im])
        print(xrange)
        print(yrange)

        axs[0,0].set_title("TMI Swath")
        axs[0,1].set_title("PR Swath")
        #axs[0,0].pcolormesh(f['PR']['2A25']['LON'][:], f['PR']['2A25']['LAT'][:], f['PR']['2A25']['NEARSURFZ'][:])
        #id = np.where(np.isnan(irch4))
        #irch4[id] = 0
        axs[0,0].pcolormesh(lons, lats, tmiswath)
        axs[0,1].pcolormesh(lons, lats, prswath)

        axs[1,0].set_title("VIRS CH4 Swath")
        axs[1,1].set_title("VIRS CH4 Resampled")
        id = np.where(np.isnan(irch4))
        irch4[id] = 0
        vmin = np.amin(irch4[idg])
        vmax = np.amax(irch4[idg])
        axs[1,0].pcolormesh(f['VIRS']['LON'][:], f['VIRS']['LAT'][:], f['VIRS']['CH4_ALL'][:], vmin=vmin, vmax=vmax)
        axs[1,1].pcolormesh(lons, lats, irch4, vmin=vmin, vmax=vmax)

        axs[2,0].set_title("TMI PCT85 Swath")
        axs[2,1].set_title("TMI PCT85 Resampled")
        id = np.where(np.isnan(pct85))
        pct85[id] = 0
        vmin = np.amin(pct85[idg])
        vmax = np.amax(pct85[idg])
        axs[2,0].pcolormesh(f['TMI']['LONHI'][:], f['TMI']['LATHI'][:], f['TMI']['PCT85'][:], vmin=vmin, vmax=vmax)
        axs[2,1].pcolormesh(lons, lats, pct85, vmin=vmin, vmax=vmax)

        axs[3,0].set_title("TMI Rain Swath")
        axs[3,1].set_title("TMI Rain Resampled")
        id = np.where(np.isnan(tmirain))
        tmirain[id] = 0
        vmin = np.amin(tmirain[idg])
        vmax = np.amax(tmirain[idg])
        axs[3,0].pcolormesh(f['TMI']['LONHI'][:], f['TMI']['LATHI'][:], f['TMI']['RAIN'][:], vmin=vmin, vmax=vmax)
        axs[3,1].pcolormesh(lons, lats, tmirain, vmin=vmin, vmax=vmax)

        print(np.shape(glon), np.shape(glat), np.shape(gfed))

        axs[4,0].set_title("LIS FED Swath")
        axs[4,1].set_title("LIS FED Resampled")
        id = np.where(np.isnan(fed_sc))
        fed_sc[id] = 0
        vmin = np.amin(fed_sc[idg])
        vmax = np.amax(fed_sc[idg])
        #axs[4,0].pcolormesh(glon, glat, gfed, vmin=vmin, vmax=vmax)
        axs[4,0].pcolormesh(lons, lats, fed_sc, vmin=vmin, vmax=vmax)
        axs[4,1].pcolormesh(lons, lats, fed_sc, vmin=vmin, vmax=vmax)

        print(np.shape(llon), np.shape(llat), np.shape(lisvt))

        axs[5,0].set_title("LIS VT Swath")
        axs[5,1].set_title("LIS VT Resampled")
        id = np.where(np.isnan(lis_vt))
        lis_vt[id] = 0
        vmin = np.amin(lis_vt[idg])
        vmax = np.amax(lis_vt[idg])
        axs[5,0].pcolormesh(llon, llat, lisvt, vmin=vmin, vmax=vmax)
        axs[5,1].pcolormesh(lons, lats, lis_vt, vmin=vmin, vmax=vmax)



        for ax in axs.flat:
          #ax.set_xlim([-105, -85])
          #ax.set_ylim([-10, -5])

          ax.set_xlim(xrange)
          ax.set_ylim(yrange)

        plt.tight_layout()

        if not(os.path.exists(pdir)):
          os.makedirs(pdir)

        plt.savefig(pdir+"swath_"+year+month+day+"_"+okey+".png", dpi=300)
        print("Saved!")
        #sys.exit()
        """


        #print("PRECIPWATER")
        tmipw = regrid_swath(f['TMI']['PRECIPWATER'][:], 
                             f['TMI']['LONHI'][:],
                             f['TMI']['LATHI'][:], lons, lats, debug=debug)                                           
        tmipw[idntmi] = 0

        so.create_dataset("tmi_pw", data=tmipw[idtmi], compression="gzip", compression_opts=9)
        del tmipw

        #print("ICEWATERPATH")
        tmiiwp = regrid_swath(f['TMI']['ICEWPATH'][:], 
                              f['TMI']['LONHI'][:],
                              f['TMI']['LATHI'][:], lons, lats, debug=debug)                                           
        tmiiwp[idntmi] = 0

        so.create_dataset("tmi_iwp", data=tmiiwp[idtmi], compression="gzip", compression_opts=9)
        del tmiiwp

        #print("CLDWPATH")
        tmiclp = regrid_swath(f['TMI']['CLDWPATH'][:], 
                              f['TMI']['LONHI'][:],
                              f['TMI']['LATHI'][:], lons, lats, debug=debug)                                           
        tmiclp[idntmi] = 0

        so.create_dataset("tmi_clp", data=tmiclp[idtmi], compression="gzip", compression_opts=9)
        del tmiclp

        #print("RAINWPATH")
        tmirwp = regrid_swath(f['TMI']['RAINWPATH'][:], 
                              f['TMI']['LONHI'][:],
                              f['TMI']['LATHI'][:], lons, lats, debug=debug)                                           
        tmirwp[idntmi] = 0

        so.create_dataset("tmi_rwp", data=tmirwp[idtmi], compression="gzip", compression_opts=9)
        del tmirwp

        #print("WINDSPEED")
        tmiwind = regrid_swath(f['TMI']['WINDSPEED'][:], 
                              f['TMI']['LONHI'][:],
                              f['TMI']['LATHI'][:], lons, lats, debug=debug)                                           
        tmiwind[idntmi] = 0

        so.create_dataset("tmi_wind", data=tmiwind[idtmi], compression="gzip", compression_opts=9)
        del tmiwind

        #print("PRMAXSURFZ")
        prnsz = regrid_swath(f['PR']['2A25']['NEARSURFZ'][:], 
                             f['PR']['2A25']['LON'][:],
                             f['PR']['2A25']['LAT'][:], lons, lats, debug=debug)                                           
        prnsz[idnpr] = 0

        so.create_dataset("pr_nsz", data=prnsz[idtmi], compression="gzip", compression_opts=9)
        del prnsz

        #print("RAINTYPE2A23")
        prraintyp = regrid_swath(f['PR']['2A23']['RAINTYPE2A23'][:], 
                             f['PR']['2A25']['LON'][:],
                             f['PR']['2A25']['LAT'][:], lons, lats, debug=debug)                                           
        prraintyp[idnpr] = 0

        so.create_dataset("pr_raintype", data=prraintyp[idtmi], compression="gzip", compression_opts=9)
        del prraintyp

        #print("STORMH")
        prstormh = regrid_swath(f['PR']['2A23']['STORMH'][:], 
                             f['PR']['2A25']['LON'][:],
                             f['PR']['2A25']['LAT'][:], lons, lats, debug=debug)                                           
        prstormh[idnpr] = 0

        so.create_dataset("pr_stormh", data=prstormh[idtmi], compression="gzip", compression_opts=9)
        del prstormh

        #print("RAIN_2B31")
        prrain = regrid_swath(f['PR']['2A25']['RAIN_2B31'][:], 
                             f['PR']['2A25']['LON'][:],
                             f['PR']['2A25']['LAT'][:], lons, lats, debug=debug)                                           
        prrain[idnpr] = 0

        so.create_dataset("pr_rain", data=prrain[idtmi], compression="gzip", compression_opts=9)
        del prrain

        #print("PRECIP_2B31")
        prprecip = regrid_swath(f['PR']['2A25']['PRECIP_2B31'][:], 
                             f['PR']['2A25']['LON'][:],
                             f['PR']['2A25']['LAT'][:], lons, lats, debug=debug)                                           
        prprecip[idnpr] = 0

        so.create_dataset("pr_precip", data=prprecip[idtmi], compression="gzip", compression_opts=9)
        del prprecip

        so.close()



