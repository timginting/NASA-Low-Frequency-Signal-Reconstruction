import os, sys
from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import h5py
from netCDF4 import Dataset
import matplotlib.pyplot as plt

from inc.io import read_sds
from inc.gridding import *
from inc.plotting import *


# Loop though GLM data, extracting gridded products...

debug = True
#ddir = '/media/sciguymp/TRMM/data/trmm/level1/1Z09/'
ddir = '../data/trmm/'
ofile = '../data/input/db_v5/{year}/trmm_view_db_v5_{date}.h5'

delta = 8
p = None

oc = 0

yy = '1998'

#debug = False

for year in sorted(os.listdir(ddir)):
  if "." in year or year != yy:
    continue
  for month in sorted(os.listdir(ddir+year)):
    if "." in month:
      continue
    for day in sorted(os.listdir(ddir+year+'/'+month)):
      if "." in day:
        continue
      
      w = None
      ocnt = 0

      for orbit in sorted(os.listdir(ddir+year+'/'+month+'/'+day)):
        if "." in orbit or '02556' in orbit:
          continue

        print(datetime.now(), year, month, day, ocnt, orbit)
        ocnt += 1

        # Prepare subset parameters for resampling to regular grid...
        # --------------------------------------------------------------------------------------------------

        #Grid Definition
        nx = int(360/0.025)+1
        ny = int(80/0.025)+1
        lons = np.repeat(np.expand_dims(np.linspace(-180,180,nx), axis=0), ny, axis=0).T
        lats = np.repeat(np.expand_dims(np.linspace(-40,40,ny), axis=0),nx, axis=0)


        # Read in LIS data...
        # --------------------------------------------------------------------------------------------------

        # APEX data...
        
        #path = '../../lisotd/data/trmmlis/'
        path = '../data/trmmlis_sc2/'

        lfile = path + year + '/trmmlis_'+year+month+day+'_group.h5'
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
        f.close()

        #print("APEX_FED")
        fed_apx = regrid_swath(gfed, glon, glat, lons, lats, debug=debug)                                           
        

        # LIS science data...
        
        #path = '../../lisotd/data/trmmlis_sc/'
        path = '../data/trmmlis_sc2/'

        lfile = path + year + '/trmmlis_'+year+month+day+'_group.h5'
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
        fed_sc = regrid_swath(gfed, glon, glat, lons, lats, debug=debug)                                           
        
        # LIS viewtimes...
        
        #for key in f:
        #  print(key)
        #for key in f["l2file"]:
        #  print(key, np.shape(f["l2file"][key]), np.amin(f["l2file"][key][:]), np.mean(f["l2file"][key][:]), np.amax(f["l2file"][key][:]))
        #sys.exit()

        #print("LIS_VT")
        lis_vt = regrid_swath(np.squeeze(f["l2file"][okey]["viewtime_effective_obs"][:]),
                              np.squeeze(f["l2file"][okey]["viewtime_lon"][:]), 
                              np.squeeze(f["l2file"][okey]["viewtime_lat"][:]), 
                              lons, lats, debug=debug)                                           
        f.close()
        
        #fed_apx = fed_apx/lis_vt
        #fed_sc = fed_sc/lis_vt


        droot = ddir+year+'/'+month+'/'+day+'/'+orbit+'/'
                
        # Read in TRMM data...
        # --------------------------------------------------------------------------------------------------

        f = read_sds(droot, debug=debug)


        # Prepare subset parameters for resampling to regular grid...
        # --------------------------------------------------------------------------------------------------

        #print("PCT10")
        h10 = regrid_swath(f['TMI']['H10'][:], 
                             f['TMI']['LONLO'][:],
                             f['TMI']['LATLO'][:], lons, lats, debug=debug)                                           
        v10 = regrid_swath(f['TMI']['V10'][:], 
                             f['TMI']['LONLO'][:],
                             f['TMI']['LATLO'][:], lons, lats, debug=debug)                                           
        pct10 = (1+1.48)*v10-1.48*h10

        #print("PCT19")
        h19 = regrid_swath(f['TMI']['H19'][:], 
                             f['TMI']['LONLO'][:],
                             f['TMI']['LATLO'][:], lons, lats, debug=debug)                                           
        v19 = regrid_swath(f['TMI']['V19'][:], 
                             f['TMI']['LONLO'][:],
                             f['TMI']['LATLO'][:], lons, lats, debug=debug)                                           
        pct19 = (1+1.36)*v19-1.36*h19

        #print("PCT37")
        pct37 = regrid_swath(f['TMI']['PCT37'][:], 
                             f['TMI']['LONLO'][:],
                             f['TMI']['LATLO'][:], lons, lats, debug=debug)                                           

        #print("PCT85")
        pct85 = regrid_swath(f['TMI']['PCT85'][:], 
                             f['TMI']['LONHI'][:],
                             f['TMI']['LATHI'][:], lons, lats, debug=debug)                                           

        #print("VIRSCH4")
        irch4 = regrid_swath(f['VIRS']['CH4_ALL'][:], 
                             f['VIRS']['LON'][:],
                             f['VIRS']['LAT'][:], lons, lats, debug=debug)                                           

        #print("PRECIPWATER")
        tmipw = regrid_swath(f['TMI']['PRECIPWATER'][:], 
                             f['TMI']['LONHI'][:],
                             f['TMI']['LATHI'][:], lons, lats, debug=debug)                                           

        #print("ICEWATERPATH")
        tmiiwp = regrid_swath(f['TMI']['ICEWPATH'][:], 
                              f['TMI']['LONHI'][:],
                              f['TMI']['LATHI'][:], lons, lats, debug=debug)                                           

        #print("RAIN")
        tmirain = regrid_swath(f['TMI']['RAIN'][:], 
                              f['TMI']['LONHI'][:],
                              f['TMI']['LATHI'][:], lons, lats, debug=debug)                                           

        #print("CLDWPATH")
        tmiclp = regrid_swath(f['TMI']['CLDWPATH'][:], 
                              f['TMI']['LONHI'][:],
                              f['TMI']['LATHI'][:], lons, lats, debug=debug)                                           

        #print("RAINWPATH")
        tmirwp = regrid_swath(f['TMI']['RAINWPATH'][:], 
                              f['TMI']['LONHI'][:],
                              f['TMI']['LATHI'][:], lons, lats, debug=debug)                                           

        #print("WINDSPEED")
        tmiwind = regrid_swath(f['TMI']['WINDSPEED'][:], 
                              f['TMI']['LONHI'][:],
                              f['TMI']['LATHI'][:], lons, lats, debug=debug)                                           


        #print("PRMAXSURFZ")
        prnsz = regrid_swath(f['PR']['2A25']['NEARSURFZ'][:], 
                             f['PR']['2A25']['LON'][:],
                             f['PR']['2A25']['LAT'][:], lons, lats, debug=debug)                                           

        #print("RAINTYPE2A23")
        prraintyp = regrid_swath(f['PR']['2A23']['RAINTYPE2A23'][:], 
                             f['PR']['2A25']['LON'][:],
                             f['PR']['2A25']['LAT'][:], lons, lats, debug=debug)                                           

        #print("STORMH")
        prstormh = regrid_swath(f['PR']['2A23']['STORMH'][:], 
                             f['PR']['2A25']['LON'][:],
                             f['PR']['2A25']['LAT'][:], lons, lats, debug=debug)                                           

        #print("RAIN_2B31")
        prrain = regrid_swath(f['PR']['2A25']['RAIN_2B31'][:], 
                             f['PR']['2A25']['LON'][:],
                             f['PR']['2A25']['LAT'][:], lons, lats, debug=debug)                                           

        #print("PRECIP_2B31")
        prprecip = regrid_swath(f['PR']['2A25']['PRECIP_2B31'][:], 
                             f['PR']['2A25']['LON'][:],
                             f['PR']['2A25']['LAT'][:], lons, lats, debug=debug)                                           


        # Plot regridded data
        # --------------------------------------------------------------------------------------------------
        
        if debug and 1 == 0:
          #m = [{"var": "TMI PCT37", "data": pct37, "vmin": 50, "vmax": 320},
          #     {"var": "TMI PCT85", "data": pct85, "vmin": 50, "vmax": 320},
          #     {"var": "VIRS CH4",  "data": irch4, "vmin": 50, "vmax": 320}]
          #plot_swath_rgb(lons, lats, m)

          #m = [{"var": "TMI PCT37", "data": pct37, "vmin": 50, "vmax": 320},
          #     {"var": "TMI PCT85", "data": pct85, "vmin": 50, "vmax": 320},
          #     {"var": "VIRS CH4",  "data": irch4, "vmin": 50, "vmax": 320}]

          m = [{"var": "TMI PCT37", "data": pct37, "vmin": 50, "vmax": 320},
               {"var": "TMI PCT85", "data": pct85, "vmin": 50, "vmax": 320},
               {"var": "PR Max Sfc. Z",  "data": prnsz, "vmin": 0, "vmax": 60}]
          plot_regrid_rgb(lons, lats, m)

        # Create unique multi-spectral views...
        # --------------------------------------------------------------------------------------------------

        #ii = np.where(pct37 > 50)
        #npct37 = (np.mean(pct37[ii])-pct37)/np.std(pct37[ii])
        #print(np.mean(pct37[ii]), np.std(pct37[ii]))

        #ii = np.where(pct85 > 50)
        #npct85 = (np.mean(pct85[ii])-pct85)/np.std(pct85[ii])
        #print(np.mean(pct85[ii]), np.std(pct85[ii]))

        #ii = np.where(irch4 > 50)
        #nirch4 = (np.mean(irch4[ii])-irch4)/np.std(irch4[ii])        
        #print(np.mean(irch4[ii]), np.std(irch4[ii]))

        #ii = np.where(tmipw >= 0)
        #print(np.mean(tmipw[ii]), np.std(tmipw[ii]))
        #ntmipw = (tmipw-np.mean(tmipw[ii]))/np.std(tmipw[ii])        
        #print(np.mean(tmipw[ii]), np.std(tmipw[ii]))

        #ii = np.where(tmiiwp >= 0)
        #print(np.mean(tmiiwp[ii]), np.std(tmiiwp[ii]))
        #ntmiiwp = (tmiiwp-np.mean(tmiiwp[ii]))/np.std(tmiiwp[ii])        
        #print(np.mean(tmiiwp[ii]), np.std(tmiiwp[ii]))
        
        
        
        #pred = prnsz
        #threshes = np.linspace(15, 55, 9)

        #pred = tmipw
        #npred = ntmipw
        #threshes = np.linspace(0, 9, 10)

        pred = prstormh
        #npred = ntmiiwp
        threshes = np.linspace(0, 19000, 20)
                        
        dt = threshes[1] - threshes[0]

        for thresh in np.flip(threshes):
          id = np.where((pred >= thresh) & (pred < thresh+dt))
          
          nw = 100
          if len(id[0]) < 100:
            nw = len(id[0])

          #nw = len(id[0])
            
          #a = np.random.shuffle(np.argsort(prnsz[id]))
          a = np.argsort(pred[id])
          #np.random.shuffle(a)
          
          iw = 0
          while iw < nw and iw < len(a):
            ix = id[0][a[iw]]
            iy = id[1][a[iw]]
            
            #print(iw, ix, iy)
            
            x0 = ix-delta
            x1 = ix+delta
            y0 = iy-delta
            y1 = iy+delta
            
            view = np.expand_dims(np.dstack((pct37[x0:x1,y0:y1],
                                             pct85[x0:x1,y0:y1],
                                             irch4[x0:x1,y0:y1],
                                             pct10[x0:x1,y0:y1],
                                             pct19[x0:x1,y0:y1],
                                             tmipw[x0:x1,y0:y1],
                                             tmiiwp[x0:x1,y0:y1],
                                             tmirain[x0:x1,y0:y1],
                                             tmiclp[x0:x1,y0:y1],
                                             tmirwp[x0:x1,y0:y1],
                                             tmiwind[x0:x1,y0:y1],
                                             prnsz[x0:x1,y0:y1],
                                             prraintyp[x0:x1,y0:y1],
                                             prstormh[x0:x1,y0:y1],
                                             prrain[x0:x1,y0:y1],
                                             prprecip[x0:x1,y0:y1],
                                             fed_apx[x0:x1,y0:y1],
                                             fed_sc[x0:x1,y0:y1],
                                             lis_vt[x0:x1,y0:y1])), axis=0)
            
            if debug and 1 == 1:
              plot_view(view, str(int(thresh)).zfill(3)+'_'+str(iw).zfill(3))
              
              #sys.exit()

            if w is None:
              w = view
            else:
              #print(np.shape(w), np.shape(view))
              try:
                w = np.vstack((w, view))
              except:
                pass

            iw += 1
            
            #sys.exit()
          
          print(thresh, np.shape(id), nw, np.shape(a))

        oc +=1
        
        #break

        sys.exit()
      sys.exit()

      # Save daily results...
      
      print(datetime.now(), oc)

      w = np.array(w)          
      print(np.shape(w))
        
      # Save training data...

      ofname = ofile.replace('{date}',year+month+day).replace('{year}',year)
      print(ofname)
      
      if not os.path.exists(os.path.dirname(ofname)):
        os.makedirs(os.path.dirname(ofname))
      if os.path.exists(ofname):
        os.remove(ofname)
          
      o = h5py.File(ofname, 'w')
      o.create_dataset("views", data=w)
      o.close()

      #sys.exit()


  #    if oc > 0:
  #      break
  #  if oc > 0:
  #    break
  #if oc > 0:
  #  break




