
import pandas as pd
import re
import xarray as xr
import numpy as np
import os
import datetime
import requests
import os
from math import *
import time
import sys
from datetime import datetime


# Define paths
path_tools = r"C:\Users\Usuario\Documents\AI\Semestre 1\GeoSpatial\Codigo\utils"

# Add paths to sys.path if not already present
if path_tools not in sys.path:
    sys.path.append(path_tools)


from tools import map_to_0_to_360,boundaries


class Modelos:
    
    def __init__(self,
                 modelo,
                 tuple_code,
                ):

        self.url_generico = r"https://nex-gddp-cmip6.s3-us-west-2.amazonaws.com/NEX-GDDP-CMIP6/{modelo}/{escenario}/r1i1p1f1/pr/pr_day_{modelo}_{escenario}_r1i1p1f1_{time_code}_{año}.nc"

        
        
        self.modelo = modelo
        ## self.escenarios = ['historical','ssp245','ssp370','ssp585']
        self.escenarios,self.time_code = tuple_code  # e.g       
        
        self.time_range_hind = (1985,2015)
        ## self.time_range_future = (2015,2051)
        self.time_range_future = (2015,2100)
        
    def generate_links(self,):
        

        escenario_links = {}
        for escenario in self.escenarios:
                    
            list_escenarios = escenario_links.setdefault(escenario,[])
            if escenario == 'historical':
                
                start_year,end_year  =  self.time_range_hind
                for año in range(start_year,end_year,1):
                    link = self.url_generico.format(modelo = self.modelo,escenario = escenario,time_code = self.time_code,año =año)
                    list_escenarios.append(link)       
            else:
                start_year,end_year  =  self.time_range_future
                
                for año in range(start_year,end_year,1):
                    link = self.url_generico.format(modelo = self.modelo,escenario = escenario,time_code = self.time_code,año =año)
                    
                    list_escenarios.append(link)
                    
        return escenario_links


def verify_completeness(url_directory :str):
    
    """
    This function is intended to verify in a rapid way that each url works properly providing alon the way both the bounds in space as well as the range of years.
    """

    current_time = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    print(f"Starting download at :{current_time}\n")

    os.chdir(url_directory)
    global modelos_metadata
    
    for modelo,type_code in modelos_metadata.items():

        print(f"Modelo = {modelo}\n")
        
        modelo_instance = Modelos(modelo = modelo,tuple_code = type_code)
        
        dict_links = modelo_instance.generate_links()

        print(f"Ventana futura : {modelo_instance.time_range_future}")
        print(f"Ventana Pasado : {modelo_instance.time_range_hind}")

        for (escenario,links_list) in dict_links.items():

            print(f"Escenario\t:{escenario}")

            for index,link in enumerate(links_list,start=1):
                
                if index == 1:
                
                    response = requests.get(link)
                    sample_name = f"GPDX_delete_{index}.nc"

                    if response.status_code == 200:

                        content = response.content


                        with open(sample_name, "wb") as f:
                            f.write(content)

                        f = None

                        # Now open the file with xarray and dask
                        ds = xr.open_dataset(sample_name)

                        lon_min = float(ds['lon'].min() )
                        lon_max = float(ds['lon'].max() )
                        lat_min = float(ds['lat'].min() )
                        lat_max = float(ds['lat'].max() )

                        print(f"lon :\t({lon_min},{lon_max})")
                        print(f"lat :\t({lat_min},{lat_max})")
                        ds.close()
                        os.remove(sample_name)
                    else:
                        print("Fail to catch")
                else:
                    continue

            print("\n")

        print("\n")

    
def main_download(url_directory :str):
    
    global boundaries


    modelos_metadata = {
                   'GFDL-ESM4': (['historical','ssp245','ssp370','ssp585'], 'gr1'),
                   'MIROC6': (['historical','ssp245','ssp370','ssp585'], 'gn'),
                   'MPI-ESM1-2-HR': (['historical','ssp245','ssp370','ssp585'], 'gn'),
                   'MPI-ESM1-2-LR': (['historical','ssp245','ssp370','ssp585'], 'gn'),
                   'MRI-ESM2-0': (['historical','ssp245','ssp370','ssp585'], 'gn'),
                   'NorESM2-MM': (['historical','ssp245','ssp370','ssp585'], 'gn'),
                 'EC-Earth3-Veg-LR' : (['historical','ssp245','ssp370','ssp585'], 'gr')
                  }


    
    
    current_time = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    print(f"Starting download at :{current_time}\n")
        
    print("Downloading all south america")
    ## lon_min,lon_max = (-84.0833129206595,  -34.73304413246237)
    ## lat_min,lat_max = (-53.43306775676489, 13.592638486971083)

    south_america = boundaries['south_america']
    lat_min,lat_max = (south_america['lat_min'], south_america['lat_max'])
    lon_min,lon_max = (south_america['lon_min'], south_america['lon_max'])

    print(f"Latitud boundaries : {(lat_min,lat_max)}")
    print(f"Longitude boundaries : {(lon_min,lon_max)}")

    print("\n")

    already_download = os.listdir(url_directory)

    if len(already_download) != 0:
        print("Empty the folder first")
        return  None
    else:
        print("Folder emppy , all good to proccede")

    print(f"Current amount of files in folder {len(already_download)}\n")
    
    print(f"Files are going into\n {url_directory}\n")
    lon_min_360 = map_to_0_to_360(regular_longitude = lon_min)
    lon_max_360 = map_to_0_to_360(regular_longitude = lon_max)
    print(f"After 0-360 transformation {(lon_min_360, lon_max_360)}")
  
    print("Downloding files from GPDX Files.\n")

    chunk_size = 10   ## Store each 10 years.
    
    pattern_time = ".+?_(\d{4}).nc$"

    look_date = re.compile(pattern_time)

    for modelo,tuple_code in modelos_metadata.items():
    
        print(f"Modelo = {modelo}\n")
        modelo_instance = Modelos(modelo=modelo,tuple_code=tuple_code)  
        dict_links = modelo_instance.generate_links()
        
        for escenario, list_urls in dict_links.items():
            
            print(f"\tEscenario = {escenario}\n")
            counter = 0

            for chunk_i, i in enumerate(range(0, len(list_urls), chunk_size),start = 1):

                name_file_nc = f"SAM_pr_{modelo}_{escenario}_{chunk_i}.nc"
                
                if name_file_nc in already_download:
                    print(f"Skiped {name_file_nc}")
                    continue

                print(f"\t\tChunk {chunk_i}")
                chunk_links = list_urls[i:i + chunk_size]

                start_date = chunk_links[0]
                end_date = chunk_links[-1]

                year_start= look_date.match(start_date).group(1)
                year_end = look_date.match(end_date).group(1)
                
                print(f"\t\t\t- Starts at {year_start} - {year_end}\n")

                escenario_cmip6 = []
                all_files_alright = True
                
                for index,link in enumerate(chunk_links,start = counter + 1):
                    
                    response = requests.get(link)
                    path_sample_complete = os.path.join(url_directory, f"GPDX_delete_{modelo}_{escenario}_{chunk_i}_{index}.nc")
    
                    if response.status_code == 200:
                        
                        content = response.content
                        with open(path_sample_complete, "wb") as f:
                            f.write(content)
                        f = None
                        # Now open the file with xarray and dask
                        ds = xr.open_dataset(path_sample_complete)
                        colombia_subset = ds.sel(lon = slice(lon_min_360,lon_max_360),lat = slice(lat_min,lat_max))
                        escenario_cmip6.append(colombia_subset)
                        
                    else:
                        print(f"Failed to fetch URL\n\tMistake at chunk {chunk_i},chunk not being saved.", response.status_code)
                        all_files_alright = False
                        break

                    counter += 1

                if all_files_alright:
                    concat_file = xr.concat(escenario_cmip6,dim = 'time')
                    offitial_path = os.path.join(url_directory, name_file_nc)
                    concat_file.to_netcdf(offitial_path,format = 'NETCDF3_64BIT')                                                            
                    time.sleep(30)
                  
                else:
                    print(f"Chunk : {chunk_i}, not being saved.")

                del escenario_cmip6
                ds.close()

                folders_to_delete = [delete for delete in os.listdir(url_directory) if 'delete' in delete]

                for delete in folders_to_delete:
                    path_deleted =  os.path.join(url_directory, delete)
                    os.remove(path_deleted)
                                    
                print("\n")

    print("Done !!! ")
            

