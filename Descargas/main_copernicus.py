import cdsapi
import os
import xarray as xr
from time import gmtime, strftime,sleep
import pandas as pd
import sys
import time

from tools import boundaries,get_current_time


def pull_south_request(start_year = 1970,
                        end_year = 2024):
    

    """
    Funcion ofical para SouthAmerica.
    variables = ["reanalysis-era5-land","reanalysis-era5-single-levels"]
    
    """    
    
    global boundaries

    get_current_time()
    ## common_folder = r"D:\OneDrive\OneDrive - INTERCONEXION ELECTRICA S.A. E.S.P\01_CC_DOC\1_Raw_Data\10_Copernicus"
    common_folder = r"D:\OneDrive\OneDrive - INTERCONEXION ELECTRICA S.A. E.S.P\01_CC_DOC\data\10_Copernicus\SouthAmerica"
    name_code_south = 'south_america'
    area_bounds = [
                      boundaries[name_code_south]['lat_max'],
                      boundaries[name_code_south]['lon_min'],
                      boundaries[name_code_south]['lat_min'],
                      boundaries[name_code_south]['lon_max']
                   ]

    print(f"Area: {area_bounds}\n")
    key_structure = { 

        ( "reanalysis-era5-single-levels", "singlelevels") : 
                                                             [

                                                           ('fg', "10m_wind_gust_since_previous_post_processing"),
                                                           ('i10fg', "instantaneous_10m_wind_gust"),
                                                           ('10uv', ["10m_u_component_of_wind","10m_v_component_of_wind"]),
                                                           ('t2m', "2m_temperature"),
                                                           ('mx2t', "maximum_2m_temperature_since_previous_post_processing"),
                                                                 
                                                         ###  ('mn2t', "minimum_2m_temperature_since_previous_post_processing"),
                                                         ###  ('100uv', ["100m_u_component_of_wind", "100m_v_component_of_wind" ]),
                                                             ],
    
    
        ( "reanalysis-era5-land", "era5l"): 
                                                          [ ('tp' , "total_precipitation"),
                                                         ##   ('t2m', "2m_temperature"),
                                                            ('skint',"skin_temperature"),
                                                       ##     ('10uv',  ["10m_u_component_of_wind",  "10m_v_component_of_wind"]),
                                                            ('snsr', "surface_net_solar_radiation"),
                                                            ('ssrd', "surface_solar_radiation_downwards"),
                                                          ]
                                            ,
                        }


    for (complete_db , short_name),list_vars in key_structure.items():

        var_path = os.path.join(common_folder, short_name)
        os.makedirs(var_path,exist_ok = True)

        for code_name , ecmwf_code in list_vars:
            

            print(f"- Downloading { ecmwf_code}\n")

            code_name_path = os.path.join(var_path, code_name)

            if not os.path.exists(code_name_path):
                print(f"Creating folder {code_name }\n\n")
                os.makedirs(code_name_path,exist_ok = True)
 
            requests = []
            skipped_files = []
            total_amount_files = 0
            
            for year in range(start_year, end_year + 1):
                # Generate the date range for the entire year
                date_range = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq='D')
                
                # Group dates by month
                grouped_dates = date_range.to_series().groupby(date_range.month)
                
                # Create the request for each month
                for month, dates in grouped_dates:
                    
                    total_amount_files += 1
                    monthly_filename_pais = "hourly_{code_name}_{year}_{month:02d}.nc".format(
                                                                                               code_name = code_name,
                                                                                               year = year,
                                                                                               month = month
                                                                                              )
                    target = os.path.join(code_name_path, monthly_filename_pais)
                    
                    # Check if the file already exists
                    if os.path.exists(target):
                        skipped_files.append(monthly_filename_pais)
                        continue
        
                    # This if- elif statement are due to the fact that the request dictionary is different for downlaod form reanalysys and era5-land
                    if complete_db == "reanalysis-era5-land":
                        
                        request = {
                                    "variable": ecmwf_code,
                                    "year": str(year),
                                    "month": f"{month:02d}",
                                    "day": [date.strftime("%d") for date in dates],
                                    "time": [
                                        "00:00", "01:00", "02:00",
                                        "03:00", "04:00", "05:00",
                                        "06:00", "07:00", "08:00",
                                        "09:00", "10:00", "11:00",
                                        "12:00", "13:00", "14:00",
                                        "15:00", "16:00", "17:00",
                                        "18:00", "19:00", "20:00",
                                        "21:00", "22:00", "23:00"
                                    ],
                                    "data_format": "netcdf",
                                    "download_format": "unarchived",
                                    "area": area_bounds
                                }
        
                    elif complete_db == "reanalysis-era5-single-levels":
        
                        request = {
                                   "product_type": ["reanalysis"] ,
                                    "variable": ecmwf_code ,
                                    "year": str(year),
                                    "month": f"{month:02d}",
                                    "day": [date.strftime("%d") for date in dates],
                                    "time": [
                                        "00:00", "01:00", "02:00",
                                        "03:00", "04:00", "05:00",
                                        "06:00", "07:00", "08:00",
                                        "09:00", "10:00", "11:00",
                                        "12:00", "13:00", "14:00",
                                        "15:00", "16:00", "17:00",
                                        "18:00", "19:00", "20:00",
                                        "21:00", "22:00", "23:00"
                                    ],
                                    "data_format": "netcdf",
                                    "download_format": "unarchived",
                                    "area": area_bounds
                                }
                    
                    requests.append((request,monthly_filename_pais, target))
                    
            ### client = cdsapi.Client()
            
            # Print skipped files

            skiped_amount = len(skipped_files)

            print(f"Amount of files downloaded {skiped_amount }\n")
            print(f"Total Amount of months to be downloaded {total_amount_files }\n")
            pctage = (skiped_amount/total_amount_files) * 100
            print(f"Percentage :\t {pctage :.2f} [%] \n")
            
            if skiped_amount == total_amount_files:
                
                print(f"########## Var {code_name } Already completed ##########")
                continue
                
            elif skiped_amount > 10:
                print("Skipped files:")
                for file in skipped_files[:5]:
                    print(file)
                print("... Hidden Files ...")
                for file in skipped_files[-5:]:
                    print(file)
            else:
                for file in skipped_files:
                    print(file)

            client = cdsapi.Client()
        
            for request,monthly_filename_pais, target in requests:
        
                print(f"Working on : {monthly_filename_pais}")
                start_time = time.time()
                client.retrieve(complete_db, request, target = target)
                end_time = time.time()
                elapsed_time = (end_time - start_time) / 60  # Convert time to minutes
                print(f"\t Successfully downloaded {monthly_filename_pais} in {elapsed_time:.2f} minutes")  
            print(f"\n\nDone {ecmwf_code}!!!\n")

    return None

    
    
