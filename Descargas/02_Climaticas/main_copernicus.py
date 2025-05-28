import cdsapi
import os
import xarray as xr
from time import gmtime, strftime,sleep
import pandas as pd
import sys
import time

## Note
## This module is attache to the foolowing notebooks 
## 03_Escalamiento/Download Database/2_Hydrology/01_Copernicus/readme.txt
## path_globales = r"D:\OneDrive\OneDrive - INTERCONEXION ELECTRICA S.A. E.S.P\01_CC_DOC\Python Code\03_Escalamiento\Articulador\01_FuncionesGlobales"


path_globales = r"D:\OneDrive\OneDrive - INTERCONEXION ELECTRICA S.A. E.S.P\01_CC_DOC\src\utils"


if path_globales not in sys.path:
    sys.path.append(path_globales)

from tools import define_boundaries,boundaries,get_current_time


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




def create_requests_multiple_var(start_year = 1970,
                                end_year = 2024,
                                pais = 'Colombia',
                                dataset = str):

    """
    Funcion ofical para Colombia.
    variables = ["reanalysis-era5-land","reanalysis-era5-single-levels"]
    
    """    
    
    global boundaries
    ## common_folder = r"D:\OneDrive\OneDrive - INTERCONEXION ELECTRICA S.A. E.S.P\01_CC_DOC\1_Raw_Data\10_Copernicus"
    common_folder = r"D:\OneDrive\OneDrive - INTERCONEXION ELECTRICA S.A. E.S.P\01_CC_DOC\data\10_Copernicus"

    area_bounds = [
                  boundaries[pais]['lat_max'],
                  boundaries[pais]['lon_min'],
                  boundaries[pais]['lat_min'],
                  boundaries[pais]['lon_max']
               ]

    print(f"Area: {area_bounds}\n")

    if dataset == "reanalysis-era5-land":
        
        short_name = "era5l"
        variables_download = [
                                "2m_temperature",
                                "skin_temperature",
                        
                                "surface_net_solar_radiation",
                                "surface_solar_radiation_downwards",
                        
                                "10m_u_component_of_wind",
                                "10m_v_component_of_wind",
                                "total_precipitation"
                            ]

    elif dataset == "reanalysis-era5-single-levels":
        
        short_name = "singlelevels"
        variables_download=  [
                            "10m_u_component_of_wind",
                            "10m_v_component_of_wind",
                    
                            "2m_temperature",
                            "maximum_2m_temperature_since_previous_post_processing",
                            "minimum_2m_temperature_since_previous_post_processing",
                    
                            "100m_u_component_of_wind",
                            "100m_v_component_of_wind",
                    
                            "10m_wind_gust_since_previous_post_processing",
                            "instantaneous_10m_wind_gust"
                        ]

    if pais == 'Colombia':
        var_pais = 'COP'
    elif pais == 'Brasil':
        var_pais = 'BRA'
        

    root_pais = os.path.join(common_folder,var_pais )
    var_path = os.path.join(root_pais, short_name)
    os.makedirs(var_path,exist_ok = True)
    
    requests = []
    skipped_files = []
    
    for year in range(start_year, end_year + 1):
        # Generate the date range for the entire year
        date_range = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq='D')
        
        # Group dates by month
        grouped_dates = date_range.to_series().groupby(date_range.month)
        
        # Create the request for each month
        for month, dates in grouped_dates:

            monthly_filename_pais = "monthly_{var_pais}_{short_name}_{year}_{month:02d}.zip".format(var_pais = var_pais,
                                                                                                   short_name=short_name,
                                                                                                   year = year ,
                                                                                                   month = month
                                                                                                  )
            target = os.path.join(var_path, monthly_filename_pais)
            
            # Check if the file already exists
            if os.path.exists(target):
                skipped_files.append(target)
                continue

            # This if- elif statement are due to the fact that the request dictionary is different for downlaod form reanalysys and era5-land
            if dataset == "reanalysis-era5-land":
                
                request = {
                            "variable": variables_download,
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
                            "download_format": "zip",
                            "area": area_bounds
                        }

            elif dataset == "reanalysis-era5-single-levels":

                request = {
                           "product_type": ["reanalysis"] ,
                            "variable": variables_download ,
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
                            "download_format": "zip",
                            "area": area_bounds
                        }
            
            requests.append((request,monthly_filename_pais, target))

    client = cdsapi.Client()
    # Print skipped files
    if len(skipped_files) > 10:
        print("Skipped files:")
        for file in skipped_files[:5]:
            print(file)
        print("... Hidden Files ...")
        for file in skipped_files[-5:]:
            print(file)
    else:
        for file in skipped_files:
            print(file)

    for request,monthly_filename_pais, target in requests:

        print(f"Working on : {monthly_filename_pais}")
        client.retrieve(dataset, request, target = target)
        print(f"\t Successfully downloaded {monthly_filename_pais}")
        
    print("\n\nDone !!!")
    return None



def create_requests_cop(start_year, end_year, variable):

    """
    Funcion ofical para Colombia.

    """
    

    global boundaries

    ## common_folder = r"D:\OneDrive\OneDrive - INTERCONEXION ELECTRICA S.A. E.S.P\01_CC_DOC\1_Raw_Data"
    common_folder = r"D:\OneDrive\OneDrive - INTERCONEXION ELECTRICA S.A. E.S.P\01_CC_DOC\data"
    
    ### dataset = "reanalysis-era5-land"

    area_colombia_bounds = [
                              boundaries['Colombia']['lat_max'],
                              boundaries['Colombia']['lon_min'],
                              boundaries['Colombia']['lat_min'],
                              boundaries['Colombia']['lon_max']
                           ]

    print(f"Area Colombiana {area_colombia_bounds}\n")

    if variable == 'total_precipitation':
        
        path_variable = r"4_Rain\01_Copernicus\reanalysis-era5-land_cop"
        variables_download = variable
        dataset = "reanalysis-era5-land"
        monthly_filename = "monthly_SouthAmerica_era5_land_{year}_{month:02d}.nc"
    
    elif variable == 'era5_land_wind_10m':
        
        path_variable = r"2_WindGust\02_ERA5\era5_land_wind_10m_cop"        
        variables_download = ["10m_u_component_of_wind", "10m_v_component_of_wind"]
        dataset = "reanalysis-era5-land"
        monthly_filename = "monthly_SouthAmerica_era5_land_{year}_{month:02d}.nc"

    elif variable == '2m_temperature':
        path_variable = r"1_Temperature\02_Copernicus\2m_temperature_cop"
        variables_download = variable
        dataset = "reanalysis-era5-land"
        monthly_filename = "monthly_SouthAmerica_era5_land_{year}_{month:02d}.nc"
    
    elif variable == "10m_wind_gust_since_previous_post_processing":
        variable_name = "gust_cop"
        path_variable = r"2_WindGust\02_ERA5\10m_wind_gust_cop"        
        variables_download =  ["10m_wind_gust_since_previous_post_processing"]
        dataset = "reanalysis-era5-single-levels"
        monthly_filename = "monthly_SouthAmerica_reanalysis_{year}_{month:02d}.nc"
                              
    elif variable == "instantaneous_10m_wind_gust":
        variable_name = "gust_cop"
        path_variable = r"2_WindGust\02_ERA5\instantaneous_10m_wind_gust_cop"        
        variables_download = ["instantaneous_10m_wind_gust"]
        dataset = "reanalysis-era5-single-levels"
        monthly_filename = "monthly_SouthAmerica_reanalysis_{year}_{month:02d}.nc"


    client = cdsapi.Client()
    root_folder = os.path.join(common_folder,path_variable)

    requests = []
    skipped_files = []
    
    for year in range(start_year, end_year + 1):
        # Generate the date range for the entire year
        date_range = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq='D')
        
        # Group dates by month
        grouped_dates = date_range.to_series().groupby(date_range.month)
        
        # Create the request for each month
        for month, dates in grouped_dates:

            format_monthly_filename = monthly_filename.format(year= year , month = month)
            complete_name = os.path.join(root_folder, format_monthly_filename)
            
            # Check if the file already exists
            if os.path.exists(complete_name):
                skipped_files.append(complete_name)
                continue


            # This if- elif statement are due to the fact that the request dictionary is different for downlaod form reanalysys and era5-land
            if dataset == "reanalysis-era5-land":
                
                request = {
                    "variable": variables_download,
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
                    "area": area_colombia_bounds
                }

            elif dataset == "reanalysis-era5-single-levels":

                request = {
                       "product_type": ["reanalysis"] ,
                        "variable": variables_download ,
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
                        "area": area_colombia_bounds
                    }
                
            requests.append((request, complete_name))
    
    # Print skipped files
    if len(skipped_files) > 10:
        print("Skipped files:")
        for file in skipped_files[:5]:
            print(file)
        print("...")
        for file in skipped_files[-5:]:
            print(file)
    else:
        for file in skipped_files:
            print(file)

    for request, target in requests:
        client.retrieve(dataset, request, target = target)
        print(f"Successfully downloaded {target}")
    
    return None

def create_requests_cteep(start_year= int, end_year= int, request = int ):


    """
    5/03/2025

    Hoy se creo esta funcion para automatizar de una vez las descaargas de brazil. Se Construyo aparte debido a que estamos apurados de tiempo. 

    Empezaremos solo con las de rafaga a 0.25 ° por que las de era5 land a 9 km no estan disponibles.
    """
    

    global boundaries

    ### common_folder = r"D:\OneDrive\OneDrive - INTERCONEXION ELECTRICA S.A. E.S.P\01_CC_DOC\1_Raw_Data"
    common_folder = r"C:\Users\Usuario\OneDrive - INTERCONEXION ELECTRICA S.A. E.S.P\01_CC_DOC\data"

    area_colombia_bounds = [
                              boundaries['Brasil']['lat_max'],
                              boundaries['Brasil']['lon_min'],
                              boundaries['Brasil']['lat_min'],
                              boundaries['Brasil']['lon_max']
                           ]

    print(f"Area Brasil {area_colombia_bounds}\n")

    dataset = "reanalysis-era5-single-levels"
    
    if request == 1:
        variable_name = "gust"
        path_variable = r"2_WindGust\02_ERA5\10m_wind_gust_cteep"        
        variables_download =  ["10m_wind_gust_since_previous_post_processing"]
                              
    elif request == 2:
        variable_name = "gust"
        path_variable = r"2_WindGust\02_ERA5\instantaneous_10m_wind_gust_cteep"        
        variables_download = ["instantaneous_10m_wind_gust"]


    print(variables_download)
    
    client = cdsapi.Client()
    root_folder = os.path.join(common_folder,path_variable)

    requests = []
    skipped_files = []
    
    for year in range(start_year, end_year + 1):
        # Generate the date range for the entire year
        date_range = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq='D')
        
        # Group dates by month
        grouped_dates = date_range.to_series().groupby(date_range.month)
        
        # Create the request for each month
        for month, dates in grouped_dates:
            
            monthly_filename = f"monthly_{variable_name}_era5l_{year}_{month:02d}.nc"
            complete_name = os.path.join(root_folder, monthly_filename)
            
            # Check if the file already exists
            if os.path.exists(complete_name):
                skipped_files.append(complete_name)
                continue
            
            request = {
                       "product_type": ["reanalysis"] ,
                        "variable": variables_download ,
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
                        "area": area_colombia_bounds
                    }
            requests.append((request, complete_name))
    
    # Print skipped files
    if len(skipped_files) > 10:
        print("Skipped files:")
        for file in skipped_files[:5]:
            print(file)
        print("...")
        for file in skipped_files[-5:]:
            print(file)
    else:
        for file in skipped_files:
            print(file)

    for request, target in requests:
        client.retrieve(dataset, request, target = target)
        print(f"Successfully downloaded {target}")
    
    return None

    
def download_era5_by_month(variable: str):
    
    """
    Request by month
        
    Variables available: [
                          'total_precipitation',
                          'era5_land_wind_10m',    
                            '2m_temperature',
                          
                          'instantaneous_10m_wind_gust'    ### Thisi s for the coarser resolution of 0.25 °... i still dont know what to do with it.
                          ]
    """
    
    
    current_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())

    if variable == 'total_precipitation':
        path_variable = r"4_Rain\01_Copernicus\reanalysis-era5-land"
        variables_download = variable

    elif variable == 'era5_land_wind_10m':
        path_variable = r"2_WindGust\02_ERA5\era5_land_wind_10m"        
        variables_download = ["10m_u_component_of_wind", "10m_v_component_of_wind"]
        
    elif variable == 'instantaneous_10m_wind_gust':
        path_variable = r"2_WindGust\02_ERA5\instantaneous_10m_wind_gust"
        variables_download = variable

    elif variable == '2m_temperature':
        path_variable = r"1_Temperature\02_Copernicus\2m_temperature"
        variables_download = variable

    common_root = r"D:\OneDrive\OneDrive - INTERCONEXION ELECTRICA S.A. E.S.P\01_CC_DOC\1_Raw_Data"
    
    root_folder = os.path.join(common_root, path_variable)
    south_america_slice = define_boundaries(all_south_america=True)

    client = cdsapi.Client()
    available_years = list(range(1980, 2024))
    months = [f"{month:02}" for month in range(1, 13)]

    already_downloaded = os.listdir(root_folder)

    print(f"Chosen variable: {variable}\nDownloading all South America...\nSlice: {south_america_slice['south_america']}\nStarting...")

    for year in available_years:
        for month in months:
            # Check if this year's month is already downloaded
            is_already_downloaded = any(f"{year}_{month}" in file for file in already_downloaded)

            if is_already_downloaded:
                print(f"- Skipping {year}-{month}, already downloaded.")
                continue

            print(f"Requesting {year}-{month}...")

            # Construct the filename
            monthly_filename = f"monthly_SouthAmerica_era5_land_{year}_{month}.nc"

            # Request details
            request = {
                        'variable': variables_download,
                        'year': str(year),
                        'month': month,
                        'day': [f"{day:02}" for day in range(1, 32)],
                        'time': [f"{hour:02}:00" for hour in range(24)],
                        'data_format': 'netcdf',
                        'download_format': 'unarchived',
                        'area': south_america_slice['south_america']
                    }

            ### os.chdir(root_folder)

            complete_name = os.path.join(root_folder, monthly_filename)

            # Submit the request
            try:
                dataset = "reanalysis-era5-land"
                client.retrieve(dataset, request, target=complete_name)
                print(f"\tCompleted {year}-{month}")
                print("Sleeping [zzz]")
                sleep(10)
                
            except Exception as e:
                print(f"\tFailed to download {year}-{month}: {e}")
            
            # Small delay between requests
            sleep(30)


    return None

    
    
