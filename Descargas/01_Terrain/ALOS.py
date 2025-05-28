## Virtual envrioment geoisa.


import asf_search as asf
import os 
import zipfile
import getpass
import pandas as pd
import re
import time

from datetime import datetime


class AlosPalsar:
    
    def __init__(self,pais : str):

        try:
            self.user_pass_session = asf.ASFSession().auth_with_creds('geoisavolkmar', '$2GL#,@JZx!bb&*')
        except asf.ASFAuthenticationError as e:
            print(f'Auth failed: {e}')
        else:
            print('Success!')
            
        ## New Colombia Map.

        if pais == 'Colombia':
            aoi = 'POLYGON ((-67.19504547199995 2.391268969000066, -67.09666442999998 1.1700752970000394, -67.40979766899994 2.141286849000039, -69.85519409299997 1.714917898000067, -69.11193084699994 0.6427631380000207, -70.03735351599994 0.5530209540000328, -69.36834716899995 -1.3337022069999307, -69.94850158699995 -4.228269576999935, -70.71313197999996 -3.7964252219999253, -70.30097760499996 -2.502577346999942, -72.93380398199997 -2.454868595999926, -74.78245735899998 -0.2013574689999587, -77.46455383399996 0.3949180250000381, -79.04722595199996 1.6130367520000277, -77.00271606399997 3.909264803000042, -77.51924133299997 4.149404049000054, -77.23348236099997 5.794978142000048, -77.88681793199999 7.220828056000073, -77.17687225299994 7.926280975000054, -77.39372253399995 8.643045425000025, -76.76051330599995 7.912589073000049, -76.93661499099994 8.55289554500007, -75.62319946399998 9.409773827000038, -75.25900268599997 10.805982590000042, -71.26570129499999 12.340830803000074, -73.37364959799999 9.177592278000077, -72.76882171599993 9.106621741000026, -72.02432250999993 7.0138750080000705, -70.08931732199994 7.002468109000063, -69.44143676799996 6.112888812000051, -67.47605133199994 6.19459056900007, -67.87779235799997 4.536076068000057, -67.30262756299999 3.3979396820000716, -67.85439300499996 2.869814396000038, -67.85923004299997 2.789511442000048, -67.19504547199995 2.391268969000066))'

            ### self.url_storage = r"C:\Users\Usuario\OneDrive - INTERCONEXION ELECTRICA S.A. E.S.P\01_CC_DOC\1_Raw_Data\06_InputGeologico\02_Alos_Palsar\dem_COL"
            ### self.url_storage = r"C:\Users\Usuario\OneDrive - INTERCONEXION ELECTRICA S.A. E.S.P\01_CC_DOC\1_Raw_Data\06_InputGeologico\02_Alos_Palsar\dem_COL\rawdems"


            ### 25/12/2024
            self.url_storage =  r"C:\Users\Usuario\OneDrive - INTERCONEXION ELECTRICA S.A. E.S.P\01_CC_DOC\1_Raw_Data\06_InputGeologico\02_Alos_Palsar\dem_COL\rawdemsv2"
            self._path_unzip = r"C:\Users\Usuario\OneDrive - INTERCONEXION ELECTRICA S.A. E.S.P\01_CC_DOC\1_Raw_Data\06_InputGeologico\02_Alos_Palsar\dem_COL\complementary_data"

        elif pais == 'Peru':
            
            aoi =  'POLYGON ((-69.95229270018514 -4.212843811278671, -71.90637525319012 -4.514425884781545, -72.81892748441774 -5.1022480054132995, -73.23770590980064 -6.03599070174865, -73.13595841057852 -6.509668182132288, -73.7113375736005 -6.83998477224466, -73.98305528005822 -7.534307291025313, -72.94101911682198 -8.985366711315534, -73.20419598579744 -9.40739713843656, -72.34576202900215 -9.501642289845018, -72.18090243365799 -10.000440534147966, -71.2217655032374 -9.96983890195003, -70.49674151607366 -9.42421843226325, -70.61377889127655 -10.999843911762069, -69.57673983008709 -10.941270314907877, -68.65312196219043 -12.500360177812018, -69.06708412870356 -13.673060265223821, -68.82796651662954 -14.220354312125702, -69.35754244666342 -14.812856240120418, -69.12022149372416 -15.259595829832985, -69.33151035837065 -15.537898499849259, -69.89493838459637 -15.29500134590016, -69.74381009095151 -15.722756099100911, -70.03422512326418 -15.722546664173198, -69.49437013769185 -16.207980362257718, -68.962664129206 -16.190680771606946, -68.99668153906408 -16.65592187933007, -69.64469389381551 -17.280664034564246, -69.46817712261996 -17.50155780829608, -69.82265437586163 -17.686084118756025, -69.85800721914698 -18.164907606520522, -70.37726886623066 -18.350538192160464, -71.38143155399268 -17.7072656989987, -71.50936009730425 -17.268528187724588, -75.14044554324191 -15.40708414592234, -75.91680613863794 -14.658824584258253, -76.39651461064024 -13.908682226302247, -76.20847167768818 -13.38090913711997, -77.64838430287547 -11.302658011844867, -78.98401559071952 -8.215382472704066, -79.98133299890071 -6.7516236207145015, -81.11153310535373 -6.06186415519027, -80.85223067950093 -5.63723935478905, -81.32815208902707 -4.670941424591032, -80.31521591778545 -3.3909430223324266, -80.12670867175008 -3.894722257000666, -80.4463540760816 -3.997156212897664, -80.44880158011004 -4.447417554144622, -79.62767028523055 -4.439241922902995, -79.01510301908178 -5.015342260324535, -78.34661777091115 -3.38081060884535, -76.63273111370462 -2.589884153665007, -75.56009986482991 -1.5615801936070606, -75.18887163957552 -0.9710666565803198, -75.61086647956465 -0.113718211942843, -75.18349290834394 -0.0386903793641273, -73.62031932913594 -1.259894770994648, -72.93556210006689 -2.4551457110060877, -72.1545350411389 -2.477752504112012, -71.73076706375679 -2.1403663946599005, -70.08291185394464 -2.650987573158221, -70.7130300100138 -3.794169515008362, -69.95229270018514 -4.212843811278671))'

            self.url_storage = r"C:\Users\Usuario\OneDrive - INTERCONEXION ELECTRICA S.A. E.S.P\01_CC_DOC\1_Raw_Data\06_InputGeologico\02_Alos_Palsar\dem_REP"
            
    
        opts = {
                'platform': asf.PLATFORM.ALOS,
         ##        'start': '2010-01-01T00:00:00Z',          Se removueven la limitacion de la fecha. para la descarga rawdemsv2
         ##        'end': '2011-02-01T23:59:59Z',            Dejamos que los mismo filtros se encarguen de las fechas mas pertinenes.
                'beamMode': 'FBS',
                'processingLevel':'RTC_HI_RES'
            }

        self.results = asf.geo_search(intersectsWith=aoi, **opts)


        ## This was before we considere valuable to download the files complete with images and all.
        ## I consider is valuable now that we are going to downlaod all the tiles to have some support of the metadata each tile comes with.
        ### self._path_unzip = r"C:\Users\Usuario\OneDrive - INTERCONEXION ELECTRICA S.A. E.S.P\01_CC_DOC\1_Raw_Data\06_InputGeologico\02_Alos_Palsar\folders_to_delete"

        self._already_download = os.listdir(self.url_storage)
        
        pattern_zip = r"(.+)\.zip$"
        self._re_zip = re.compile(pattern = pattern_zip)
        self._only_dem = re.compile(".+\.dem.tif$")


    def _GettingFilesReady(self,):
        
        alos_palsar_col =  pd.DataFrame([( resul.properties['orbit'], 
                                          resul.properties['pathNumber'],
                                          resul.properties['frameNumber'],
                                          resul.properties['flightDirection'] ,
                                          resul.properties['fileName'],
                                          resul.properties['groupID'],
                                          resul.properties['startTime'],
                                         resul.properties['offNadirAngle'])
                                         for resul in self.results  ]).rename(columns = {0:'orbit',
                                                                                         1:'pathNumber' ,
                                                                                         2:'frameNumber',
                                                                                         3:'flightDirection',
                                                                                         4:'fileName',
                                                                                         5: 'groupID',
                                                                                         6:'startTime',
                                                                                         7:'offNadirAngle'
                                
                                                                                        })

       
        dict_balance = alos_palsar_col.offNadirAngle.value_counts().to_dict()

        for idx,(angle,count) in enumerate(dict_balance.items(),start = 1):

            print(f"{idx} - For dir angle {angle} , theres {count} filter")

        print("Filtering only 34.3")
        alos_palsar_col = alos_palsar_col[alos_palsar_col['offNadirAngle'] == 34.3]

        print(f"Total files found = {alos_palsar_col.shape[0]}")

        alos_palsar_col['startTime'] =  pd.to_datetime(alos_palsar_col.startTime )
        
        sorting_v2 = ['startTime', 'pathNumber' , 'frameNumber',]
        
        self.alos_palsar_col = ( alos_palsar_col.sort_values(by = sorting_v2 ,ascending = [False,False,True])
                                                 .drop_duplicates(subset = ['pathNumber', 'frameNumber'])
                                  )
        
        self.alos_palsar_col = self.alos_palsar_col.loc[self.alos_palsar_col.flightDirection == 'ASCENDING']

        index_planchas = self.alos_palsar_col.index
        file_tuple = []
        okey = 0
        
        for idx in index_planchas:
            
            name_file = self.results[idx].properties['fileName']
            url_file = self.results[idx].properties['url']
            match_zip = self._re_zip.match(name_file).group(1)

            dem_name = match_zip + ".dem.tif"
            
            if dem_name in self._already_download:
                
                print(f"{dem_name} is already download ! , pass")
                okey += 1
                continue
            else:
                tuple_code = (name_file,url_file)
                file_tuple.append(tuple_code)

        print(f"Theres {okey} files that are already downloaded.\nRemaining files = {len(file_tuple)}")
        
        return file_tuple
    
    def DownloadingFiles(self,):


        print(f"Files been store at {self.url_storage}\n")
        print(f"Complementary data being store at  {self._path_unzip}\n")
        
        current_time = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        print(f"Starting download at :{current_time}\n")
        
        list_files = self._GettingFilesReady( )

        total = len(list_files)
        
        # storing the current time in the variable
        c = datetime.now()
        progress = 0
        
        for name_zip,url_tile_colombia in list_files:
            
            print(f"Downloading {name_zip} [...]")
            asf.download_url(url = url_tile_colombia, 
                             path = self._path_unzip  ,
                             session = self.user_pass_session
                            )
            
            print("\tComplete Downloading !")
            print("\tOpening Folder ...")

            os.chdir(self._path_unzip)
            
            # Replace 'your_file.zip' with the actual filename
            with zipfile.ZipFile(name_zip, 'r') as zip_ref:
                # Extract the first file (modify if needed)
                zip_ref.extractall()
                ##  extracted_file = zip_ref.namelist()
            time.sleep(10)
            
            ##'AP_26731_FBS_F3440_RT1.zip'
            os.remove(name_zip)
            name_folder = self._re_zip.match(name_zip).group(1)   ##remove the .zip
            sub_folder = os.path.join(self._path_unzip , name_folder)
            extracted_file = os.listdir(sub_folder)
            
            os.chdir(sub_folder)
            for index in range(len(extracted_file)):

                current_name = extracted_file[index]
                match = self._only_dem.match(string= current_name )
                
                if match:
                    
                    os.rename( os.path.join(sub_folder, current_name ), os.path.join(self.url_storage ,current_name)) 
                    print(f"\t\tDownload succefully for {name_zip}\n")
                    break
                    
                else:
                    #os.remove(current_name)   Una guvonada que no volvi hacer
                    #time.sleep(4)
                    continue

            progress += 1
            time.sleep(5)

            if progress % 40 == 0:
                print(f"\tprogress is {progress}")

            else:
                continue


#token = "eyJ0eXAiOiJKV1QiLCJvcmlnaW4iOiJFYXJ0aGRhdGEgTG9naW4iLCJzaWciOiJlZGxqd3RwdWJrZXlfb3BzIiwiYWxnIjoiUlMyNTYifQ.eyJ0eXBlIjoiVXNlciIsInVpZCI6Imdlb2lzYXZvbGttYXIiLCJleHAiOjE3MjI0ODE2MTAsImlhdCI6MTcxNzI5NzYxMCwiaXNzIjoiRWFydGhkYXRhIExvZ2luIn0.VIm_wHopgfnU_7gB7-SKod1d0Ban_DcU-H_KHEYN738bhVcQ7ehTCRH16H53RuUUepN2w-dExadD2wjGveWYIYcN7eQ55DyudjzkhKDFE0IOrIPNG0T7o7iFkro3tZWywTNra_VmHeb_ULehoCh8G71_aukbLCVRxL9gPXLVrHSimfFfK9op2IDSz96HT_hhAds2wM5AZnrqsFByoXl7OYkYdF45bFe_4rqt-D1UJUa0EA0QMlbueFCTKe1UsAbAAQyJyhAh6bmmqSDLH9JOeB88wzbim5Ujni8ZMC0nQRUgm6bXB7n6AyIiW8449arv5IlRH17_G6y3iMGzoXt41g"
