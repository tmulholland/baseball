import subprocess
import time
import zipfile

for year in range(1871,2018):
    
    year_str = str(year)

    ## retrosheet url of game logs for year
    url = 'http://www.retrosheet.org/gamelogs/gl'+year_str+'.zip'

    ## make directory to place the game logs
    subprocess.call(['mkdir', '-p', 'data/'+year_str])

    ## wget the game logs from retrosheet
    subprocess.call(['wget', '-P', 'data/'+year_str, url])

    ## unzip the game logs
    zip_ref = zipfile.ZipFile('data/'+year_str+'/gl'+year_str+'.zip', 'r')
    zip_ref.extractall('data/'+year_str)
    zip_ref.close()

    ## pause for half a second to avoid overloading server
    time.sleep(0.5)
