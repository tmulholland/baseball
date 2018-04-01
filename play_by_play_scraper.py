import subprocess
import time
import zipfile

for year in range(1921,2018):
    
    year_str = str(year)

    ## missing two seasons
    if year == 1923 or year == 1924:
        continue
    
    ## retrosheet url of play-by-play logs for year
    url = 'http://www.retrosheet.org/events/'+year_str+'eve.zip'

    ## make directory to place the play-by-play logs
    subprocess.call(['mkdir', '-p', 'data/'+year_str])

    ## wget the play-by-play logs from retrosheet
    subprocess.call(['wget', '-P', 'data/'+year_str, url])

    ## unzip the play-by-play logs
    zip_ref = zipfile.ZipFile('data/'+year_str+'/'+year_str+'eve.zip', 'r')
    zip_ref.extractall('data/'+year_str)
    zip_ref.close()

    ## pause for half a second to avoid overloading server
    time.sleep(0.5)

    subprocess.call(['tcsh','event_parser.csh',year_str])
