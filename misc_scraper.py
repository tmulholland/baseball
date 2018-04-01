import subprocess
import time
import zipfile


## seamheads
url = 'http://www.seamheads.com/ballparks/Seamheads_Ballparks_Database_2017.zip'


## make directory to place the ballpark database
subprocess.call(['mkdir', '-p', 'data/misc/Seamheads'])

## wget the ballpark database from retrosheet
subprocess.call(['wget', '-P', 'data/misc/Seamheads', url])

## unzip the ballpark database
zip_ref = zipfile.ZipFile('data/misc/Seamheads/Seamheads_Ballparks_Database_2017.zip', 'r')
zip_ref.extractall('data/misc/Seamheads/')
zip_ref.close()

## pause for half a second to avoid overloading server
time.sleep(0.5)
