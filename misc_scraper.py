import subprocess
import time
import zipfile

year = str(2017)

###########################
####### seamheads #########
###########################
url = 'http://www.seamheads.com/ballparks/Seamheads_Ballparks_Database_'+year+'.zip'

## make directory to place the ballpark database
subprocess.call(['mkdir', '-p', 'data/misc/Seamheads'])

## wget the ballpark database from retrosheet
subprocess.call(['wget', '-P', 'data/misc/Seamheads', url])

## unzip the ballpark database
zip_ref = zipfile.ZipFile('data/misc/Seamheads/Seamheads_Ballparks_Database_'+year+'.zip', 'r')
zip_ref.extractall('data/misc/Seamheads/')
zip_ref.close()

###########################
####### crunchtime ########
###########################

url = 'http://crunchtimebaseball.com/master.csv'

## make directory to place the map of mlb player names
subprocess.call(['mkdir', '-p', 'data/misc/crunchtime'])

## wget the ballpark database from retrosheet
subprocess.call(['wget', '-P', 'data/misc/crunchtime', url])
