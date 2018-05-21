# Major League Baseball Analysis

Scrape and analyze MLB data from several sources:
* PITCHf/x and Statcast/TrackMan
* retrosheet game logs
* retrosheet event logs
* Seamheads ballparks database
* Cruchtime player name mapping

![alt text](https://raw.githubusercontent.com/tmulholland/baseball/master/figs/jon-gray-2017.png)

Python analysis of data involves seamless combination of data sources. 
Example:
```
## game log class instance
GL = analyzer.GameLogs(2017)

## PITCHf/x class instance
Pfx = analyzer.PITCHfx(2017)

## overload addition operator to merge on game id
Pfx_GL = Pfx+GL
```

## Getting Started

This project has been tested on ubuntu linux with python 2.7

Please see example_notebook.ipynb for a quick dive

### Prerequisites

python packages to install:  
zipfile  
pandas  
numpy  
pytz  
geopy  
timezonefinder  
urllib2  
BeautifulSoup  

