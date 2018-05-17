# Major League Baseball Analysis

Scrape and analyze MLB data from several sources:
* PITCHf/x
* retrosheet game logs
* retrosheet event logs
* Seamheads ballparks database
* Cruchtime player name mapping

Python analysis of data involves seamless combination of data sources. 
Example:
`
## game log class instance
GL = analyzer.GameLogs(2017)
## PITCHf/x class instance
Pfx = analyzer.PITCHfx(2017)
## overload addition operator to merge on game id
Pfx_GL = Pfx+GL
`

## Getting Started

This project has been tested on ubuntu linux with python 2.7

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

```
Give examples
```

### Installing

A step by step series of examples that tell you have to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone who's code was used
* Inspiration
* etc
