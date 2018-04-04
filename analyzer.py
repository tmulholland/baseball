import pandas as pd
import numpy as np
import pytz
from geopy.distance import vincenty
from timezonefinder import TimezoneFinder
import datetime as dt
import copy

class Abstract(object):
    ## abstract class to handle some overloading operations

    def __add__(self, other):

        Copy = copy.deepcopy(self)
        
        ## if both are instances of the same class, just concat the dataframes
        if self.__class__ == other.__class__:
            Copy.df = pd.concat([self.df, other.df])

        ## else if the self or other class is an instance of Parks, then merge in park_id
        elif 'Parks' in str(self.__class__) or 'Parks' in str(other.__class__):
            Copy.df = self.df.merge(other.df,on='park_id')

        ## PitchFx, EventLogs, and EventInfo can all be merged on 'identifier' 
        else:
            Copy.df = self.df.merge(other.df,on='identifier')

        return Copy

    def add_time_zone(self,):

        try:
            assert ('Longitude' in self.df.columns and 'Latitude' in self.df.columns)
        except:
            print "ERROR: must have Longitude and Latitude as columns to compute time zone"
            
            return 0

        ## time zone finder
        tf = TimezoneFinder()

        self.df['time_zone'] = self.df.apply(lambda x: tf.timezone_at(lng=x.Longitude, lat=x.Latitude), 
                                             axis=1)
        

    def get_unix_from_time_stamp(self,x):
        
        ## time stamp of first pitch
        stamp = dt.datetime.strptime(x.time_stamp,'%Y%m%d%I:%M%p')
        ## start of unix time
        start = dt.datetime(1970,1,1) 
    
        ## park time zone
        prk_tz = pytz.timezone(x.time_zone)
        ## gmt time zone (for unix stamp)
        gmt_tz = pytz.timezone("GMT")
    
        start = gmt_tz.localize(start)
        stamp = prk_tz.localize(stamp)
    
        return int((stamp-start).total_seconds())

    def add_time_stamp(self,):

        self.df['time_stamp'] = self.df.apply(lambda x: str(x.date)+str(x.starttime),axis=1)

        if 'time_zone' not in self.df.columns:
            self.add_time_zone()

        self.df['unix_time_start'] = self.df.apply(lambda x: self.get_unix_from_time_stamp(x), axis=1)

        self.df['unix_time_end'] = self.df.unix_time_start + self.df.game_length_min*60

        self.df['year'] = self.df.apply(lambda x: x.time_stamp[:4],axis=1)
        self.df['month'] = self.df.apply(lambda x: x.time_stamp[4:6],axis=1)
        self.df['day'] = self.df.apply(lambda x: x.time_stamp[6:8],axis=1)

class Parks(Abstract):

    def __init__(self,):
        
        ## parks data from Seamheads
        self.df = pd.read_csv('data/misc/Seamheads/Parks.csv')

        ## use 'park_id' convention to match GameLogs
        self.df['park_id']=self.df['PARKID']

class GameLogs(Abstract):

    def __init__(self, year):

        ## each game log class instance corresponds to a full season
        ## if you want to combine seasons, use the overloaded addition operator like so:
        #########################
        ## GL2016 = GameLogs(2016)
        ## GL2017 = GameLogs(2017)
        ## GL = GL2016+GL2017
        #########################
        self.year = year

        ## see glfields.txt for more infomation on column names
        self.column_names = ['date','game_number','dow',
                             'away_team','away_team_league','away_team_game_num',
                             'home_team','home_team_league','home_team_game_num',
                             'away_team_score','home_team_score','game_length_outs',
                             'day_or_night','completion_info','forfeit_info','protest_info',
                             'park_id','attendance','game_length_min',
                             'away_line_scores','home_line_scores',
                         
                             'away_ABs','away_Hs','away_2Bs','away_3Bs','away_HRs','away_RBIs',
                             'away_SHs','away_SFs','away_HBPs','away_BBs','away_IBBs','away_SOs',
                             'away_SBs','away_CSs','away_GIDPs','away_CIs','away_LOBs',
                             
                             'away_PUs','away_IERs','away_TERs','away_WPs','away_BALKs',
                             'away_POs','away_As','away_Es','away_PBs','away_DPs','away_TPs',
                
                             'home_ABs','home_Hs','home_2Bs','home_3Bs','home_HRs','home_RBIs',
                             'home_SHs','home_SFs','home_HBPs','home_BBs','home_IBBs','home_SOs',
                             'home_SBs','home_CSs','home_GIDPs','home_CIs','home_LOBs',
                
                             'home_PUs','home_IERs','home_TERs','home_WPs','home_BALKs',
                             'home_POs','home_As','home_Es','home_PBs','home_DPs','home_TPs',
                
                             'ump_HP_id','ump_HP_name','ump_1B_id','ump_1B_name',
                             'ump_2B_id','ump_2B_name','ump_3B_id','ump_3B_name',
                             'ump_LF_id','ump_LF_name','ump_RF_id','ump_RF_name',

                             'away_manager_id','away_manager_name',
                             'home_manager_id','home_manager_name',
                             'winning_pitcher_id','winning_pitcher_name',
                             'losing_pitcher_id','losing_pitcher_name',
                             'saving_pitcher_id','saving_pitcher_name',
                             'gm_winning_RBI_batter_id','gm_winning_RBI_batter_name',
                             'away_starting_pitcher_id','away_starting_pitcher_name',
                             'home_starting_pitcher_id','home_starting_pitcher_name', 
                
                             'away_batter1_name','away_batter1_id','away_batter1_pos',
                             'away_batter2_name','away_batter2_id','away_batter2_pos',
                             'away_batter3_name','away_batter3_id','away_batter3_pos',
                             'away_batter4_name','away_batter4_id','away_batter4_pos',
                             'away_batter5_name','away_batter5_id','away_batter5_pos',
                             'away_batter6_name','away_batter6_id','away_batter6_pos',
                             'away_batter7_name','away_batter7_id','away_batter7_pos',
                             'away_batter8_name','away_batter8_id','away_batter8_pos',
                             'away_batter9_name','away_batter9_id','away_batter9_pos',

                             'home_batter1_name','home_batter1_id','home_batter1_pos',
                             'home_batter2_name','home_batter2_id','home_batter2_pos',
                             'home_batter3_name','home_batter3_id','home_batter3_pos',
                             'home_batter4_name','home_batter4_id','home_batter4_pos',
                             'home_batter5_name','home_batter5_id','home_batter5_pos',
                             'home_batter6_name','home_batter6_id','home_batter6_pos',
                             'home_batter7_name','home_batter7_id','home_batter7_pos',
                             'home_batter8_name','home_batter8_id','home_batter8_pos',
                             'home_batter9_name','home_batter9_id','home_batter9_pos',

                             'extra_info','acquisition_info'
                         ]

        ## pandas data frame from corresponding year
        self.df = pd.read_csv('data/'+str(self.year)+'/GL'+str(self.year)+'.TXT', names=self.column_names)

        ## identifier common to event dataframes
        self.df['identifier'] = self.df.apply(lambda x: 
                                              str(x.date)+x.home_team+x.away_team+str(x.game_number),
                                              axis=1) 




class EventInfo(Abstract):

    def __init__(self, year):

        ## output from event_parser.csh
        self.year = year

        ## pandas data frame from corresponding year
        self.df = pd.read_csv('data/'+str(self.year)+'/event_info_'+str(self.year)+'.csv')

        ## identifier common to event dataframes
        self.df['identifier'] = self.df.apply(lambda x: 
                                              x.date.replace("/","")+x.hometeam+x.visteam+str(x.number), 
                                              axis=1)

        ## keep columns not already in EventLogs
        self.df = self.df[['starttime','usedh','howscored','pitches','oscorer',
                           'temp','winddir','windspeed','fieldcond','precip',
                           'sky','timeofgame','identifier']]
