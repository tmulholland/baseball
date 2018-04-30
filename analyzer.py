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

        ## else if the self or other class is an instane of PlayerMap, then merge on player id
        elif 'PlayerMap' in str(self.__class__) or 'PlayerMap' in str(other.__class__):
            ## PITCHf/x uses mlb id
            if 'PITCHfx' in str(self.__class__) or 'PITCHfx' in str(other.__class__):
                Copy.df = self.df.merge(other.df,on='mlb_id')
            ## else use retro_id 
            else:
                Copy.df = self.df.merge(other.df,on='retro_id')

        ## PitchFx, EventLogs, and EventInfo can all be merged on 'identifier' 
        else:
            Copy.df = self.df.merge(other.df,on='identifier')

        return Copy

    def __radd__(self, other):

        if other == 0:
            return self
        else:
            return self.__add__(other)

    def slim_frame(self,cols_to_keep):
        """List of columns to keep, remove others"""

        tmp_df = self.df[cols_to_keep]
        del self.df

        self.df = tmp_df

    def filter_by_pitch_type(self, pitch_type):
        """
        Subset the dataframe by pitch types
        """

        ## fastball, four-seam, two-seam, cutter, sinker, split-fingered
        fastballs = ['FA', 'FF', 'FT', 'FC', 'FS', 'SI', 'SF']

        ## slider, changeup, curveball, knuckle-curve, knuckle ball, eephus
        off_speed = ['SL', 'CH', 'CB', 'CU', 'KC', 'KN', 'EP']

        if 'fastball' in pitch_type.lower():
            tmp_df = self.df[ self.df.pitch_type.isin(fastballs) ]
        elif 'breaking' in pitch_type.lower() or 'off' in pitch_type.lower():
            tmp_df = self.df[ self.df.pitch_type.isin(off_speed) ]
        elif type(pitch_type) is list:
            tmp_df = self.df[ self.df.pitch_type.isin(pitch_type) ]
        else:
            tmp_df = self.df[ self.df.pitch_type==pitch_type ]

        del self.df
        self.df = tmp_df

    def filter_by_team(self, team_name, keep=True):
        """
        Subset the dataframe by team (home and away)
        keep==True (default) means keep only team,
        otherwise, keep all other teams
        """

        mask = (self.df.home_team==team_name) | (self.df.away_team==team_name)

        if keep:
            tmp_df = self.df[mask]
        else:
            tmp_df = self.df[~mask]

        del self.df
        self.df = tmp_df

    def filter_by_starting_pitchers(self, pitcher_ids, keep=True):
        """
        Args:
            starting pitcher retro id or list of starting pitcher retro ids

        Subset the dataframe by starting pitcher ids (home and away)
        keep==True (default) means keep only starting pitchers,
        otherwise, keep all other pitchers
        """

        ## convert to list if given one pitcher name as a string
        if type(pitcher_ids) is str:
            pitcher_ids = [pitcher_ids]

        ## initialize mask of Falses
        mask = pd.Series([False]*self.df.shape[0])

        for pitcher in pitcher_ids:
            mask = mask | ( ((self.df.home_starting_pitcher_id==pitcher) & 
                             (self.df.retro_id==pitcher)) |
                            ((self.df.away_starting_pitcher_id==pitcher) &
                             (self.df.retro_id==pitcher)) )

        if keep:
            tmp_df = self.df[mask]
        else:
            tmp_df = self.df[~mask]

        del self.df
        self.df = tmp_df

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

        self.df['time_stamp'] = self.df.apply(lambda x: str(x.date)+str(x.starttime),
                                              axis=1)

        if 'time_zone' not in self.df.columns:
            self.add_time_zone()

        self.df['unix_time_start'] = self.df.apply(lambda x: self.get_unix_from_time_stamp(x), 
                                                   axis=1)

        self.df['unix_time_end'] = self.df.unix_time_start + self.df.game_length_min*60

        self.df['year'] = self.df.apply(lambda x: x.time_stamp[:4],axis=1)
        self.df['month'] = self.df.apply(lambda x: x.time_stamp[4:6],axis=1)
        self.df['day'] = self.df.apply(lambda x: x.time_stamp[6:8],axis=1)

    def add_travel_columns(self,knots=500):

        home_dist_last_travel_game = []
        away_dist_last_travel_game = []

        for index, row in self.df.iterrows():
            
            home_games = self.df[self.df.home_team==row.home_team]
            distance = 0.0
            for game in xrange(row.home_team_game_num-1,0,-1):
                tmp_df = self.df[((self.df.home_team_game_num==game) &
                                  (self.df.home_team==row.home_team)) | 
                                 ((self.df.away_team_game_num==game) &
                                  (self.df.away_team==row.home_team)) ]
                
                if tmp_df.park_id.values[0] != row.park_id:
                    distance = vincenty((row.Latitude, row.Longitude), 
                                            (tmp_df.Latitude.values[0], 
                                             tmp_df.Longitude.values[0])).miles
                    break
ape
            home_dist_last_travel_game.append(distance)

            away_games = self.df[self.df.away_team==row.away_team]
            distance = 0.0
            for game in xrange(row.away_team_game_num-1,0,-1):
                tmp_df = self.df[((self.df.home_team_game_num==game) &
                                  (self.df.home_team==row.away_team)) | 
                                 ((self.df.away_team_game_num==game) &
                                  (self.df.away_team==row.away_team)) ]
                if tmp_df.park_id.values[0] != row.park_id:
                    distance = vincenty((row.Latitude, row.Longitude), 
                                        (tmp_df.Latitude.values[0], 
                                         tmp_df.Longitude.values[0])).miles
                    break
            away_dist_last_travel_game.append(distance)

        self.df['home_distance_since_last_travel_game'] = home_dist_last_travel_game
        self.df['away_distance_since_last_travel_game'] = away_dist_last_travel_game

        ## 1 knot = 1.15 mph
        self.df['home_travel_time_since_last_travel_game'] = np.array(home_dist_last_travel_game)/(knots*1.15)
        self.df['away_travel_time_since_last_travel_game'] = np.array(away_dist_last_travel_game)/(knots*1.15)

    def add_jet_lag(self,):
        """For each 24hrs, jet lag is reduced by 1 hr until there is no jet lag"""

        def direction(tdiff):
            """ Computes direction of time difference travel 

            Args:
                 Time difference (positive, negative, or zero)
            
            Returns:
                 Westward, Eastward, or '' (for no travel)
            """

            if tdiff<0:
                return 'Eastward'
            if tdiff>0:
                return 'Westward'
            return ''

        def get_lag_direction(x, home_or_away):
            """Takes dataframe and home/away status and returns direction of jet lag"""
            
            if x[home_or_away+'_jet_lag']==0:
                return ''
            if x[home_or_away+'_jet_lag']==x[home_or_away+'_time_zones_crossed_in_last_24hrs']:
                return x[home_or_away+'_time_zones_crossed_direction_in_last_24hrs']
            if x[home_or_away+'_jet_lag']==x[home_or_away+'_time_zones_crossed_in_last_48hrs']-1:
                return x[home_or_away+'_time_zones_crossed_direction_in_last_48hrs']
            if x[home_or_away+'_jet_lag']==x[home_or_away+'_time_zones_crossed_in_last_72hrs']-2:
                return x[home_or_away+'_time_zones_crossed_direction_in_last_72hrs']


        home_time_zones_crossed_in_last_24hrs = []
        away_time_zones_crossed_in_last_24hrs = []
        home_time_zones_crossed_in_last_48hrs = []
        away_time_zones_crossed_in_last_48hrs = []
        home_time_zones_crossed_in_last_72hrs = []
        away_time_zones_crossed_in_last_72hrs = []

        home_time_zones_crossed_direction_in_last_24hrs = []
        away_time_zones_crossed_direction_in_last_24hrs = []
        home_time_zones_crossed_direction_in_last_48hrs = []
        away_time_zones_crossed_direction_in_last_48hrs = []
        home_time_zones_crossed_direction_in_last_72hrs = []
        away_time_zones_crossed_direction_in_last_72hrs = []

        home_dist_traveled_in_last_24hrs = []
        away_dist_traveled_in_last_24hrs = []
        
        for index, row in self.df.iterrows():
            home_team_games = self.df[(self.df.home_team==row.home_team) | 
                                      (self.df.away_team==row.home_team)]
            away_team_games = self.df[(self.df.home_team==row.away_team) | 
                                      (self.df.away_team==row.away_team)]

            home_hours_since = (row.unix_time_start 
                                - home_team_games.unix_time_end)/(60.*60)
            away_hours_since = (row.unix_time_start 
                                - away_team_games.unix_time_end)/(60.*60)

            home_time_zones_24 = home_team_games[(home_hours_since>0) & 
                                                 (home_hours_since<24+
                                                  row.home_travel_time_since_last_travel_game)].time_zone.unique()
            away_time_zones_24 = away_team_games[(away_hours_since>0) & 
                                                 (away_hours_since<24+
                                                  row.away_travel_time_since_last_travel_game)].time_zone.unique()
            home_time_zones_48 = home_team_games[(home_hours_since>0) & 
                                                 (home_hours_since<48+
                                                  row.home_travel_time_since_last_travel_game)].time_zone.unique()
            away_time_zones_48 = away_team_games[(away_hours_since>0) & 
                                                 (away_hours_since<48+
                                                  row.away_travel_time_since_last_travel_game)].time_zone.unique()
            home_time_zones_72 = home_team_games[(home_hours_since>0) & 
                                                 (home_hours_since<72+
                                                  row.home_travel_time_since_last_travel_game)].time_zone.unique()
            away_time_zones_72 = away_team_games[(away_hours_since>0) & 
                                                 (away_hours_since<72+
                                                  row.away_travel_time_since_last_travel_game)].time_zone.unique()

            home_lats_24 = home_team_games[(home_hours_since>0) & (home_hours_since<24)].Latitude.unique()
            home_lngs_24 = home_team_games[(home_hours_since>0) & (home_hours_since<24)].Longitude.unique()

            away_lats_24 = away_team_games[(away_hours_since>0) & (away_hours_since<24)].Latitude.unique()
            away_lngs_24 = away_team_games[(away_hours_since>0) & (away_hours_since<24)].Longitude.unique()

            tz = pytz.timezone(row.time_zone)
            game_dt = dt.datetime.fromtimestamp(row.unix_time_start)
    
            home_tzs_24 = [pytz.timezone(tz_24) for tz_24 in home_time_zones_24]
            away_tzs_24 = [pytz.timezone(tz_24) for tz_24 in away_time_zones_24]
            home_tzs_48 = [pytz.timezone(tz_48) for tz_48 in home_time_zones_48]
            away_tzs_48 = [pytz.timezone(tz_48) for tz_48 in away_time_zones_48]
            home_tzs_72 = [pytz.timezone(tz_72) for tz_72 in home_time_zones_72]
            away_tzs_72 = [pytz.timezone(tz_72) for tz_72 in away_time_zones_72]

            home_dist_24 = [vincenty(coords, (row.Latitude, row.Longitude)).miles for coords in zip(home_lats_24, home_lngs_24)]
            away_dist_24 = [vincenty(coords, (row.Latitude, row.Longitude)).miles for coords in zip(away_lats_24, away_lngs_24)]
            
            home_change_24 = [abs((tz.localize(game_dt)-tz_24.localize(game_dt)).total_seconds()/(60.*60)) for tz_24 in home_tzs_24]
            away_change_24 = [abs((tz.localize(game_dt)-tz_24.localize(game_dt)).total_seconds()/(60.*60)) for tz_24 in away_tzs_24]
            home_dir_24 = [direction((tz.localize(game_dt)-tz_24.localize(game_dt)).total_seconds()) for tz_24 in home_tzs_24]
            away_dir_24 = [direction((tz.localize(game_dt)-tz_24.localize(game_dt)).total_seconds()) for tz_24 in away_tzs_24]

            home_change_48 = [abs((tz.localize(game_dt)-tz_48.localize(game_dt)).total_seconds()/(60.*60)) for tz_48 in home_tzs_48]
            away_change_48 = [abs((tz.localize(game_dt)-tz_48.localize(game_dt)).total_seconds()/(60.*60)) for tz_48 in away_tzs_48]
            home_dir_48 = [direction((tz.localize(game_dt)-tz_48.localize(game_dt)).total_seconds()) for tz_48 in home_tzs_48]
            away_dir_48 = [direction((tz.localize(game_dt)-tz_48.localize(game_dt)).total_seconds()) for tz_48 in away_tzs_48]
    
            home_change_72 = [abs((tz.localize(game_dt)-tz_72.localize(game_dt)).total_seconds()/(60.*60)) for tz_72 in home_tzs_72]
            away_change_72 = [abs((tz.localize(game_dt)-tz_72.localize(game_dt)).total_seconds()/(60.*60)) for tz_72 in away_tzs_72]
            home_dir_72 = [direction((tz.localize(game_dt)-tz_72.localize(game_dt)).total_seconds()) for tz_72 in home_tzs_72]
            away_dir_72 = [direction((tz.localize(game_dt)-tz_72.localize(game_dt)).total_seconds()) for tz_72 in away_tzs_72]
    

            home_time_zones_crossed_in_last_24hrs.append( max(home_change_24+[0.0]))
            away_time_zones_crossed_in_last_24hrs.append( max(away_change_24+[0.0]))
            home_time_zones_crossed_in_last_48hrs.append( max(home_change_48+[0.0]))
            away_time_zones_crossed_in_last_48hrs.append( max(away_change_48+[0.0]))
            home_time_zones_crossed_in_last_72hrs.append( max(home_change_72+[0.0]))
            away_time_zones_crossed_in_last_72hrs.append( max(away_change_72+[0.0]))
                
            home_time_zones_crossed_direction_in_last_24hrs.append( max(home_dir_24+['']))
            away_time_zones_crossed_direction_in_last_24hrs.append( max(away_dir_24+['']))
            home_time_zones_crossed_direction_in_last_48hrs.append( max(home_dir_48+['']))
            away_time_zones_crossed_direction_in_last_48hrs.append( max(away_dir_48+['']))
            home_time_zones_crossed_direction_in_last_72hrs.append( max(home_dir_72+['']))
            away_time_zones_crossed_direction_in_last_72hrs.append( max(away_dir_72+['']))

            home_dist_traveled_in_last_24hrs.append( max(home_dist_24+[0.0]))
            away_dist_traveled_in_last_24hrs.append( max(away_dist_24+[0.0]))
            
        self.df['home_time_zones_crossed_in_last_24hrs'] = home_time_zones_crossed_in_last_24hrs
        self.df['away_time_zones_crossed_in_last_24hrs'] = away_time_zones_crossed_in_last_24hrs
        self.df['home_time_zones_crossed_in_last_48hrs'] = home_time_zones_crossed_in_last_48hrs
        self.df['away_time_zones_crossed_in_last_48hrs'] = away_time_zones_crossed_in_last_48hrs
        self.df['home_time_zones_crossed_in_last_72hrs'] = home_time_zones_crossed_in_last_72hrs
        self.df['away_time_zones_crossed_in_last_72hrs'] = away_time_zones_crossed_in_last_72hrs

        self.df['home_time_zones_crossed_direction_in_last_24hrs'] = home_time_zones_crossed_direction_in_last_24hrs
        self.df['away_time_zones_crossed_direction_in_last_24hrs'] = away_time_zones_crossed_direction_in_last_24hrs
        self.df['home_time_zones_crossed_direction_in_last_48hrs'] = home_time_zones_crossed_direction_in_last_48hrs
        self.df['away_time_zones_crossed_direction_in_last_48hrs'] = away_time_zones_crossed_direction_in_last_48hrs
        self.df['home_time_zones_crossed_direction_in_last_72hrs'] = home_time_zones_crossed_direction_in_last_72hrs
        self.df['away_time_zones_crossed_direction_in_last_72hrs'] = away_time_zones_crossed_direction_in_last_72hrs    

        self.df['home_dist_traveled_in_last_24hrs'] = home_dist_traveled_in_last_24hrs
        self.df['away_dist_traveled_in_last_24hrs'] = away_dist_traveled_in_last_24hrs
    
        self.df['home_jet_lag'] = self.df.apply(lambda x: max(x.home_time_zones_crossed_in_last_24hrs, 
                                                              x.home_time_zones_crossed_in_last_48hrs-1,
                                                              x.home_time_zones_crossed_in_last_72hrs-2,
                                                          ), axis=1)
        self.df['home_jet_lag_direction'] = self.df.apply(lambda x: get_lag_direction(x,'home'), axis=1)

        self.df['away_jet_lag'] = self.df.apply(lambda x: max(x.away_time_zones_crossed_in_last_24hrs, 
                                                              x.away_time_zones_crossed_in_last_48hrs-1,
                                                              x.away_time_zones_crossed_in_last_72hrs-2,
                                                          ), axis=1)
        self.df['away_jet_lag_direction'] = self.df.apply(lambda x: get_lag_direction(x,'away'), axis=1)


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

        ## keep columns not already in GameLogs
        self.df = self.df[['starttime','usedh','howscored','pitches','oscorer',
                           'temp','winddir','windspeed','fieldcond','precip',
                           'sky','timeofgame','identifier']]

class PITCHfx(Abstract):

    def __init__(self, year):

        ## output from event_parser.csh
        self.year = year

        ## pandas data frame from corresponding year
        self.df = pd.read_csv('data/'+str(self.year)+'/pfx'+str(self.year)+'.csv')

class PlayerMap(Abstract):

    def __init__(self,):

        self.df = pd.read_csv('data/misc/crunchtime/master.csv')
