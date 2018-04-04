import pandas as pd
import numpy as np
import urllib2 
from bs4 import BeautifulSoup
import re

from analyzer import GameLogs, EventInfo, Parks

def get_mlb_url(x):
    """ Takes game information and returns the correct PitchFx url

    Args:
         dataframe with proper year, month, day, away team, home team, 
         and game number (for double and triple headers) defined
    Returns:
         string with mlb base url defined
    """

    ## concatenate game info to build url
    base_url = "http://gd2.mlb.com/components/game/mlb"
    base_url += "/year_"+x.year
    base_url += "/month_"+x.month
    base_url += "/day_"+x.day
    base_url += "/gid_"+x.year+"_"+x.month+"_"+x.day+"_"
    base_url += x.away_team.lower()+"mlb_"+x.home_team.lower()+"mlb_"
    base_url += str(max(x.game_number,1))+"/"

    return base_url

## Get start and end years from console
start = int(raw_input("First year to scrape (leave blank to use earliest): ") 
            or "2006")
end = int(raw_input("Last year to scrape (leave blank to use latest): ")
          or "2017")
years = range(start, end+1)

for year in years:
    
    print "Scraping PitchFx for year", year

    ## placeholder for list of dataframes from games
    g_dfs = []

    ## GameLogs instance merged with Parks and EventInfo
    Games = GameLogs(year) + Parks() + EventInfo(year)
    Games.add_time_stamp()

    ## add base url
    Games.df['mlb_url'] = Games.df.apply(lambda x: get_mlb_url(x), axis=1)

    for index, row in Games.df.iterrows():

        ## progress indicator
        if index%100==0 and index>0:
            print "processed",index,"of",Games.df.shape[0],"games in year",year

        ## placeholder for list of dataframes from innings
        i_dfs = []

        ## scraping from inning_all.xml, 
        ## contains all relavent info for entire game
        url = row.mlb_url + "inning/inning_all.xml"

        ## Use BeautifulSoup module with lxml to parse PITCHf/x data
        game_soup = BeautifulSoup(urllib2.urlopen(url),'lxml')

        
        ## pitch columns to add to dataframe
        ## not all games have every field,
        ## in which case, NaN values will be
        ## stored to be handled later
        pitch_cols = ['ax','ay','az','pitch_type','event_num','break_angle',
                      'break_length','break_y','cc','code','des','end_speed',
                      'id','mt','nasty','pfx_x','pfx_z','play_guid','px','pz',
                      'spin_dir','spin_rate','start_speed','sv_id','sz_bot',
                      'sz_top','tfs','tfs_zulu','type','type_confidence','vx0',
                      'vy0','vz0','x','x0','y','y0','z0','zone'
                  ]
        ## pitch columns to add to dataframe
        ## not all games have every field,
        ## in which case, NaN values will be
        ## stored to be handled later
        ab_cols = ['b','b_height','home_team_runs','start_tfs','des',
                   'pitcher','o','end_tfs_zulu','s','num','batter',
                   'stand','away_team_runs','p_throws','event','start_tfs_zulu']

        ## scrape all innings in game
        for inning in game_soup.find_all("inning"):

            ## split by half inning
            half_innings = [inning.top, inning.bottom]
            
            ## placeholder list for half inning dataframes
            hi_dfs = []

            ## exception handling (when no ab in inning (e.g. rain out)

            for half_inning in half_innings:

                ## handle bottom of 9th when home team is winning
                if half_inning is None:
                    break

                ## scrape all at bats (plate appearences) in half inning
                atbats = half_inning.find_all('atbat')
                
                ## handle rain outs after inning begins with no at bats
                if atbats==[]:
                    break

                ## placeholder for list of AB dataframes
                ab_dfs = []

                for atbat in atbats:
                    
                    ## scrame all pitches in AB
                    pitches = atbat.find_all('pitch')
                    
                    ## placeholder dictionary to be converted into dataframe
                    temp_d = {}

                    ## initialize an empty list for each pitch
                    ## column in dictionary
                    for col in pitch_cols:
                        temp_d[col] = []

                    ## initialize an empty list for each AB
                    ## column in dictionary
                    for col in ab_cols:
                        ## handle case when AB column name is the
                        ## same as the pitch column name
                        if col in pitch_cols:
                            temp_d[col+'_atbat'] = []
                        else:
                            temp_d[col] = []

                    ## Fill column values
                    for pitch in pitches:
                        
                        ## handle new intentional walk rule
                        if pitch['des']=="Automatic Ball":
                            continue

                        ## Fill pitch columns
                        for col in pitch_cols:
                            ## if column not in pitch, fill with NaN
                            try:
                                temp_d[col].append(pitch[col]) 
                            except:
                                temp_d[col].append(np.nan)

                        ## Fill AB columns
                        for col in ab_cols:

                            ## handle case when AB column name is the 
                            ## same as the pitch column name
                            if col in pitch_cols:
                                ## if column not in atbat, fill with NaN
                                try:
                                    temp_d[col+'_atbat'].append(atbat[col])
                                except:
                                    temp_d[col+'_atbat'].append(np.nan)

                            ## regular senario, i.e. AB column name
                            ## different than pitch column name
                            else:                            
                                ## if column not in atbat, fill with NaN
                                try:
                                    temp_d[col].append(atbat[col])
                                except:
                                    temp_d[col].append(np.nan)

                    ## Add inning identifier column
                    temp_d['inning'] = [inning['num']]*len(temp_d['ax'])

                    ## Add game identifier column to allow merging with 
                    ## other dataframes
                    temp_d['identifier'] = [row.identifier]*len(temp_d['ax'])

                    ## cascade of dataframe aggregating and concating 
                    ab_dfs.append(pd.DataFrame(temp_d))
                hi_dfs.append(pd.concat(ab_dfs))
            ## handle rain outs
            if hi_dfs == []:
                break
            i_dfs.append(pd.concat(hi_dfs))
        g_dfs.append(pd.concat(i_dfs))

    ## main PITCHf/x Dataframe with all games
    pfx_df = pd.concat(g_dfs)

    ## save to csv in year directory structure
    pfx_df.to_csv('data/'+year+'/pfx'+year+'.csv')

    ## clear year from memory 
    del Games
    del pfx_df
