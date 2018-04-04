#/bin/tcsh

setenv YEAR $1

dos2unix data/$YEAR/$YEAR*.EV*

#cat data/$YEAR/$YEAR*.EV* | grep '^info,' | head -26 | sed 's/info,//' | sed 's/,.*//' | tr '\n' ',' | sed 's/,$/\n/' > ! data/$YEAR/event_info_$YEAR.csv

echo "visteam,hometeam,site,date,number,starttime,daynight,usedh,umphome,ump1b,ump2b,ump3b,howscored,pitches,oscorer,temp,winddir,windspeed,fieldcond,precip,sky,timeofgame,attendance,wp,lp,save" >! data/$YEAR/event_info_$YEAR.csv
cat data/$YEAR/$YEAR*.EV* | egrep '(^version,|^info,visteam,|^info,hometeam,|^info,site,|^info,date,|^info,number,|^info,starttime,|^info,daynight,|^info,usedh,|^info,umphome,|^info,ump1b,|^info,ump2b,|^info,ump3b,|^info,howscored,|^info,pitches,|^info,oscorer,|^info,temp,|^info,winddir,|^info,windspeed,|^info,fieldcond,|^info,precip,|^info,sky,|^info,timeofgame,|^info,attendance,|^info,wp,|^info,lp,|^info,save,)' | sed 's/info,//' | sed 's/^version.*/@@@/' | sed 's/.*,//' | tr '\n' ',' | sed 's/,@@@,/\n/g' | sed 's/^@@@,//' | sed 's/,$/\n/' >> data/$YEAR/event_info_$YEAR.csv
