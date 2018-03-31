#/bin/tcsh

setenv YEAR $1

cat data/$YEAR/$YEAR*.EV* | grep '^info,' | head -26 | sed 's/info,//' | sed 's/,.*//' | tr '\n' ',' | sed 's/,$/\n/' > ! data/$YEAR/event_info_$YEAR.csv

cat data/$YEAR/$YEAR*.EV* | egrep '(^info,|^version,)' | sed 's/info,//' | sed 's/^version.*/@@@/' | sed 's/.*,//' | tr '\n' ',' | sed 's/,@@@,/\n/g' | sed 's/^@@@,//' | sed 's/,$/\n/' >> data/$YEAR/event_info_$YEAR.csv
