## Dataset schema

### Team attributes
`id`, `conference`, `division`, `city`, `name`, `full_name`, `abbreviation`

Example:
```
id=10 conference='West' division='Pacific' city='Golden State' name='Warriors' full_name='Golden State Warriors' abbreviation='GSW'
```

### Player attributes
`id`, `first_name`, `last_name`, `position`, `height`, `weight`, `jersey_number`, `college`, `country`, `draft_year`, `draft_round`, `draft_number`, `team=NBATeam(...)`

Example:
```
id=115 first_name='Stephen' last_name='Curry' position='G' height='6-2' weight='185' jersey_number='30' college='Davidson' country='USA' draft_year=2009.0 draft_round=1.0 draft_number=7.0 team=NBATeam(id=10, conference='West', division='Pacific', city='Golden State', name='Warriors', full_name='Golden State Warriors', abbreviation='GSW') team_id=None
```

### Game attributes
`id`, `date`, `season=2023`, `status`, `period`, `time`, `postseason`, `home_team_score`, `visitor_team_score`, `home_team=NBATeam(...)`, `visitor_team=NBATeam(...)`

Example:
```
id=3277498 date='2023-11-24' season=2023 status='Final' period=4.0 time='Final' postseason=False home_team_score=113.0 visitor_team_score=96.0 home_team=NBATeam(id=22, conference='East', division='Southeast', city='Orlando', name='Magic', full_name='Orlando Magic', abbreviation='ORL') home_team_id=None visitor_team=NBATeam(id=2, conference='East', division='Atlantic', city='Boston', name='Celtics', full_name='Boston Celtics', abbreviation='BOS') visitor_team_id=None
```