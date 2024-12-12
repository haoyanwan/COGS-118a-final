import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import os
from ratelimit import limits, sleep_and_retry
import random

# Constants
RIOT_API_KEY = os.getenv('RIOT_API_KEY')
BASE_HEADERS = {
    "X-Riot-Token": RIOT_API_KEY
}

# Rate limits
CALLS_PER_SECOND = 20
PERIOD = 1

@sleep_and_retry
@limits(calls=CALLS_PER_SECOND, period=PERIOD)
def call_riot_api(url):
    """Make a rate-limited call to Riot API"""
    response = requests.get(url, headers=BASE_HEADERS)
    if response.status_code == 429:
        retry_after = int(response.headers.get('Retry-After', 5))
        time.sleep(retry_after)
        return call_riot_api(url)
    response.raise_for_status()
    return response.json()

def get_league_players(tier, queue="RANKED_SOLO_5x5", region="euw1"):
    """Get players from specified league tier"""
    if tier.lower() in ["challenger", "grandmaster", "master"]:
        url = f"https://{region}.api.riotgames.com/lol/league/v4/{tier.lower()}leagues/by-queue/{queue}"
        data = call_riot_api(url)
        return data['entries']
    else:
        # For other tiers (DIAMOND, PLATINUM, etc), need to get by division
        players = []
        divisions = ["I", "II", "III", "IV"]
        for division in divisions:
            url = f"https://{region}.api.riotgames.com/lol/league/v4/entries/{queue}/{tier}/{division}"
            players.extend(call_riot_api(url))
        return players

def get_summoner_by_id(summoner_id, region="euw1"):
    """Get summoner data by ID"""
    url = f"https://{region}.api.riotgames.com/lol/summoner/v4/summoners/{summoner_id}"
    return call_riot_api(url)

def get_matches_by_puuid(puuid, region="europe", count=100, queue=420):
    """Get match IDs for a player"""
    url = f"https://{region}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?queue={queue}&count={count}"
    return call_riot_api(url)

def get_match_details(match_id, region="europe"):
    """Get detailed match data"""
    url = f"https://{region}.api.riotgames.com/lol/match/v5/matches/{match_id}"
    return call_riot_api(url)

def process_match_data(match_data):
    """Process match data into the required format"""
    try:
        match_id = match_data['metadata']['matchId']
        game_info = match_data['info']
        
        # Skip if not ranked solo/duo
        if game_info['queueId'] != 420:
            return None
            
        winning_team = 1 if game_info['teams'][0]['win'] else 2
        
        players_data = {}
        for i, participant in enumerate(game_info['participants'], 1):
            players_data[f'p{i}_name'] = participant['summonerName']
            players_data[f'p{i}_key'] = participant['summonerId']
            players_data[f'p{i}_champId'] = participant['championId']
            players_data[f'p{i}_champName'] = participant['championName']
            
        match_row = {
            'match_id': match_id,
            'winning_team': winning_team,
            **players_data
        }
        
        return match_row
    except KeyError as e:
        print(f"Error processing match {match_data['metadata']['matchId']}: {e}")
        return None

def continuously_collect_matches(region="euw1", save_interval=100):
    """Continuously collect matches from all ranks"""
    all_matches = []
    unique_match_ids = set()
    matches_since_save = 0
    
    # Read existing match IDs if file exists
    output_path = "matches.csv"
    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path)
        unique_match_ids.update(existing_df['match_id'].values)
        print(f"Loaded {len(unique_match_ids)} existing match IDs")

    tiers = ["CHALLENGER", "GRANDMASTER", "MASTER", "DIAMOND"]
    
    while True:
        try:
            # Cycle through tiers
            for tier in tiers:
                print(f"\nFetching {tier} players...")
                players = get_league_players(tier, region=region)
                
                # Shuffle players to get different matches each time
                random.shuffle(players)
                
                for player in players:
                    try:
                        # Get summoner details
                        summoner = get_summoner_by_id(player['summonerId'], region)
                        print(f"Processing {tier} player: {summoner['name']} (LP: {player['leaguePoints']})")
                        
                        # Get match IDs
                        matches = get_matches_by_puuid(summoner['puuid'])
                        new_matches = 0
                        
                        # Get match details
                        for match_id in matches:
                            if match_id not in unique_match_ids:
                                unique_match_ids.add(match_id)
                                match_data = get_match_details(match_id)
                                processed_match = process_match_data(match_data)
                                if processed_match:
                                    all_matches.append(processed_match)
                                    matches_since_save += 1
                                    new_matches += 1
                                    
                                    # Save periodically
                                    if matches_since_save >= save_interval:
                                        df = pd.DataFrame(all_matches)
                                        save_matches(df)
                                        all_matches = []
                                        matches_since_save = 0
                        
                        print(f"Added {new_matches} new matches from {summoner['name']}")
                        
                    except Exception as e:
                        print(f"Error processing player {player.get('summonerName', 'Unknown')}: {e}")
                        continue
                        
        except Exception as e:
            print(f"Error in main loop: {e}")
            time.sleep(60)  # Wait a minute before retrying
            continue

def save_matches(df, output_path="matches.csv"):
    """Save matches to CSV file"""
    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path)
        df = pd.concat([existing_df, df]).drop_duplicates(subset=['match_id'])
    
    df.to_csv(output_path, index=False)
    print(f"Saved total of {len(df)} matches to {output_path}")

def main():
    continuously_collect_matches()

if __name__ == "__main__":
    if not RIOT_API_KEY:
        print("Please set RIOT_API_KEY environment variable")
    else:
        main()