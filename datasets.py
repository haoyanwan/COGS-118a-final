import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from progress.bar import Bar

class MatchDataset(Dataset):
    def __init__(self, device, df: pd.DataFrame, seed=42, test_train_ratio=0.90, champ_only=True):
        self.device = device
        self.champ_only = champ_only

        df = self._clean_data(df)

        print('Building dicts...')
        self._init_dicts(df)
        processed_data = self._preprocess(df)

        torch.manual_seed(seed)
        shuffle_ind = torch.randperm(processed_data.size()[0])
        processed_data = processed_data[shuffle_ind]
        
        test_train_idx = round(test_train_ratio*processed_data.shape[0])
        self.train, self.dev = processed_data[:test_train_idx], processed_data[test_train_idx:]

    def _clean_data(self, df: pd.DataFrame):
        for i in range(1, 11):
            df[f'p{i}_champId'] = df[f'p{i}_champId'].fillna(0).astype(float).astype(int)
            df[f'p{i}_key'] = df[f'p{i}_key'].fillna(0).astype(float).astype(int)
        return df

    def __len__(self):
        return self.train.shape[0]

    def __getitem__(self, idx):
        match = self.train[idx, 2:]
        outcome = self.train[idx, 0].to(torch.float)
        time = self.train[idx, 1]
        return match, outcome, time
        
    def get_dev(self): 
        matches = self.dev[:, 2:]
        outcomes = self.dev[:, 0].to(torch.float)
        times = self.dev[:, 1]
        return matches, outcomes, times

    def get_train(self):
        matches = self.train[:, 2:]
        outcomes = self.train[:, 0].to(torch.float)
        times = self.train[:, 1]
        return matches, outcomes, times

    def _init_dicts(self, df: pd.DataFrame):
        champs_present = []
        for i in range(1, 11):
            champs = df[f'p{i}_champId'].unique()
            champs = champs[~np.isnan(champs)]
            champs_present += list(champs)
        champs_present = list(set(champs_present))
        
        if 0 in champs_present:
            champs_present.remove(0)

        self.key_to_champid = {k+1:v for k, v in enumerate(champs_present)}
        self.champid_to_key = {v:k+1 for k, v in enumerate(champs_present)}
        self.num_champs = len(champs_present)+1

        if not self.champ_only:
            players_present = []
            for i in range(1, 11):
                players = df[f'p{i}_key'].unique()
                players = players[~np.isnan(players)]
                players_present += list(players)
            players_present = list(set(players_present))
            
            if 0 in players_present:
                players_present.remove(0)
            
            self.num_players = len(players_present)
            
            self.key_to_player = {key: player for key, player in enumerate(players_present)}
            self.player_to_key = {player: key for key, player in enumerate(players_present)}
    
    def _preprocess(self, df: pd.DataFrame):
        matches = []
        with Bar('Preprocessing matches...', max=df.shape[0]) as bar:
            for _, match_row in df.iterrows():
                match = {}
                match['blue_win'] = 1 if match_row['winning_team'] == 1 else 0
                match['created_at'] = 0
                
                for i in range(1, 11):
                    champ_id = match_row[f'p{i}_champId']
                    match[f'p{i}_champ'] = self.champid_to_key.get(champ_id, 0)
                    
                    if not self.champ_only:
                        player_key = match_row[f'p{i}_key']
                        match[f'p{i}_id'] = self.player_to_key.get(player_key, 0)
                
                matches.append(match)
                bar.next()
        
        ordered_cols = ['blue_win', 'created_at']
        for i in range(1, 11):
            if not self.champ_only:
                ordered_cols.append(f'p{i}_id')
            ordered_cols.append(f'p{i}_champ')

        processed_data = pd.DataFrame.from_records(matches).astype(int)[ordered_cols]
        processed_data = torch.tensor(processed_data.values, dtype=torch.long).to(self.device)
        
        return processed_data


class PlayerHistory(Dataset):
    def __init__(self, match_dataset: MatchDataset, history_len: int): 
        self.history_len = history_len
        self.match_history = self._preprocess(match_dataset)
        
    def __len__(self):
        return self.match_history.shape[0]

    def __getitem__(self, idx):
        return self.match_history[idx]
        
    def _preprocess(self, match_dataset): 
        match_history_dict = {}
        for i in range(match_dataset.num_players):
            match_history_dict[i] = []
        with Bar('Building match history...', max=len(match_dataset)) as bar:
            for idx in range(len(match_dataset)):
                match, outcome, time = match_dataset[idx]
                for i in range(0, 10, 2):
                    match_history_dict[match[i].item()].append([match[i+1].item(), outcome.item(), time.item()])
                for i in range(10, 20, 2):
                    match_history_dict[match[i].item()].append([match[i+1].item(), 1-outcome.item(), time.item()])
                bar.next()

        for k, match_history in match_history_dict.items():
            match_history_len = len(match_history)
            if match_history_len < self.history_len:
                match_history += [[0, -1, 0]]*(self.history_len - match_history_len)
            match_history.sort(key=lambda x: x[2], reverse=True)
            match_history_dict[k] = match_history[:self.history_len]

        return torch.stack([torch.tensor(match_history_dict[i], dtype=torch.long) for i in range(len(match_history_dict))], dim=0).to(match_dataset.device)