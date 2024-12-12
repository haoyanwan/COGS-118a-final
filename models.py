import torch
from torch import nn as nn
from datasets import PlayerHistory
import math


class TemporalPositionEncoder(nn.Module):
    def __init__(self, embedding_dim: int, dropout_rate: float = 0.1, sequence_length: int = 10):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_rate)

        positions = torch.arange(sequence_length).unsqueeze(1)
        scaling_factor = torch.exp(torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim))
        positional_encoding = torch.zeros(1, sequence_length, embedding_dim)
        positional_encoding[0, :, 0::2] = torch.sin(positions * scaling_factor)
        positional_encoding[0, :, 1::2] = torch.cos(positions * scaling_factor)
        self.register_buffer('positional_encoding', positional_encoding)

    def forward(self, input_tensor):
        encoded = input_tensor + self.positional_encoding[:input_tensor.size(0)]
        return self.dropout(encoded)


class MultiHeadAttentionModule(nn.Module):
    def __init__(self, embedding_dim: int, num_attention_heads: int):
        super(MultiHeadAttentionModule, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        
        self.query_transform = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.key_transform = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.value_transform = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.multihead_attention = nn.MultiheadAttention(
            embedding_dim, 
            num_attention_heads, 
            batch_first=True, 
            add_zero_attn=True
        )

    def forward(self, history_embeddings, current_embedding, attention_mask):
        query = self.query_transform(current_embedding).view(-1, 1, self.embedding_dim)
        keys = self.key_transform(history_embeddings)
        values = self.value_transform(history_embeddings)

        attention_output = self.multihead_attention(
            query, 
            keys, 
            values, 
            key_padding_mask=attention_mask
        )[0]
        return attention_output.view(-1, self.embedding_dim)


class PlayerMatchHistoryEncoder(nn.Module):
    def __init__(self, num_champions: int, embedding_dim: int, num_attention_heads: int, dropout_rate=0.1):
        super(PlayerMatchHistoryEncoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.champion_embedding = nn.Embedding(num_champions, embedding_dim, padding_idx=0)
        self.win_projection = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.loss_projection = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.attention = MultiHeadAttentionModule(embedding_dim, num_attention_heads)
        self.attention_dropout = nn.Dropout(dropout_rate)
        self.attention_norm = nn.LayerNorm(embedding_dim)

        self.feedforward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.Dropout(dropout_rate)
        )
        self.feedforward_norm = nn.LayerNorm(embedding_dim)
    
    def forward(self, match_history, current_match):
        history_embeddings, current_embeddings = self.encode_match_data(match_history, current_match)
        attention_mask = self.create_temporal_mask(match_history, current_match)
        attention_output = self.process_attention(history_embeddings, current_embeddings, attention_mask)
        normalized_attention = self.attention_norm(attention_output + current_embeddings)

        feedforward_output = self.feedforward(normalized_attention)
        final_output = self.feedforward_norm(feedforward_output + normalized_attention)

        return final_output
    
    def encode_match_data(self, match_history, current_match):
        current_champion_embedding = self.champion_embedding(current_match[:, 0])
        
        history_champions = match_history[:,:,0]
        match_outcomes = match_history[:,:,1]
        outcome_mask = torch.unsqueeze(match_outcomes, -1).expand(-1, -1, self.embedding_dim)
        
        champion_embeddings = self.champion_embedding(history_champions)
        transformed_embeddings = (
            self.win_projection(torch.where(outcome_mask == 1, champion_embeddings, 0)) + 
            self.loss_projection(torch.where(outcome_mask == 0, champion_embeddings, 0))
        )
        
        return transformed_embeddings, current_champion_embedding

    def process_attention(self, history_embeddings, current_embeddings, attention_mask):
        attention_output = self.attention(history_embeddings, current_embeddings, attention_mask)
        return self.attention_dropout(attention_output)

    def create_temporal_mask(self, match_history, current_match):
        history_length = match_history.shape[1]
        current_time = torch.unsqueeze(current_match[:, 1], -1).expand(-1, history_length)
        mask = (current_time == match_history[:, :, 2]) + (match_history[:, :, 2] == 0)
        return mask


class TeamCompositionTransformer(nn.Module):
    def __init__(self, num_champions: int, embedding_dim: int, num_attention_heads: int, dropout_rate=0.1):
        super(TeamCompositionTransformer, self).__init__()
        
        self.champion_embedding = nn.Embedding(num_champions, embedding_dim)
        self.positional_encoder = TemporalPositionEncoder(embedding_dim)
        self.transformer_encoder = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_attention_heads,
            dropout=dropout_rate,
            dim_feedforward=embedding_dim,
            batch_first=True
        )
        
        self.feedforward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.Dropout(dropout_rate)
        )
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
    def forward(self, team_composition):
        embedded_champions = self.champion_embedding(team_composition)
        positional_encoded = self.positional_encoder(embedded_champions)
        transformer_output = self.transformer_encoder(positional_encoded)
        
        feedforward_output = self.feedforward(transformer_output)
        normalized_output = self.layer_norm(feedforward_output + transformer_output)
        
        return normalized_output


class GamePredictionTransformer(nn.Module):
    def __init__(self, player_history: PlayerHistory, num_champions: int,
                 history_embedding_dim: int, history_attention_heads: int, 
                 team_embedding_dim: int, team_attention_heads: int, 
                 dropout_rate=0.1):

        super(GamePredictionTransformer, self).__init__()
        self.player_history = player_history
        self.history_embedding_dim = history_embedding_dim
        self.team_embedding_dim = team_embedding_dim
        self.history_attention_heads = history_attention_heads
        self.team_attention_heads = team_attention_heads
        self.name = 'combined'

        self.player_history_encoder = PlayerMatchHistoryEncoder(
            num_champions, 
            history_embedding_dim, 
            history_attention_heads, 
            dropout_rate
        )
        self.history_projection = nn.Linear(history_embedding_dim, history_embedding_dim)
        self.history_activation = nn.ReLU()
        self.history_dropout = nn.Dropout(dropout_rate)
        self.history_norm = nn.LayerNorm(history_embedding_dim, eps=1e-5)
    
        self.team_composition_encoder = TeamCompositionTransformer(
            num_champions, 
            team_embedding_dim, 
            team_attention_heads, 
            dropout_rate
        )

        combined_dim = 10 * history_embedding_dim + 10 * team_embedding_dim
        self.flatten = nn.Flatten()
        self.final_projection = nn.Linear(combined_dim, combined_dim)
        self.final_activation = nn.ReLU()
        self.final_dropout = nn.Dropout(dropout_rate)

        self.output_layer = nn.Linear(combined_dim, 1)
        self.output_activation = nn.Sigmoid()

    def forward(self, match_data, timestamp):
        history_encoded = self.encode_player_histories(*self.prepare_history_data(match_data, timestamp))
        team_composition_encoded = self.encode_team_composition(match_data)
        
        combined_features = torch.cat((
            self.flatten(history_encoded), 
            self.flatten(team_composition_encoded)
        ), dim=1)
        
        projected_features = self.final_dropout(
            self.final_activation(self.final_projection(combined_features))
        )
        
        prediction = self.output_activation(self.output_layer(projected_features))
        return prediction
    
    def prepare_history_data(self, match_data, timestamp):
        player_ids = match_data[:, 0::2]
        match_history = self.player_history[player_ids]
        
        current_champions = match_data[:, 1::2].view(-1, 10, 1)
        current_timestamps = timestamp.view(-1, 1, 1).expand(-1, 10, 1).clone()
        current_match_data = torch.cat((current_champions, current_timestamps), dim=2)
        
        return match_history.flatten(0, 1), current_match_data.flatten(0, 1)
        
    def encode_player_histories(self, match_history, current_match):
        attention_output = self.player_history_encoder(match_history, current_match).reshape(-1, 10, self.history_embedding_dim)
        projected_output = self.history_dropout(
            self.history_activation(self.history_projection(attention_output))
        )
        normalized_output = self.history_norm(attention_output + projected_output)
        return normalized_output

    def encode_team_composition(self, match_data):
        champion_ids = match_data[:, 1::2]
        return self.team_composition_encoder(champion_ids)

