import os
import torch
import trainer
import datasets
import models
import pandas as pd

# Model Configuration
CSV_PATH = "matches.csv"
SAVE_PATH = "./"
NUM_EPOCHS = 100

# Hyperparameters
HISTORY_LEN = 40
H_EMBED_DIM = 12
H_NUM_HEADS = 4
TC_EMBED_DIM = 12
TC_NUM_HEADS = 4
DROPOUT_PROB = 0.7
BATCH_SIZE = 64
LEARNING_RATE = 0.002

def load_match_data(csv_path):
    return pd.read_csv(csv_path)

def train_model():
    # Use CUDA if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == 'cuda':
        print(f'Using cuda device {torch.cuda.get_device_name()}')
    else:
        print(f"Using {device} device")
        
    # Load datasets
    match_data = load_match_data(CSV_PATH)
    match_dataset = datasets.MatchDataset(device, match_data, champ_only=False)
    player_history = datasets.PlayerHistory(match_dataset, history_len=HISTORY_LEN)
    
    # Load model
    model = models.GamePredictionTransformer(
        player_history, 
        match_dataset.num_champs, 
        H_EMBED_DIM, 
        H_NUM_HEADS, 
        TC_EMBED_DIM, 
        TC_NUM_HEADS, 
        DROPOUT_PROB
    ).to(device)
        
    print(model)
    print(f'{sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters.')

    # Load trainer
    train = trainer.Trainer(model, match_dataset, lr=LEARNING_RATE)
    train.train(num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)

    # Save model and plot results
    torch.save(model.state_dict(), os.path.join(SAVE_PATH, 'combined_model_params.txt'))
    train.plot_training_history(SAVE_PATH)

if __name__ == '__main__':
    train_model()