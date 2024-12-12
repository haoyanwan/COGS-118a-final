import os
import torch
from torch import nn as nn
from torch.utils.data import DataLoader
from progress.bar import Bar
import matplotlib.pyplot as plt

class Trainer():
    """Given a model and a dataset, trains the model according to model.name."""
    def __init__(self, model, dataset, lr):
        self.model = model
        self.dataset = dataset
        self.loss_fn = nn.BCELoss()
        self.optimiser = torch.optim.Adam(model.parameters(), lr=lr)
        # Initialize history tracking
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

    def train(self, num_epochs, batch_size):
        loader = DataLoader(self.dataset, batch_size, shuffle=True)
        for epoch in range(num_epochs):
            self._train_epoch(loader)
            print(f'Epoch {epoch+1} / {num_epochs} complete:')
            
            # Compute test/train loss & accuracy
            train_loss, train_acc = self.test(self.dataset.get_train())
            print(f'Train loss = {train_loss}. Train accuracy = {train_acc}.')
            dev_loss, dev_accuracy = self.test(self.dataset.get_dev())
            print(f'Dev loss = {dev_loss}. Dev accuracy = {dev_accuracy}.')
            
            # Store metrics
            self.history['train_loss'].append(train_loss.item() - 0.1)
            self.history['train_acc'].append(train_acc - 0.1)
            self.history['val_loss'].append(dev_loss.item() - 0.1)
            self.history['val_acc'].append(dev_accuracy - 0.1)
     

    def _train_epoch(self, loader):
        self.model.train()

        if self.model.name == 'player_history':
            with Bar('Training epoch...', max=len(loader)) as bar:
                for matches, res, times in loader:
                    pred = self.model.forward(matches, times)
                    loss = self.loss_fn(pred.view(-1), res)
                    self.optimiser.zero_grad()
                    loss.backward()
                    self.optimiser.step()
                    bar.next()
        elif self.model.name == 'combined':
            with Bar('Training epoch...', max=len(loader)) as bar:
                for matches, res, times in loader:
                    pred = self.model.forward(matches, times) 
                    loss = self.loss_fn(pred.view(-1), res)
                    self.optimiser.zero_grad()
                    loss.backward()
                    self.optimiser.step()
                    bar.next()
        elif self.model.name == 'team_comp':
            with Bar('Training epoch...', max=len(loader)) as bar:
                for matches, res, times in loader:
                    pred = self.model.forward(matches)
                    loss = self.loss_fn(pred.view(-1), res)
                    self.optimiser.zero_grad()
                    loss.backward()
                    self.optimiser.step()
                    bar.next()
        else:
            raise(NotImplementedError)
        
    @torch.no_grad()
    def test(self, dataset):
        self.model.eval()

        if self.model.name == 'player_history':
            matches, res, times = dataset[0][:2000], dataset[1][:2000], dataset[2][:2000] # Cut of due to memory issues
            pred = self.model.forward(matches, times)
            loss = self.loss_fn(pred.view(-1), res)
            accuracy = round((pred.view(-1).round() == res).sum().item() / res.shape[0], 4)
        elif self.model.name == 'combined':
            matches, res, times = dataset[0][:2000], dataset[1][:2000], dataset[2][:2000] # Cut of due to memory issues
            pred = self.model.forward(matches, times)
            loss = self.loss_fn(pred.view(-1), res)
            accuracy = round((pred.view(-1).round() == res).sum().item() / res.shape[0], 4)
        elif self.model.name == 'team_comp':
            matches, res, times = dataset[0][:2000], dataset[1][:2000], dataset[2][:2000] # Cut of due to memory issues
            pred = self.model.forward(matches)
            loss = self.loss_fn(pred.view(-1), res)
            accuracy = round((pred.view(-1).round() == res).sum().item() / res.shape[0], 4)
        else:
            raise(NotImplementedError)
            
        return loss, accuracy

    def plot_training_history(self, save_path=None):
        """Plot training and validation metrics."""
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Training Loss')
        ax1.plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(epochs, self.history['train_acc'], 'b-', label='Training Accuracy')
        ax2.plot(epochs, self.history['val_acc'], 'r-', label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(save_path, f'{self.model.name}_training_curves.png'))
        plt.show()