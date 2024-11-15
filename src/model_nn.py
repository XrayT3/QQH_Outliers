import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def get_optimal_fractions(probabilities: np.ndarray, odds: pd.Series) -> np.ndarray:
    # Kelly criterion
    b = odds.to_numpy()

    kelly_home = (1 - probabilities) - (probabilities / b[:, 0])
    kelly_away = probabilities - ((1 - probabilities) / b[:, 1])

    fractions = np.array([kelly_home, kelly_away]).T

    # q = 1 - probabilities
    # fractions = probabilities - (q / b)
    # var = 0.289
    # coefficient = np.square((b + 1) * probabilities - 1) / (
    #             np.square((b + 1) * probabilities - 1) + np.square((b + 1) * var))
    # fractions *= coefficient

    # remove negative values
    fractions[fractions < 0] = 0
    return fractions

def uniform_betting_strategy():
    # TODO
    pass

def sharp_betting_strategy():
    # TODO
    pass


class TeamLevelNN(nn.Module):
    def __init__(self):
        super(TeamLevelNN, self).__init__()
        # Define layers
        self.model = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Using sigmoid for binary classification
        )

    def forward(self, x):
        return self.model(x)

def classify(model, x_test):
    # Convert test data to PyTorch tensor
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)

    # Set the model to evaluation mode
    model.eval()

    # Perform forward pass to get predictions
    with torch.no_grad():  # Disable gradient calculation for inference
        probs = model(x_test_tensor).squeeze()

    # Convert probabilities to binary class predictions
    predictions = (probs >= 0.5).int()  # Threshold of 0.5 for binary classification

    # return predictions.numpy()
    return probs.numpy()


def train_model(x_train, y_train, epochs=50, batch_size=32, lr=0.001):
    # Convert numpy arrays to PyTorch tensors
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    # Create DataLoader
    dataset = TensorDataset(x_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss function, and optimizer
    model = TeamLevelNN()
    # TODO: change loss function
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)  # L2 regularization

    # Training loop
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in dataloader:
            # Forward pass
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    return model

class Model:

    def __init_team(self, team_id):
        self.team_stats[team_id] = {
            "GM CNT": 0,
            "AST SUM": 0
        }


    def __init__(self):
        self.teams_n = 2
        self.year = 1900
        self.team_stats = {}
        self.train_data = [[], []]  # [[X_1, X_2, ...], [y_1, y_2, ...]]
        self.first_try = True
        self.has_model = False
        self.minimal_games = 10
        self.cnf_threshold = 0.2
        self.trained_seasons = []
        self.games = pd.DataFrame()
        self.model = TeamLevelNN()
        self.features_n = 1 * self.teams_n

    def __store_inc(self, games: pd.DataFrame):
        if games.empty: return

        if self.games.empty: self.games = games
        else: self.games = pd.concat([self.games, games])

    def __is_enough_data(self):
        # 5 seasons is enough data
        return len(self.train_data[0]) >= 5

    def __is_new_season(self, summary: pd.DataFrame):
        date = summary.iloc[0]['Date']
        curr_month, curr_year = date.month, date.year
        if curr_month == 11 and curr_year > self.year:  # if it's November and new year => it's a new season
            self.year = curr_year
            return True
        else:
            return False

    def __update_stats(self, match: pd.Series, team_h: int, team_a: int):
        if team_h not in self.team_stats: self.__init_team(team_h)
        if team_a not in self.team_stats: self.__init_team(team_a)

        self.team_stats[team_h]['GM CNT'] += 1
        self.team_stats[team_a]['GM CNT'] += 1

        self.team_stats[team_h]['AST SUM'] += match['HAST']
        self.team_stats[team_a]['AST SUM'] += match['AAST']
        # TODO: add other statistics

    def __process_one_season(self, season_df: pd.DataFrame):
        x_data = np.zeros((season_df.shape[0], self.features_n))
        y_data = np.zeros(season_df.shape[0])
        skipped_rows = []

        # remove rows with incomplete information
        season_df = season_df.dropna()

        for i, row in enumerate(season_df.iterrows()):
            match = row[1]
            team_h, team_a = match['HID'], match['AID']
            if team_h not in self.team_stats: self.__init_team(team_h)
            if team_a not in self.team_stats: self.__init_team(team_a)

            if self.team_stats[team_a]['GM CNT'] > self.minimal_games and self.team_stats[team_h]['GM CNT'] > self.minimal_games:
                x_data[i, :] = self.get_features(team_h, team_a)
                y_data[i] = match['A']
            else:
                skipped_rows.append(i)
            self.__update_stats(match, team_h, team_a)

        # save train data
        mask = np.ones(x_data.shape[0], dtype='bool')
        mask[skipped_rows] = False
        self.train_data[0].append(x_data[mask])
        self.train_data[1].append(y_data[mask])

        # remove this season from self.games
        season_num = season_df['Season'].iloc[0]
        self.games.drop(self.games[self.games['Season'] == season_num].index, inplace=True)

    def __prepare_train_data(self):
        if self.games.empty: return

        # delete prev stats
        for team_id in self.team_stats:
            self.__init_team(team_id)

        seasons_list = self.games['Season'].unique().tolist()
        for season in seasons_list:
            season_df = self.games[self.games['Season'] == season]
            self.__process_one_season(season_df)

    def get_features(self, team_h: int, team_a: int) -> np.ndarray:
        x_features = np.zeros(self.features_n)

        x_features[0] = self.team_stats[team_h]['AST SUM'] / self.team_stats[team_h]['GM CNT']
        x_features[1] = self.team_stats[team_a]['AST SUM'] / self.team_stats[team_a]['GM CNT']

        return x_features

    def get_data(self, opps: pd.DataFrame) -> (np.array, list):
        x_data = np.zeros((opps.shape[0], self.features_n))
        skipped = []

        for i, row in enumerate(opps.iterrows()):
            match = row[1]
            team_h, team_a = match['HID'], match['AID']

            # if there is no info about team, skip
            if team_a not in self.team_stats or team_h not in self.team_stats:
                skipped.append(i)
                continue
            if self.team_stats[team_a]['GM CNT'] < self.minimal_games or self.team_stats[team_h]['GM CNT'] < self.minimal_games:
                skipped.append(i)
                continue

            x_data[i, :] = self.get_features(team_h, team_a)

        return x_data, skipped

    def train_model(self):
        x_train = np.concatenate(self.train_data[0], axis=0)
        y_train = np.concatenate(self.train_data[1], axis=0)
        self.model = train_model(x_train, y_train)
        self.has_model = True

    def place_bets(self, summary: pd.DataFrame, opps: pd.DataFrame, inc: tuple[pd.DataFrame, pd.DataFrame]):
        bankroll = summary.iloc[0]['Bankroll']
        min_bet = summary.iloc[0]['Min_bet']
        max_bet = summary.iloc[0]['Max_bet']
        n = len(opps)
        bets = np.zeros((n, 2))

        # store data
        self.__store_inc(inc[0])

        # update statistics in the current season for teams
        if not self.first_try:  # for qualification upload
            for _, match in inc[0].iterrows():
                self.__update_stats(match, match['HID'], match['AID'])
        self.first_try = False

        # retrain model every season
        if self.__is_new_season(summary):
            self.__prepare_train_data()
            if self.__is_enough_data():
                self.train_model()

        # skip betting on training seasons
        if not self.has_model or opps.empty:
            bets = pd.DataFrame(data=bets, columns=["BetH", "BetA"], index=opps.index)
            return bets

        # make prediction
        x, no_info = self.get_data(opps)
        probs = classify(self.model, x)
        # confidence threshold
        confidence_threshold = np.where(probs < 0.5 + self.cnf_threshold)[0]
        no_bet = np.union1d(no_info, confidence_threshold).astype(int)

        # chose bets
        prev_bets = opps[['BetH', 'BetA']].to_numpy()
        fractions = get_optimal_fractions(probs, opps[['OddsH', 'OddsA']])
        budget = bankroll * 0.1
        budget_per_match = budget / n
        my_bets = fractions * budget_per_match
        my_bets -= prev_bets
        my_bets[my_bets < min_bet] = 0
        my_bets[my_bets > max_bet] = max_bet

        # place bets
        bets = my_bets
        bets[no_bet, :] = 0
        bets = pd.DataFrame(data=bets, columns=["BetH", "BetA"], index=opps.index)
        return bets
