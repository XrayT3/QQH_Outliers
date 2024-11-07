import numpy as np
import pandas as pd

from xgboost import XGBClassifier

class Model:

    # 0 - H
    # 1 - A

    # 0 - H WIN PCT home winning percentage
    # 1 - A WIN PCT away winning percentage
    # 2 - Amount of Games
    # 3 - Amount of goals
    # 4 - H GS AVG home goals scored average
    # 5 - A GS AVG away goals scored average
    # 6 - H GC AVG home goals conceded average
    # 7 - A GC AVG away goals conceded average
    # 8 - Amount of goals conceded
    # 9 - H GS STD home goals scored standard deviation
    # 10- A GS STD away goals scored standard deviation
    # 11- H GC STD home goals conceded standard deviation
    # 12- A GC STD away goals conceded standard deviation


    def __init__(self):
        self.season = -1
        self.model = None
        self.games = pd.DataFrame()
        self.players = pd.DataFrame()


    def store_games_info(self, games, players):
        if self.games.empty:
            self.games = games
        else:
            self.games = pd.concat([self.games, games])
        # TODO uncomment
        # if self.players.empty:
        #     self.players = players
        # else:
        #     self.players = pd.concat([self.players, players])


    def get_x_train(self):
        seasons_array = self.games['Season'].unique()

        # remove old seasons
        if len(seasons_array) > 3:
            self.games = self.games[self.games['Season'].isin([seasons_array[-3:]])]

        last_season = self.games['Season'].iloc[-1]
        train_size = self.games[self.games['Season'] == last_season].shape[0]
        x_train = np.zeros((train_size, 4))
        for i, idx in enumerate(self.games[self.games['Season'] == last_season].index):
            # TODO what if we have no data
            # TODO 4 features for now, add more
            id_1 = self.games['HID'].iloc[idx]
            id_2 = self.games['AID'].iloc[idx]
            home_match_1 = self.games[(self.games['HID'] == id_1) & (self.games.index < idx)].shape[0]
            away_match_1 = self.games[(self.games['AID'] == id_1) & (self.games.index < idx)].shape[0]
            home_match_2 = self.games[(self.games['HID'] == id_2) & (self.games.index < idx)].shape[0]
            away_match_2 = self.games[(self.games['AID'] == id_2) & (self.games.index < idx)].shape[0]
            home_w_math_1 = self.games[(self.games['HID'] == id_1) & (self.games.index < idx) & (self.games['H'] == 1)].shape[0]
            away_w_math_1 = self.games[(self.games['AID'] == id_1) & (self.games.index < idx) & (self.games['A'] == 1)].shape[0]
            home_w_math_2 = self.games[(self.games['HID'] == id_2) & (self.games.index < idx) & (self.games['H'] == 1)].shape[0]
            away_w_math_2 = self.games[(self.games['AID'] == id_2) & (self.games.index < idx) & (self.games['A'] == 1)].shape[0]
            h_win_pct_1 = 0.5 if home_match_1 == 0 else home_w_math_1 / home_match_1
            a_win_pct_1 = 0.5 if away_match_1 == 0 else away_w_math_1 / away_match_1
            h_win_pct_2 = 0.5 if home_match_2 == 0 else home_w_math_2 / home_match_2
            a_win_pct_2 = 0.5 if away_match_2 == 0 else away_w_math_2 / away_match_2
            x_train[i, 0] = h_win_pct_1
            x_train[i, 1] = a_win_pct_1
            x_train[i, 2] = h_win_pct_2
            x_train[i, 3] = a_win_pct_2

        return x_train

    def get_y_train(self):
        last_season = self.games['Season'].iloc[-1]
        y_train = self.games[self.games['Season'] == last_season]['A']
        return y_train

    def train(self):
        # create feature
        x_train = self.get_x_train()
        # create labels
        y_train = self.get_y_train()
        # TODO setup hyperparameters
        self.model = XGBClassifier(n_estimators=1, max_depth=4, learning_rate=1, objective='binary:logistic')
        # fit model
        self.model.fit(x_train, y_train)

    def is_enough_data(self):
        # two seasons is enough data
        return len(self.games['Season'].unique()) >= 3

    def is_new_season(self, opps):
        curr_season = opps['Season'].iloc[0]
        if self.season == curr_season:
            return False
        else:
            self.season = curr_season
            print("New season: " + str(self.season))
            return True


    def get_features(self, opps):
        features = np.zeros((opps.shape[0], 4))
        for i, idx in enumerate(opps.index):
            id_1 = opps['HID'].iloc[idx]
            id_2 = opps['AID'].iloc[idx]

            home_match_1 = self.games[(self.games['HID'] == id_1)]
            away_match_1 = self.games[(self.games['AID'] == id_1)]
            home_match_2 = self.games[(self.games['HID'] == id_2)]
            away_match_2 = self.games[(self.games['AID'] == id_2)]
            home_w_math_1 = self.games[(self.games['HID'] == id_1) & (self.games['H'] == 1)]
            away_w_math_1 = self.games[(self.games['AID'] == id_1) & (self.games['A'] == 1)]
            home_w_math_2 = self.games[(self.games['HID'] == id_2) & (self.games['H'] == 1)]
            away_w_math_2 = self.games[(self.games['AID'] == id_2) & (self.games['A'] == 1)]
            h_win_pct_1 = home_w_math_1 / home_match_1
            a_win_pct_1 = away_w_math_1 / away_match_1
            h_win_pct_2 = home_w_math_2 / home_match_2
            a_win_pct_2 = away_w_math_2 / away_match_2
            features[i, 0] = h_win_pct_1
            features[i, 1] = a_win_pct_1
            features[i, 2] = h_win_pct_2
            features[i, 3] = a_win_pct_2

        return features


    def place_bets(self, summary: pd.DataFrame, opps: pd.DataFrame, inc: tuple[pd.DataFrame, pd.DataFrame]):
        # store data
        self.store_games_info(inc[0], inc[1])

        if self.is_new_season(opps) and self.is_enough_data():
            self.train()

        min_bet = summary.iloc[0]["Min_bet"]
        N = len(opps)
        bets = np.zeros((N, 2))
        # bets[np.arange(N), np.random.choice([0, 1])] = min_bet

        # predict
        if self.model:
            features = self.get_features(opps)
            predictions = self.model.predict(features)
            print(predictions)

        bets[np.arange(N), np.argmin(opps[['OddsH', 'OddsA']], axis=1)] = min_bet
        bets = pd.DataFrame(data=bets, columns=["BetH", "BetA"], index=opps.index)
        return bets
    