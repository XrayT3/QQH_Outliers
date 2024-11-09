import numpy as np
import pandas as pd
from math import log10

from xgboost import XGBClassifier


def get_optimal_fractions(probabilities: np.ndarray, odds: pd.Series) -> np.ndarray:
    odds = odds.to_numpy()
    q = 1 - probabilities
    return probabilities - (q / odds)


class Model:

    def __init_team(self, team_id):
        self.team_stats[team_id] = {
                    "H GM CNT": 0,
                    "A GM CNT": 0,
                    "H WIN CNT": 0,
                    "A WIN CNT": 0,
                    "H GS": 0,
                    "A GS": 0,
                    "H GC": 0,
                    "A GC": 0,
                    "H GS HISTORY": [],
                    "A GS HISTORY": [],
                    "H GC HISTORY": [],
                    "A GC HISTORY": [],
        }
        self.team_historical_strength[team_id] = {
                    "H WIN PCT": 0,
                    "A WIN PCT": 0,
                    "H GS AVG": 0,
                    "A GS AVG": 0,
                    "H GC AVG": 0,
                    "A GC AVG": 0,
                    "H GS STD": 0,
                    "A GS STD": 0,
                    "H GC STD": 0,
                    "A GC STD": 0
        }
        self.team_pi_rating[team_id] = {
                    "H RTG": 0.0,
                    "A RTG": 0.0
        }

    def __init__(self):
        self.lr = 0.26
        self.gamma = 0.5 # or 0.3
        self.teams_n = 2
        self.cnf_threshold = 0.2
        self.minimal_games = 39 * 4
        self.features_n = 2 * self.teams_n
        self.has_model = False
        self.team_stats = {}
        self.team_pi_rating = {}
        self.team_historical_strength = {}
        self.games = pd.DataFrame()
        self.players = pd.DataFrame()
        self.season = -1
        # TODO: set up hyperparameters
        self.model = XGBClassifier(max_depth=4, subsample=0.8, min_child_weight=5, colsample_bytree=0.25, seed=24)

    def __store_inc(self, games: pd.DataFrame, players: pd.DataFrame):
        if self.games.empty: self.games = games
        else: self.games = pd.concat([self.games, games])

        if self.players.empty: self.players = players
        else: self.players = pd.concat([self.players, players])

    def __is_enough_data(self):
        # 3 seasons is enough data
        return len(self.games['Season'].unique()) >= 3

    def __is_new_season(self, opps):
        curr_season = opps['Season'].iloc[0]
        if self.season == curr_season:
            return False
        else:
            self.season = curr_season
            print("New season: " + str(self.season))
            return True

    def __update_team_historical_strength(self, team_id):
        self.team_historical_strength[team_id]['H WIN PCT'] = (self.team_stats[team_id]['H WIN CNT'] /
                                                               self.team_stats[team_id]['H GM CNT'])
        self.team_historical_strength[team_id]['A WIN PCT'] = (self.team_stats[team_id]['A WIN CNT'] /
                                                               self.team_stats[team_id]['A GM CNT'])
        self.team_historical_strength[team_id]['H GS AVG'] = (self.team_stats[team_id]['H GS'] /
                                                              self.team_stats[team_id]['H GM CNT'])
        self.team_historical_strength[team_id]['A GS AVG'] = (self.team_stats[team_id]['A GS'] /
                                                              self.team_stats[team_id]['A GM CNT'])
        self.team_historical_strength[team_id]['H GC AVG'] = (self.team_stats[team_id]['H GC'] /
                                                              self.team_stats[team_id]['H GM CNT'])
        self.team_historical_strength[team_id]['A GC AVG'] = (self.team_stats[team_id]['A GC'] /
                                                              self.team_stats[team_id]['A GM CNT'])
        self.team_historical_strength[team_id]['H GS STD'] = np.std(self.team_stats[team_id]['H GS HISTORY'])
        self.team_historical_strength[team_id]['A GS STD'] = np.std(self.team_stats[team_id]['A GS HISTORY'])
        self.team_historical_strength[team_id]['H GC STD'] = np.std(self.team_stats[team_id]['H GC HISTORY'])
        self.team_historical_strength[team_id]['A GC STD'] = np.std(self.team_stats[team_id]['A GC HISTORY'])

    def __expected_goal_diff(self, team_id, key: str):
        rating = self.team_pi_rating[team_id][key]
        b, c = 10, 2
        norm = 1 if rating >= 0 else -1
        return norm * (b ** (abs(rating) / c) - 1)

    def __update_pi_rating(self, match: pd.Series):
        team_h = match['HID']
        team_a = match['AID']
        score_h = match['HSC']
        score_a = match['ASC']

        expected_diff = self.__expected_goal_diff(team_h, 'H RTG') - self.__expected_goal_diff(team_a, 'A RTG')
        real_diff = score_h - score_a
        error = abs(real_diff - expected_diff)
        ps = 2 * log10(1 + error)  # c = 2, b = 10
        ps_h = ps if expected_diff < real_diff else -ps
        ps_a = ps if expected_diff > real_diff else -ps

        old_team_h_h = self.team_pi_rating[team_h]['H RTG']
        self.team_pi_rating[team_h]['H RTG'] += ps_h * self.lr
        self.team_pi_rating[team_h]['A RTG'] += (self.team_pi_rating[team_h]['H RTG'] - old_team_h_h) * self.gamma

        old_team_a_a = self.team_pi_rating[team_a]['A RTG']
        self.team_pi_rating[team_a]['A RTG'] += ps_a * self.lr
        self.team_pi_rating[team_a]['H RTG'] += (self.team_pi_rating[team_a]['A RTG'] - old_team_a_a) * self.gamma

    def __delete_pi_ratings(self):
        for key in self.team_pi_rating.keys():
            self.team_pi_rating[key] = { "H RTG": 0.0, "A RTG": 0.0, "EGD": 0.0}

    def __calculate_stats_2_seasons(self, curr_season: int):
        last_2_seasons = [curr_season - 1, curr_season - 2]
        games_2_seasons = self.games[self.games['Season'].isin(last_2_seasons)]

        all_teams_id = set(games_2_seasons['HID']).union(games_2_seasons['AID'])

        for team_id in all_teams_id:
            if team_id not in self.team_stats: self.__init_team(team_id)

            self.team_stats[team_id]['H GM CNT'] = len(games_2_seasons.loc[games_2_seasons['HID'] == team_id])
            self.team_stats[team_id]['A GM CNT'] = len(games_2_seasons.loc[games_2_seasons['AID'] == team_id])
            self.team_stats[team_id]['H WIN CNT'] = len(games_2_seasons.loc[(games_2_seasons['HID'] == team_id) &
                                                                            (games_2_seasons['H'] == 1)])
            self.team_stats[team_id]['A WIN CNT'] = len(games_2_seasons.loc[(games_2_seasons['AID'] == team_id) &
                                                                            (games_2_seasons['A'] == 1)])
            self.team_stats[team_id]['H GS'] = games_2_seasons.loc[games_2_seasons['HID'] == team_id, 'HSC'].sum()
            self.team_stats[team_id]['A GS'] = games_2_seasons.loc[games_2_seasons['AID'] == team_id, 'ASC'].sum()
            self.team_stats[team_id]['H GC'] = games_2_seasons.loc[games_2_seasons['HID'] == team_id, 'ASC'].sum()
            self.team_stats[team_id]['A GC'] = games_2_seasons.loc[games_2_seasons['AID'] == team_id, 'HSC'].sum()
            self.team_stats[team_id]['H GS HISTORY'] = games_2_seasons.loc[
                games_2_seasons['HID'] == team_id, 'HSC'].tolist()
            self.team_stats[team_id]['A GS HISTORY'] = games_2_seasons.loc[
                games_2_seasons['AID'] == team_id, 'ASC'].tolist()
            self.team_stats[team_id]['H GC HISTORY'] = games_2_seasons.loc[
                games_2_seasons['HID'] == team_id, 'ASC'].tolist()
            self.team_stats[team_id]['A GC HISTORY'] = games_2_seasons.loc[
                games_2_seasons['AID'] == team_id, 'HSC'].tolist()

            self.__update_team_historical_strength(team_id)

        self.__delete_pi_ratings()
        for _, row in games_2_seasons.iterrows():
            self.__update_pi_rating(row)

    def add_new_data_from_current_season(self, team_h: int, team_a: int, match: pd.Series):
        if match['H']: self.team_stats[team_h]['H WIN CNT'] += 1
        self.team_stats[team_h]['H GM CNT'] += 1

        if match['A']: self.team_stats[team_a]['A WIN CNT'] += 1
        self.team_stats[team_a]['A GM CNT'] += 1

        self.team_stats[team_h]['H GS'] += match['HSC']
        self.team_stats[team_h]['H GC'] += match['ASC']
        self.team_stats[team_a]['A GS'] += match['ASC']
        self.team_stats[team_a]['A GC'] += match['HSC']
        self.team_stats[team_h]['H GS HISTORY'].append(match['HSC'])
        self.team_stats[team_h]['H GC HISTORY'].append(match['ASC'])
        self.team_stats[team_a]['A GS HISTORY'].append(match['ASC'])
        self.team_stats[team_a]['A GC HISTORY'].append(match['HSC'])

        self.__update_team_historical_strength(team_a)
        self.__update_team_historical_strength(team_h)
        self.__update_pi_rating(match)

    def get_features(self, team_h: int, team_a: int) -> np.ndarray:
        x_features = np.zeros(self.features_n)
        # home team's features
        x_features[0] = self.team_pi_rating[team_h]['H RTG']
        x_features[1] = self.team_pi_rating[team_h]['A RTG']
        # x_features[2] = self.team_historical_strength[team_h]['H WIN PCT']
        # x_features[3] = self.team_historical_strength[team_h]['A WIN PCT']
        # x_features[2] = self.team_historical_strength[team_h]['H GS AVG']
        # x_features[3] = self.team_historical_strength[team_h]['A GS AVG']
        # x_features[4] = self.team_historical_strength[team_h]['H GC AVG']
        # x_features[5] = self.team_historical_strength[team_h]['A GC AVG']
        # x_features[6] = self.team_historical_strength[team_h]['H GS STD']
        # x_features[7] = self.team_historical_strength[team_h]['A GS STD']
        # x_features[8] = self.team_historical_strength[team_h]['H GC STD']
        # x_features[9] = self.team_historical_strength[team_h]['A GC STD']

        # away team's features
        x_features[2] = self.team_pi_rating[team_a]['H RTG']
        x_features[3] = self.team_pi_rating[team_a]['A RTG']
        # x_features[6] = self.team_historical_strength[team_a]['H WIN PCT']
        # x_features[7] = self.team_historical_strength[team_a]['A WIN PCT']
        # x_features[12] = self.team_historical_strength[team_a]['H GS AVG']
        # x_features[13] = self.team_historical_strength[team_a]['A GS AVG']
        # x_features[14] = self.team_historical_strength[team_a]['H GC AVG']
        # x_features[15] = self.team_historical_strength[team_a]['A GC AVG']
        #
        # expected_diff = self.__expected_goal_diff(team_h, 'H RTG') - self.__expected_goal_diff(team_a, 'A RTG')
        # x_features[4] = expected_diff
        # x_features[16] = self.team_historical_strength[team_a]['H GS STD']
        # x_features[17] = self.team_historical_strength[team_a]['A GS STD']
        # x_features[18] = self.team_historical_strength[team_a]['H GC STD']
        # x_features[19] = self.team_historical_strength[team_a]['A GC STD']

        return x_features

    def get_train_data(self) -> (np.array, np.array):
        # TODO: remove rows with incomplete information
        curr_season = self.season - 1

        # get statistics for the last 2 seasons
        self.__calculate_stats_2_seasons(curr_season)

        games_curr_season = self.games[self.games['Season'] == curr_season]
        x_train = np.zeros((games_curr_season.shape[0], self.features_n))
        y_train = np.zeros(games_curr_season.shape[0])

        for i, row in enumerate(games_curr_season.iterrows()):
            match = row[1] # one row of dataframe
            team_h, team_a = match['HID'], match['AID']

            # if we have a new team in the current season we skip these games
            if team_a not in self.team_stats or team_h not in self.team_stats: continue
            # if we have little information about team we skip it
            if self.team_stats[team_a]['H GM CNT'] + self.team_stats[team_a]['A GM CNT'] < self.minimal_games: continue
            if self.team_stats[team_h]['H GM CNT'] + self.team_stats[team_h]['A GM CNT'] < self.minimal_games: continue

            # get features
            x_train[i, :] = self.get_features(team_h, team_a)
            y_train[i] = match['A']

            # add new data from the current season
            self.add_new_data_from_current_season(team_h, team_a, match)

        # remove old season
        self.games = self.games[self.games['Season'].isin([curr_season, curr_season - 1])]

        # reset teams stat before new season
        self.__calculate_stats_2_seasons(self.season)

        return x_train, y_train

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

            if self.team_stats[team_a]['H GM CNT'] + self.team_stats[team_a]['A GM CNT'] < self.minimal_games:
                skipped.append(i)
                continue
            if self.team_stats[team_h]['H GM CNT'] + self.team_stats[team_h]['A GM CNT'] < self.minimal_games:
                skipped.append(i)
                continue

            x_data[i, :] = self.get_features(team_h, team_a)

        return x_data, skipped

    def place_bets(self, summary: pd.DataFrame, opps: pd.DataFrame, inc: tuple[pd.DataFrame, pd.DataFrame]):
        bankroll = summary.iloc[0]['Bankroll']
        min_bet = summary.iloc[0]['Min_bet']
        n = len(opps)

        # store data
        self.__store_inc(inc[0], inc[1])

        # update statistics for teams
        for i, row in enumerate(inc[0].iterrows()):
            match = row[1]
            team_h, team_a = match['HID'], match['AID']
            if team_a not in self.team_stats or team_h not in self.team_stats: continue
            self.add_new_data_from_current_season(team_h, team_a, match)

        # retrain model every season
        if self.__is_new_season(opps) and self.__is_enough_data():
            x_train, y_train = self.get_train_data()
            self.model.fit(x_train, y_train)
            self.has_model = True

        # make prediction
        predictions = np.argmin(opps[['OddsH', 'OddsA']], axis=1)
        no_bet = [np.arange(n)]

        if self.has_model:
            x, no_info = self.get_data(opps)
            predictions = self.model.predict(x)
            probs = self.model.predict_proba(x)
            # confidence threshold
            confidence_threshold = np.where(np.all(probs < 0.5 + self.cnf_threshold, axis=1))[0]
            no_bet = np.union1d(no_info, confidence_threshold).astype(int)
            fractions = get_optimal_fractions(probs, opps[['OddsH', 'OddsA']])

        # chose bets

        # TODO: check if there is already my bet


        # place bets
        bets = np.zeros((n, 2))
        bets[np.arange(n), predictions] = min_bet
        bets[no_bet, :] = 0
        # bets[np.arange(N), np.argmin(opps[['OddsH', 'OddsA']], axis=1)] = min_bet
        bets = pd.DataFrame(data=bets, columns=["BetH", "BetA"], index=opps.index)
        return bets
