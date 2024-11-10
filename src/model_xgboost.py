import numpy as np
import pandas as pd
from math import log10

from xgboost import XGBClassifier


def get_optimal_fractions(probabilities: np.ndarray, odds: pd.Series) -> np.ndarray:
    # Kelly criterion
    b = odds.to_numpy()
    q = 1 - probabilities
    fractions = probabilities - (q / b)

    var = 0.289
    coefficient = np.square((b+1)*probabilities - 1) / (np.square((b + 1)*probabilities - 1) + np.square((b + 1)*var))
    fractions *= coefficient

    # remove negative values
    fractions[fractions < 0] = 0
    return fractions


class Model:

    def __init_team(self, team_id):
        self.team_stats[team_id] = {
                    "GM CNT": 0,
                    "GM CNT CS": 0
        }
        self.team_pi_rating[team_id] = {
                    "H RTG": 0.0,
                    "A RTG": 0.0
        }

    def __init__(self):
        self.first_try = True
        self.lr = 0.26
        self.gamma = 0.5 # or 0.3
        self.teams_n = 2
        self.cnf_threshold = 0.15
        self.minimal_games = 39 * 4
        self.features_n = 2 * self.teams_n
        self.has_model = False
        self.team_stats = {}
        self.team_pi_rating = {}
        self.games = pd.DataFrame()
        self.season = -1
        self.model = XGBClassifier(max_depth=4, subsample=0.8, min_child_weight=5, colsample_bytree=0.25, seed=24)

    def __store_inc(self, games: pd.DataFrame):
        if self.games.empty: self.games = games
        else: self.games = pd.concat([self.games, games])

    def __is_enough_data(self):
        # 3 seasons is enough data
        return len(self.games['Season'].unique()) >= 3

    def __is_new_season(self, opps):
        curr_season = opps['Season'].iloc[-1]
        if self.season == curr_season:
            return False
        else:
            self.season = curr_season
            return True

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
            if team_id not in self.team_pi_rating: self.__init_team(team_id)

        for team_id in all_teams_id:
            if team_id not in self.team_stats: self.__init_team(team_id)

            h_gm_cnt = len(games_2_seasons.loc[games_2_seasons['HID'] == team_id])
            a_gm_cnt = len(games_2_seasons.loc[games_2_seasons['AID'] == team_id])
            self.team_stats[team_id]['GM CNT'] = h_gm_cnt + a_gm_cnt
            self.team_stats[team_id]['GM CNT CS'] = 0

        self.__delete_pi_ratings()
        for _, row in games_2_seasons.iterrows():
            self.__update_pi_rating(row)

    def add_new_data_from_current_season(self, team_h: int, team_a: int, match: pd.Series):
        self.team_stats[team_h]['GM CNT CS'] += 1
        self.team_stats[team_a]['GM CNT CS'] += 1

        self.__update_pi_rating(match)

    def get_features(self, team_h: int, team_a: int) -> np.ndarray:
        x_features = np.zeros(self.features_n)
        # home team's features
        x_features[0] = self.team_pi_rating[team_h]['H RTG']
        x_features[1] = self.team_pi_rating[team_h]['A RTG']
        # x_features[2] = self.team_historical_strength[team_h]['H WIN PCT']
        # x_features[3] = self.team_historical_strength[team_h]['A WIN PCT']

        # away team's features
        x_features[2] = self.team_pi_rating[team_a]['H RTG']
        x_features[3] = self.team_pi_rating[team_a]['A RTG']
        # x_features[6] = self.team_historical_strength[team_a]['H WIN PCT']
        # x_features[7] = self.team_historical_strength[team_a]['A WIN PCT']

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
            # # if we have little information about team we skip it
            # if self.team_stats[team_a]['GM CNT'] < self.minimal_games: continue
            # if self.team_stats[team_h]['GM CNT'] < self.minimal_games: continue

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

            if self.team_stats[team_a]['GM CNT'] < self.minimal_games:
                skipped.append(i)
                continue
            if self.team_stats[team_h]['GM CNT'] < self.minimal_games:
                skipped.append(i)
                continue

            x_data[i, :] = self.get_features(team_h, team_a)

        return x_data, skipped

    def place_bets(self, summary: pd.DataFrame, opps: pd.DataFrame, inc: tuple[pd.DataFrame, pd.DataFrame]):
        bankroll = summary.iloc[0]['Bankroll']
        min_bet = summary.iloc[0]['Min_bet']
        max_bet = summary.iloc[0]['Max_bet']
        n = len(opps)
        bets = np.zeros((n, 2))

        # store data
        self.__store_inc(inc[0])

        # update statistics for teams
        if not self.first_try: # for qualification upload
            for i, row in enumerate(inc[0].iterrows()):
                match = row[1]
                team_h, team_a = match['HID'], match['AID']
                if team_a not in self.team_stats or team_h not in self.team_stats: continue
                self.add_new_data_from_current_season(team_h, team_a, match)
        self.first_try = False

        # retrain model every season
        if self.__is_new_season(opps) and self.__is_enough_data():
            x_train, y_train = self.get_train_data()
            self.model.fit(x_train, y_train)
            self.has_model = True

        # skip betting on training seasons
        if not self.has_model:
            bets = pd.DataFrame(data=bets, columns=["BetH", "BetA"], index=opps.index)
            return bets

        # make prediction
        x, no_info = self.get_data(opps)
        probs = self.model.predict_proba(x)
        # confidence threshold
        confidence_threshold = np.where(np.all(probs < 0.5 + self.cnf_threshold, axis=1))[0]
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
