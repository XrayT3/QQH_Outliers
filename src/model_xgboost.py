import numpy as np
import pandas as pd
from math import log10
from scipy.optimize import minimize

from xgboost import XGBClassifier


def get_optimal_fractions(probabilities: np.ndarray, odds: pd.Series) -> np.ndarray:
    # Kelly criterion
    b = odds.to_numpy()
    q = 1 - probabilities
    fractions = probabilities - (q / b)

    var = 0.289
    coefficient = np.square((b + 1) * probabilities - 1) / (
                np.square((b + 1) * probabilities - 1) + np.square((b + 1) * var))
    fractions *= coefficient

    # remove negative values
    fractions[fractions < 0] = 0
    return fractions

def sharp_betting_strategy(probabilities: np.ndarray):
    n = probabilities.shape[0]

    def objective(f):
        # Calculate expected return of the portfolio
        expected_return = np.sum(f * probabilities.flatten())
        # Calculate portfolio standard deviation (risk)
        portfolio_variance = np.sum(f ** 2)  # Simple risk approximation
        portfolio_std = np.sqrt(portfolio_variance)
        # Return negative Sharpe ratio (since we're minimizing)
        return - (expected_return / portfolio_std)

    # Bounds: weights should be between 0 and 1 (no short selling)
    bounds = [(0.0, 1.0) for _ in range(2 * n)]
    # Constraint: sum of weights must be 1
    constraints = ({'type': 'eq', 'fun': lambda f: np.sum(f) - 1})
    # Initial guess (equal distribution)
    initial_guess = np.ones(2 * n) / (2 * n)

    result = minimize(objective, initial_guess, bounds=bounds, constraints=constraints)
    # Return the optimized weights
    return result.x.reshape((n, 2)) if result.success else None


class EloModel:
    def __init__(self):
        self.homeAdv = 5
        self.k0 = 6
        self.gamma = 1.45
        self.divisor = 10
        self.power = 400

        self.ratings = {}

    def remove(self):
        for key in self.ratings.keys():
            self.ratings[key] = 1500

    def update_rating(self, game):
        homeRating = self.ratings[game["HID"]]
        awayRating = self.ratings[game["AID"]]
        homeScore, awayScore = game["HSC"], game["ASC"]

        diff = homeRating - awayRating + self.homeAdv
        expected = 1 / (1 + self.divisor ** (-diff / self.power))

        k = self.k0 * ((1 + np.abs(homeScore - awayScore)) ** self.gamma)

        self.ratings[game["HID"]] = homeRating + k * (game["A"] - expected)
        self.ratings[game["AID"]] = awayRating - k * (game["A"] - expected)


class Berrar:
    def __init__(self):
        self.att_ratings = {}
        self.def_ratings = {}
        self.beta = 0.11
        self.gamma = 3.6
        self.omega_att = 0.54
        self.omega_def = 0.3

    def remove(self):
        for key in self.att_ratings.keys():
            self.att_ratings[key] = 1500
            self.def_ratings[key] = 1500

    def update_rating(self, match):
        def eXg(att_rtg, def_rtg, alfa, beta, gamma):
            return alfa / (1 + np.exp(-beta * (att_rtg + def_rtg) - gamma))

        alpha = 5

        HID, AID, HSC, ASC = match['HID'], match['AID'], match['HSC'], match['ASC']

        artg_h = self.att_ratings[HID]
        drtg_h = self.def_ratings[HID]
        artg_a = self.att_ratings[AID]
        drtg_a = self.def_ratings[AID]

        gh = eXg(artg_h, drtg_a, alpha, self.beta, self.gamma)
        ga = eXg(artg_a, drtg_h, alpha, self.beta, self.gamma)

        self.att_ratings[HID] += self.omega_att * (HSC - gh)
        self.def_ratings[HID] += self.omega_def * (ASC - ga)
        self.att_ratings[AID] += self.omega_att * (ASC - ga)
        self.def_ratings[AID] += self.omega_def * (HSC - gh)


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
        self.elo_rating.ratings[team_id] = 1500
        self.berrar.def_ratings[team_id] = 1500
        self.berrar.att_ratings[team_id] = 1500

    def __init__(self):
        self.prev_bets_cnt = 0
        self.first_try = True
        self.lr = 0.26
        self.gamma = 0.5  # or 0.3
        self.teams_n = 2
        self.cnf_threshold = 0.2
        self.window = 2
        self.games_in_season = 78
        self.minimal_games = self.games_in_season * self.window
        self.features_n = 1 * self.teams_n
        self.has_model = False
        self.team_stats = {}
        self.team_pi_rating = {}
        self.elo_rating = EloModel()
        self.berrar = Berrar()
        self.games = pd.DataFrame()
        self.season = -1
        self.wins_cnt = np.zeros((100, 100))
        self.games_cnt = np.zeros((100, 100))
        # self.model = XGBClassifier(max_depth=4, subsample=0.8, min_child_weight=5, colsample_bytree=0.25, seed=24)
        # self.model = XGBClassifier(eta=0.2, max_depth=3, min_child_weight=5, colsample_bytree=0.5, seed=24)
        self.model_pi_rating = XGBClassifier(eta=0.2, max_depth=3, min_child_weight=5, colsample_bytree=0.5, seed=24)
        self.model_page_rank = XGBClassifier(eta=0.2, max_depth=3, min_child_weight=5, colsample_bytree=0.5, seed=24)
        # self.model_berrar = XGBClassifier(eta=0.2, max_depth=3, min_child_weight=5, colsample_bytree=0.5, seed=24)
        self.model_elo = XGBClassifier(eta=0.2, max_depth=3, min_child_weight=5, colsample_bytree=0.5, seed=24)


    def __store_inc(self, games: pd.DataFrame):
        if self.games.empty: self.games = games
        else: self.games = pd.concat([self.games, games])

    def __is_enough_data(self):
        # 3 seasons is enough data
        return len(self.games['Season'].unique()) >= self.window + 1

    def __is_new_season(self, data):
        curr_season = data['Season'].iloc[-1]
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

    def __update_page_rank(self, match: pd.Series):
        hid, aid = match['HID'], match['AID']

        self.games_cnt[hid, aid] += 1
        self.games_cnt[aid, hid] += 1

        if match['H']: self.wins_cnt[hid, aid] += 1
        if match['A']: self.wins_cnt[aid, hid] += 1

    def __delete_pi_ratings(self):
        for key in self.team_pi_rating.keys():
            self.team_pi_rating[key] = {"H RTG": 0.0, "A RTG": 0.0, "EGD": 0.0}

    def __delete_page_rank(self):
        self.wins_cnt = np.zeros((100, 100))
        self.games_cnt = np.zeros((100, 100))

    def __calculate_stats_2_seasons(self, curr_season: int):
        last_2_seasons = [curr_season - 1, curr_season - 2]
        games_2_seasons = self.games[self.games['Season'].isin(last_2_seasons)]

        all_teams_id = set(games_2_seasons['HID']).union(games_2_seasons['AID'])

        for team_id in all_teams_id:
            if team_id not in self.team_stats: self.__init_team(team_id)

            h_gm_cnt = len(games_2_seasons.loc[games_2_seasons['HID'] == team_id])
            a_gm_cnt = len(games_2_seasons.loc[games_2_seasons['AID'] == team_id])
            self.team_stats[team_id]['GM CNT'] = h_gm_cnt + a_gm_cnt
            self.team_stats[team_id]['GM CNT CS'] = 0


        self.__delete_pi_ratings()
        self.__delete_page_rank()
        self.elo_rating.remove()
        self.berrar.remove()
        for _, row in games_2_seasons.iterrows():
            self.__update_pi_rating(row)
            self.__update_page_rank(row)
            self.elo_rating.update_rating(row)
            self.berrar.update_rating(row)

    def add_new_data_from_current_season(self, team_h: int, team_a: int, match: pd.Series):
        self.team_stats[team_h]['GM CNT CS'] += 1
        self.team_stats[team_a]['GM CNT CS'] += 1

        self.__update_pi_rating(match)
        self.__update_page_rank(match)
        self.elo_rating.update_rating(match)
        self.berrar.update_rating(match)

    def get_features(self, team_h: int, team_a: int):
        # x_features = np.zeros(self.features_n)
        x_features_pi = np.zeros(4)
        x_features_page = np.zeros(2)
        # x_features_berrar = np.zeros(4)
        x_features_elo = np.zeros(2)

        # home team's features
        x_features_pi[0] = self.team_pi_rating[team_h]['H RTG']
        x_features_pi[1] = self.team_pi_rating[team_h]['A RTG']

        # away team's features
        x_features_pi[2] = self.team_pi_rating[team_a]['H RTG']
        x_features_pi[3] = self.team_pi_rating[team_a]['A RTG']

        x_features_page[0] = self.wins_cnt[team_h, team_a] / self.games_cnt[team_h, team_a]
        x_features_page[1] = self.wins_cnt[team_a, team_h] / self.games_cnt[team_a, team_h]

        # x_features_berrar[0] = self.berrar.att_ratings[team_a]
        # x_features_berrar[1] = self.berrar.def_ratings[team_a]
        # x_features_berrar[2] = self.berrar.att_ratings[team_h]
        # x_features_berrar[3] = self.berrar.def_ratings[team_h]

        x_features_elo[0] = self.elo_rating.ratings[team_h]
        x_features_elo[1] = self.elo_rating.ratings[team_a]

        return x_features_pi, x_features_page, x_features_elo

    def get_train_data(self) -> (np.array, np.array, np.array, np.array, np.array):
        curr_season = self.season - 1

        # get statistics for the last 2 seasons
        self.__calculate_stats_2_seasons(curr_season)

        games_curr_season = self.games[self.games['Season'] == curr_season]
        x_train_pi = np.zeros((games_curr_season.shape[0], 4))
        x_train_page = np.zeros((games_curr_season.shape[0], 2))
        # x_train_berrar = np.zeros((games_curr_season.shape[0], 4))
        x_train_elo = np.zeros((games_curr_season.shape[0], 2))
        y_train = np.zeros(games_curr_season.shape[0])
        good_rows = np.ones(games_curr_season.shape[0], dtype=bool)

        for i, row in enumerate(games_curr_season.iterrows()):
            match = row[1]  # one row of dataframe
            team_h, team_a = match['HID'], match['AID']

            # if we have a new team in the current season we skip these games
            if team_a not in self.team_stats or team_h not in self.team_stats:
                good_rows[i] = 0
                continue

            # # if we have little information about team we skip it
            if self.team_stats[team_a]['GM CNT'] < self.minimal_games:
                good_rows[i] = 0
                continue
            if self.team_stats[team_h]['GM CNT'] < self.minimal_games:
                good_rows[i] = 0
                continue

            # get features
            x_train_pi[i, :], x_train_page[i, :], x_train_elo[i, :] = self.get_features(team_h, team_a)
            y_train[i] = match['A']

            # add new data from the current season
            self.add_new_data_from_current_season(team_h, team_a, match)

        x_train_pi = x_train_pi[good_rows, :]
        x_train_page = x_train_page[good_rows, :]
        # x_train_berrar = x_train_berrar[good_rows, :]
        x_train_elo = x_train_elo[good_rows, :]
        y_train_without_zeros = y_train[good_rows]

        # remove old season
        self.games = self.games[self.games['Season'].isin([curr_season, curr_season - 1])]

        # reset teams stat before new season
        self.__calculate_stats_2_seasons(self.season)

        return x_train_pi, x_train_page, x_train_elo, y_train_without_zeros

    def get_data(self, opps: pd.DataFrame):
        x_data_pi = np.zeros((opps.shape[0], 4))
        x_data_page = np.zeros((opps.shape[0], 2))
        # x_data_berrar = np.zeros((opps.shape[0], 4))
        x_data_elo = np.zeros((opps.shape[0], 2))
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

            x_data_pi[i, :], x_data_page[i, :], x_data_elo[i, :] = self.get_features(team_h, team_a)

        return x_data_pi, x_data_page, x_data_elo, skipped

    def place_bets(self, summary: pd.DataFrame, opps: pd.DataFrame, inc: tuple[pd.DataFrame, pd.DataFrame]):
        bankroll = summary.iloc[0]['Bankroll']
        min_bet = summary.iloc[0]['Min_bet']
        max_bet = summary.iloc[0]['Max_bet']
        n = len(opps)
        bets = np.zeros((n, 2))

        # store data
        self.__store_inc(inc[0])

        # update statistics for teams
        if not self.first_try:  # for qualification upload
            for i, row in enumerate(inc[0].iterrows()):
                match = row[1]
                team_h, team_a = match['HID'], match['AID']
                if team_a not in self.team_stats or team_h not in self.team_stats: continue
                self.add_new_data_from_current_season(team_h, team_a, match)
        self.first_try = False

        # retrain model every season
        if opps.empty: data = inc[0]
        else: data = opps
        if self.__is_new_season(data) and self.__is_enough_data():
            x_train_pi, x_train_page, x_train_elo, y_train = self.get_train_data()
            self.model_pi_rating.fit(x_train_pi, y_train)
            self.model_page_rank.fit(x_train_page, y_train)
            # self.model_berrar.fit(x_train_berrar, y_train)
            self.model_elo.fit(x_train_elo, y_train)
            self.has_model = True

        if not opps.empty:
            big_odds_idx = np.where(np.all(opps[['OddsH', 'OddsA']] > 1.6, axis=1))[0]
            big_n = len(big_odds_idx)
            a = self.prev_bets_cnt + 1
            b = self.prev_bets_cnt + big_n

            if big_n > 0:
                if a == 1:
                    bets[big_odds_idx[0], 0] = max_bet
                if a <= 2 <= b:
                    bets[big_odds_idx[2 - a], 0] = max_bet
                if a <= 3 <= b:
                    bets[big_odds_idx[3 - a], 1] = max_bet
                if a <= 4 <= b:
                    bets[big_odds_idx[4 - a], 1] = max_bet
                if a <= 5 <= b:
                    bets[big_odds_idx[5 - a], 0] = max_bet
                if a <= 6 <= b:
                    bets[big_odds_idx[6 - a], 0] = max_bet
                if a <= 7 <= b:
                    bets[big_odds_idx[7 - a], 1] = max_bet
                if a <= 8 <= b:
                    bets[big_odds_idx[8 - a], 0] = max_bet
                if a <= 9 <= b:
                    bets[big_odds_idx[9 - a], 1] = max_bet
                if a <= 10 <= b:
                    bets[big_odds_idx[10 - a], 1] = max_bet
                if a <= 11 <= b:
                    bets[big_odds_idx[11 - a], 0] = max_bet

                self.prev_bets_cnt += big_n
            return pd.DataFrame(data=bets, columns=["BetH", "BetA"], index=opps.index)

        # skip betting on training seasons
        if not self.has_model or opps.empty:
            bets = pd.DataFrame(data=bets, columns=["BetH", "BetA"], index=opps.index)
            return bets

        # make prediction
        x1, x2, x4, no_info = self.get_data(opps)
        probs1 = self.model_pi_rating.predict_proba(x1)
        probs2 = self.model_page_rank.predict_proba(x2)
        # probs3 = self.model_berrar.predict_proba(x3)
        probs4 = self.model_elo.predict_proba(x4)
        # probs = (probs1 + probs2 + probs4) / 3
        probs = 0.0*probs1 + 1.0*probs2 + 0.0*probs4
        # confidence threshold
        confidence_threshold = np.where(np.all(probs < 0.5 + self.cnf_threshold, axis=1))[0]
        no_bet = np.union1d(no_info, confidence_threshold).astype(int)

        # chose bets
        prev_bets = opps[['BetH', 'BetA']].to_numpy()
        fractions = get_optimal_fractions(probs, opps[['OddsH', 'OddsA']])
        # fractions = sharp_betting_strategy(probs)
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
