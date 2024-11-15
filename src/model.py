import numpy as np
import pandas as pd

class EloModel:
    def __init__(self):
        self.homeAdv = 5
        self.k0 = 6
        self.gamma = 1.45
        self.ratings = {}
        self.cntMatches = {}
        self.teamMembers = {}
        self.divisor = 10
        self.power = 400
        return

    def clean(self):
        self.ratings = {}
        self.cntMatches = {}

    def get_rating(self, team_id):
        if team_id not in self.ratings:
            self.ratings[team_id] = 1500
        return self.ratings[team_id]

    def get_cnt_matches(self, team_id):
        if team_id not in self.cntMatches:
            self.cntMatches[team_id] = 0
        return self.cntMatches[team_id]

    def get_team_members(self, team_id):
        if team_id not in self.teamMembers:
            self.teamMembers[team_id] = []
        return self.teamMembers[team_id]

    def update_rating(self, game: dict):
        homeRating = self.get_rating(game["HID"])
        awayRating = self.get_rating(game["AID"])

        diff = homeRating - awayRating + self.homeAdv
        expectedA = 1 / (1 + self.divisor ** (-diff / self.power))
        expectedB = 1 - expectedA

        homeScore, awayScore = game["HSC"], game["ASC"]
        k = self.k0 * ((1 + np.abs(homeScore - awayScore)) ** self.gamma)

        self.ratings[game["HID"]] = homeRating + k*(float(game["H"]) - expectedA)
        self.ratings[game["AID"]] = awayRating + k*(float(game["A"]) - expectedB)

        self.cntMatches[game["HID"]] = self.get_cnt_matches(game["HID"]) + 1
        self.cntMatches[game["AID"]] = self.get_cnt_matches(game["AID"]) + 1

    def get_probability(self, teamA, teamB):
        if self.get_cnt_matches(teamA) < 3 or self.get_cnt_matches(teamB) < 3: return 0.0
        ratingA, ratingB = self.get_rating(teamA), self.get_rating(teamB)
        diff = ratingA - ratingB
        if diff > 10000: exit(1)
        probability = 1 - 1 / (1 + np.exp(0.00583 * diff - 0.0505) )
        return float(probability)

class Model:
    def __init__(self):
        self.Elo = EloModel()
        self.minBet = 0
        self.maxBet = 0
        self.bankroll = 0
        self.minConfidence = 0.7
        self.curSeason = 0

    def get_optimal_fractions(self, probabilities: np.ndarray, odds: pd.Series) -> np.ndarray:
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

    def set_bets(self, probabilities, fractions, betSize):
        bets = np.array([0, 0], dtype=int)
        if probabilities[0] >= self.minConfidence:
            # print(probabilities[0])
            bets[0] = fractions[0] * betSize
        if probabilities[1] >= self.minConfidence:
            # print(probabilities[1])
            bets[1] = fractions[1] * betSize

        for id in [0, 1]:
            if bets[id] < self.minBet: bets[id] = 0
            if bets[id] > self.maxBet: bets[id] = self.maxBet
            bets[id] = int(bets[id])

        return bets

    def place_bets(self, summary: pd.DataFrame, opps: pd.DataFrame, inc: tuple[pd.DataFrame, pd.DataFrame]):
        self.minBet, self.maxBet = summary.iloc[0]["Min_bet"], summary.iloc[0]["Max_bet"]
        self.bankroll = summary.iloc[0]["Bankroll"]
        N = len(opps)
        bets = np.zeros((opps.shape[0], 2))
        games_data, players_data = inc
        
        cur_teams = {}
        for index, row in players_data.iterrows():
            if row["Team"] not in cur_teams:
                cur_teams[row["Team"]] = []
            cur_teams[row["Team"]].append(row["Player"])
        
        for team_id in cur_teams.keys():
            cur_teams[team_id] = sorted(cur_teams[team_id])
            if self.Elo.get_team_members(team_id) != cur_teams[team_id]:
                self.Elo.ratings[team_id] = 0
                self.Elo.cntMatches[team_id] = 0
                self.Elo.teamMembers[team_id] = cur_teams[team_id]
        

        for index, row in games_data.iterrows():
            if self.curSeason != row["Season"]:
                self.Elo.clean()
                self.curSeason = row["Season"]

            game_train = pd.Series.to_dict(row)
            self.Elo.update_rating(game_train)

        # print(opps)

        id = -1
        for index_row, row in opps.iterrows():
            id += 1
            game_predict = pd.Series.to_dict(row)
            chanceH = self.Elo.get_probability(game_predict["HID"], game_predict["AID"])
            chanceA = self.Elo.get_probability(game_predict["AID"], game_predict["HID"])

            probabilities = np.array([chanceH, chanceA], dtype=float)
            odds = pd.Series([game_predict["OddsH"], game_predict["OddsA"]])

            budget = self.bankroll * 0.1
            current = budget / opps.shape[0]
            # print(id, "   << ", N)
            bets[id] = self.set_bets(probabilities,
                                        self.get_optimal_fractions(probabilities, odds),
                                        current)

            # print(probabilities)
            # print(odds)
            # print()
            # print(bets[id])
            # exit(1)

        # print("\n\n")

        bets = pd.DataFrame(data=bets, columns=["BetH", "BetA"], index=opps.index)
        return bets
    