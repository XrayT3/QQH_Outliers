import numpy as np
import pandas as pd
from math import log10

class EloModel:
    def __init__(self):
        self.homeAdv = 5
        self.k0 = 6
        self.gamma = 1.45
        self.ratings = {}
        self.divisor = 10
        self.power = 400
        return

    def get_rating(self, team_id):
        if team_id not in self.ratings:
            self.ratings[team_id] = 1500
        return self.ratings[team_id]

    def update_rating(self, game):
        homeRating = self.get_rating(game["HID"])
        awayRating = self.get_rating(game["AID"])

        diff = homeRating - awayRating + self.homeAdv
        expectedA = 1 / (1 + self.divisor ** (-diff / self.power))
        expectedB = 1 - expectedA

        homeScore, awayScore = game["HSC"], game["ASC"]
        k = self.k0 * ((1 + np.abs(homeScore - awayScore)) ** self.gamma)

        self.ratings[game["HID"]] = homeRating + k*(float(game["H"]) - expectedA)
        self.ratings[game["AID"]] = awayRating + k*(float(game["A"]) - expectedB)


class PiRatingModel:
    # TODO: Finish this part. I don't want to spend too much time trying to adopt for general purposes :(

    def __init__(self):
        self.games = pd.DataFrame()
        self.team_pi_rating = {}
        self.season = -1

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

    def __init_team(self, team_id):
        self.team_pi_rating[team_id] = {
                    "H RTG": 0.0,
                    "A RTG": 0.0
        }
    
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

        self.elo_rating.update_rating(match)
    
    def __delete_pi_ratings(self):
        for key in self.team_pi_rating.keys():
            self.team_pi_rating[key] = { "H RTG": 0.0, "A RTG": 0.0, "EGD": 0.0}
        self.elo_rating.remove()

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

    def get_ratings(self, team_id):
        if team_id not in self.team_pi_rating: self.init_team(team_id)
        return (self.team_pi_rating[team_id]['H RTG'], self.team_pi_rating[team_id]['A RTG'])


class AllFeatures:
    def __init__(self):
        self.team_stats = {}
        self.Elo = EloModel()
        self.PiRating = PiRatingModel()
    
    def clean(self):
        self.team_stats = {}
        self.Elo = EloModel()
        self.PiRating = PiRatingModel()

    def _update_team_stats(self, team, team_score, opponent_score, game):
        if team not in self.team_stats:
            self.team_stats[team] = {
                'games_played': 0,
                'total_wins': 0,
                'total_points': 0,
                'total_fg': 0,
                'total_fga': 0,
                'total_3p': 0,
                'total_3pa': 0,
                'total_ft': 0,
                'total_fta': 0,
                'total_orb': 0,
                'total_drb': 0,
                'total_ast': 0,
                'total_stl': 0,
                'total_blk': 0,
                'total_tov': 0,
                'total_pf': 0
            }
        
        home_team = game['HID']
        away_team = game['AID']
        
        if team == home_team:
            is_home_team = True
        elif team == away_team:
            is_home_team = False
        else:
            raise ValueError(f"Team {team} not found in the game data.")
        
        if is_home_team:
            self.team_stats[team]['total_points'] += team_score
            self.team_stats[team]['total_fg'] += game['HFGM']
            self.team_stats[team]['total_fga'] += game['HFGA']
            self.team_stats[team]['total_3p'] += game['HFG3M']
            self.team_stats[team]['total_3pa'] += game['HFG3A']
            self.team_stats[team]['total_ft'] += game['HFTM']
            self.team_stats[team]['total_fta'] += game['HFTA']
            self.team_stats[team]['total_orb'] += game['HORB']
            self.team_stats[team]['total_drb'] += game['HDRB']
            self.team_stats[team]['total_ast'] += game['HAST']
            self.team_stats[team]['total_stl'] += game['HSTL']
            self.team_stats[team]['total_blk'] += game['HBLK']
            self.team_stats[team]['total_tov'] += game['HTOV']
            self.team_stats[team]['total_pf'] += game['HPF']
        else:
            self.team_stats[team]['total_points'] += team_score
            self.team_stats[team]['total_fg'] += game['AFGM']
            self.team_stats[team]['total_fga'] += game['AFGA']
            self.team_stats[team]['total_3p'] += game['AFG3M']
            self.team_stats[team]['total_3pa'] += game['AFG3A']
            self.team_stats[team]['total_ft'] += game['AFTM']
            self.team_stats[team]['total_fta'] += game['AFTA']
            self.team_stats[team]['total_orb'] += game['AORB']
            self.team_stats[team]['total_drb'] += game['ADRB']
            self.team_stats[team]['total_ast'] += game['AAST']
            self.team_stats[team]['total_stl'] += game['ASTL']
            self.team_stats[team]['total_blk'] += game['ABLK']
            self.team_stats[team]['total_tov'] += game['ATOV']
            self.team_stats[team]['total_pf'] += game['APF']
        
        self.team_stats[team]['games_played'] += 1
        
        if team_score > opponent_score:
            self.team_stats[team]['total_wins'] += 1


    def update(self, summary, opps, inc):
        games, players = inc

        for _, game in games.iterrows():
            home_team = game['HID']
            away_team = game['AID']
            home_score = game['HSC']
            away_score = game['ASC']
            
            self._update_team_stats(home_team, home_score, away_score, game)
            self._update_team_stats(away_team, away_score, home_score, game)
        
        for _, game in games.iterrows():
            self.Elo.update_rating(game)
            self.PiRating.update_rating(game)
        
        # TODO: Here we should somehow update our PiRating. too much time for me to update this :(
        self.PiRating.__is_new_season(opps)
        self.PiRating.__calculate_stats_2_seasons(self.season)
    

    def cnt_wins(self, team_id):
        if team_id in self.team_stats:
            return self.team_stats[team_id]['total_wins']
        else:
            return 0
    
    def win_rate(self, team_id):
        if team_id in self.team_stats:
            wins = self.team_stats[team_id]['wins']
            total_matches = self.team_stats[team_id]['matches']
            if total_matches == 0:
                return 0
            return wins / total_matches * 100
        else:
            return 0
    
    def total_matches(self, team_id):
        if team_id in self.team_stats:
            return self.team_stats[team_id]['matches']
        else:
            return 0
    
    def elo_rating(self, team_id):
        return self.Elo.get_rating(team_id)

    def pi_rating(self, team_id):
        return self.PiRating.get_ratings(team_id)

    

