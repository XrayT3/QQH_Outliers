import numpy as np
import pandas as pd
from scipy.optimize import minimize

import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def kelly_betting_strategy(probabilities: np.ndarray, odds: pd.Series) -> np.ndarray:
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


def uniform_betting_strategy(probabilities: np.ndarray, odds: pd.Series):
    books_pst = odds.to_numpy()
    books_pst = 1 / books_pst
    margin = books_pst.sum(axis=1)
    books_pst[:, 0] = books_pst[:, 0] / margin
    books_pst[:, 1] = books_pst[:, 1] / margin

    my_pst = np.empty(books_pst.shape)
    my_pst[:, 1] = probabilities
    my_pst[:, 0] = 1 - probabilities

    return (my_pst - books_pst > 0).astype(int)


def sharp_betting_strategy(probabilities: np.ndarray):
    n = probabilities.shape[0]

    my_pst = np.empty((n, 2))
    my_pst[:, 1] = probabilities
    my_pst[:, 0] = 1 - probabilities

    def objective(f):
        # Calculate expected return of the portfolio
        expected_return = np.sum(f * my_pst.flatten())
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
    # if not result.success:
        # print(result.message)
        # print(my_pst)
        # print(n)
    return result.x.reshape((n, 2)) if result.success else None


def coefficients_to_probs(coefficient1, coefficient2):
    p1 = 1 / coefficient1
    p2 = 1 / coefficient2
    p2_norm = p2 / (p1 + p2)
    return p2_norm


# Function to set the random seed
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Custom loss function
class CustomMSELoss(nn.Module):
    def __init__(self, gamma=0.65):
        super(CustomMSELoss, self).__init__()
        self.gamma = gamma

    def forward(self, predictions, targets):
        r = targets[:, 0]  # True class label (0 or 1)
        m = targets[:, 1]  # Bookmaker's probability
        t = predictions  # Model's predicted probability

        # Compute the custom MSE loss
        loss = torch.mean((t - r) ** 2 - self.gamma * (t - m) ** 2)
        return loss


class TeamLevelNN(nn.Module):
    def __init__(self):
        super(TeamLevelNN, self).__init__()
        # Define layers
        self.model = nn.Sequential(
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Dropout(0.2),

            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Dropout(0.2),

            nn.Linear(16, 1),
            nn.Sigmoid()  # Output probability between 0 and 1
        )

    def forward(self, x):
        return self.model(x)


def train_model(model: TeamLevelNN, x_train, y_train, seed=42, epochs=100, batch_size=1024, lr=0.001):
    # Set seed for reproducibility
    set_seed(seed)

    # Convert numpy arrays to PyTorch tensors
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    # Create DataLoader
    dataset = TensorDataset(x_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, custom loss function, and optimizer
    # model = TeamLevelNN()
    criterion = CustomMSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)

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

        # if (epoch + 1) % 20 == 0:
            # print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.3f}')

    # return model


def classify(model, x_test):
    # Convert test data to PyTorch tensor
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)

    model.eval()
    with torch.no_grad():  # Disable gradient calculation for inference
        probs = model(x_test_tensor).squeeze()

    # Ensure probs is always returned as a numpy array with at least 1 dimension
    probs = np.asarray(probs).reshape(-1)
    # Check for NaN values and replace them with 0.5
    if np.any(np.isnan(probs)):
        probs = np.nan_to_num(probs, nan=0.5)

    return probs


class Model:

    def __init_team(self, team_id):
        self.team_stats[team_id] = {
            'WIN': 0,
            'GM CNT': 0,
            'AST': 0,
            'BLK': 0,
            'DRB': 0,
            'FGA': 0,
            'FGM': 0,
            'FG3A': 0,
            'FG3M': 0,
            'FTA': 0,
            'FTM': 0,
            'ORB': 0,
            'PF': 0,
            'PM': 0,
            'PTS': 0,
            'RB': 0,
            'STL': 0,
            'TOV': 0,
            'OPP_FG3M': 0,
            'OPP_FGM': 0,
            'OPP_FGA': 0,
            'OPP_FTA': 0,
            'OPP_RB': 0,
            'OPP_ORB': 0,
        }

    def __init__(self):
        self.season = 0
        self.year = 1900
        self.team_stats = {}
        self.train_data = [[], []]  # [[X_1, X_2, ...], [y_1, y_2, ...]]
        self.first_try = True
        self.has_model = False
        self.minimal_games = 10
        self.cnf_threshold = 0.4
        self.trained_seasons = []
        self.games = pd.DataFrame()
        self.model = TeamLevelNN()
        self.features_n = 64

    def __store_inc(self, games: pd.DataFrame):
        if games.empty: return

        if self.games.empty:
            self.games = games
        else:
            self.games = pd.concat([self.games, games])

    def __is_enough_data(self):
        # 5 seasons is enough data
        return len(self.train_data[0]) >= 5

    def __is_new_season(self, future: pd.DataFrame, past: pd.DataFrame):
        if future.empty: curr_season = past.iloc[0]['Season']
        else: curr_season = future.iloc[0]['Season']

        if curr_season > self.season:
            self.season = curr_season
            return True
        else:
            return False

    def __update_stats(self, match: pd.Series, team_h: int, team_a: int):
        if team_h not in self.team_stats: self.__init_team(team_h)
        if team_a not in self.team_stats: self.__init_team(team_a)

        self.team_stats[team_h]['GM CNT'] += 1
        self.team_stats[team_a]['GM CNT'] += 1
        self.team_stats[team_h]['WIN'] += match['H']
        self.team_stats[team_a]['WIN'] += match['A']
        # Basic statistics
        self.team_stats[team_h]['AST'] += match['HAST']
        self.team_stats[team_a]['AST'] += match['AAST']
        self.team_stats[team_h]['BLK'] += match['HBLK']
        self.team_stats[team_a]['BLK'] += match['ABLK']
        self.team_stats[team_h]['DRB'] += match['HDRB']
        self.team_stats[team_a]['DRB'] += match['ADRB']
        self.team_stats[team_h]['FGA'] += match['HFGA']
        self.team_stats[team_a]['FGA'] += match['AFGA']
        self.team_stats[team_h]['FGM'] += match['HFGM']
        self.team_stats[team_a]['FGM'] += match['AFGM']
        self.team_stats[team_h]['FG3A'] += match['HFG3A']
        self.team_stats[team_a]['FG3A'] += match['AFG3A']
        self.team_stats[team_h]['FG3M'] += match['HFG3M']
        self.team_stats[team_a]['FG3M'] += match['AFG3M']
        self.team_stats[team_h]['FTA'] += match['HFTA']
        self.team_stats[team_a]['FTA'] += match['AFTA']
        self.team_stats[team_h]['FTM'] += match['HFTM']
        self.team_stats[team_a]['FTM'] += match['AFTM']
        self.team_stats[team_h]['ORB'] += match['HORB']
        self.team_stats[team_a]['ORB'] += match['AORB']
        self.team_stats[team_h]['PF'] += match['HPF']
        self.team_stats[team_a]['PF'] += match['APF']
        self.team_stats[team_h]['PM'] += match['HSC'] - match['ASC']
        self.team_stats[team_a]['PM'] += match['ASC'] - match['HSC']
        self.team_stats[team_h]['PTS'] += match['HSC']
        self.team_stats[team_a]['PTS'] += match['ASC']
        self.team_stats[team_h]['RB'] += match['HRB']
        self.team_stats[team_a]['RB'] += match['ARB']
        self.team_stats[team_h]['STL'] += match['HSTL']
        self.team_stats[team_a]['STL'] += match['ASTL']
        self.team_stats[team_h]['TOV'] += match['HTOV']
        self.team_stats[team_a]['TOV'] += match['ATOV']
        # Advanced statistics
        self.team_stats[team_h]['OPP_FG3M'] += match['AFG3M']
        self.team_stats[team_a]['OPP_FG3M'] += match['HFG3M']
        self.team_stats[team_h]['OPP_FGM'] += match['AFGM']
        self.team_stats[team_a]['OPP_FGM'] += match['HFGM']
        self.team_stats[team_h]['OPP_FGA'] += match['AFGA']
        self.team_stats[team_a]['OPP_FGA'] += match['HFGA']
        self.team_stats[team_h]['OPP_FTA'] += match['AFTA']
        self.team_stats[team_a]['OPP_FTA'] += match['HFTA']
        self.team_stats[team_h]['OPP_RB'] += match['ARB']
        self.team_stats[team_a]['OPP_RB'] += match['HRB']
        self.team_stats[team_h]['OPP_ORB'] += match['AORB']
        self.team_stats[team_a]['OPP_ORB'] += match['HORB']

    def __calculate_pie(self, team_id: int):
        pie = (self.team_stats[team_id]['PTS'] + self.team_stats[team_id]['FGM'] + self.team_stats[team_id]['FTM'] -
               self.team_stats[team_id]['FGA'] - self.team_stats[team_id]['FTA'] + self.team_stats[team_id]['DRB'] +
               self.team_stats[team_id]['ORB'] / 2 + self.team_stats[team_id]['AST'] + self.team_stats[team_id]['STL'] +
               self.team_stats[team_id]['BLK'] / 2 - self.team_stats[team_id]['PF'] - self.team_stats[team_id]['TOV'])
        avg_pie = pie / self.team_stats[team_id]['GM CNT']
        return avg_pie

    def __process_one_season(self, season_df: pd.DataFrame):
        x_data = np.empty((season_df.shape[0], self.features_n))
        y_data = np.empty((season_df.shape[0], 2))
        skipped_rows = []

        # remove rows with incomplete information
        season_df = season_df.dropna()

        for i, row in enumerate(season_df.iterrows()):
            match = row[1]
            team_h, team_a = match['HID'], match['AID']
            if team_h not in self.team_stats: self.__init_team(team_h)
            if team_a not in self.team_stats: self.__init_team(team_a)

            if self.team_stats[team_a]['GM CNT'] > self.minimal_games and self.team_stats[team_h][
                'GM CNT'] > self.minimal_games:
                x_data[i, :] = self.get_features(team_h, team_a, match['OddsH'], match['OddsA'])
                y_data[i] = match['A'], coefficients_to_probs(match['OddsH'], match['OddsA'])
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

    def get_features(self, team_h: int, team_a: int, odd_h, odd_a) -> np.ndarray:
        x_features = np.empty(self.features_n)

        game_cnt_h = self.team_stats[team_h]['GM CNT']
        game_cnt_a = self.team_stats[team_a]['GM CNT']

        # BASIC STATISTICS
        x_features[0] = self.team_stats[team_h]['AST'] / game_cnt_h
        x_features[1] = self.team_stats[team_a]['AST'] / game_cnt_a
        x_features[2] = self.team_stats[team_h]['BLK'] / game_cnt_h
        x_features[3] = self.team_stats[team_a]['BLK'] / game_cnt_a
        # DREB: Defense Rebounds
        x_features[4] = self.team_stats[team_h]['DRB'] / game_cnt_h
        x_features[5] = self.team_stats[team_a]['DRB'] / game_cnt_a
        # FG_PCT: Field Goals Made / Field Goals Attempted
        x_features[6] = self.team_stats[team_h]['FGM'] / self.team_stats[team_h]['FGA']
        x_features[7] = self.team_stats[team_a]['FGM'] / self.team_stats[team_a]['FGA']
        # FG3_PCT: 3 points Field Goals Made / 3 points Field Goals Attempted
        x_features[8] = self.team_stats[team_h]['FG3M'] / self.team_stats[team_h]['FG3A']
        x_features[9] = self.team_stats[team_a]['FG3M'] / self.team_stats[team_a]['FG3A']
        # FG3A: 3 points Filed Goals Attempted
        x_features[10] = self.team_stats[team_h]['FG3A'] / game_cnt_h
        x_features[11] = self.team_stats[team_a]['FG3A'] / game_cnt_a
        # FG3M: 3 points Filed Goals Made
        x_features[12] = self.team_stats[team_h]['FG3M'] / game_cnt_h
        x_features[13] = self.team_stats[team_a]['FG3M'] / game_cnt_a
        # FGA: Filed Goals Attempted
        x_features[14] = self.team_stats[team_h]['FGA'] / game_cnt_h
        x_features[15] = self.team_stats[team_a]['FGA'] / game_cnt_a
        # FGM: Filed Goals Made
        x_features[16] = self.team_stats[team_h]['FGM'] / game_cnt_h
        x_features[17] = self.team_stats[team_a]['FGM'] / game_cnt_a
        # FT_PCT: Percentage of free throws that team has made
        x_features[18] = self.team_stats[team_h]['FTM'] / self.team_stats[team_h]['FTA']
        x_features[19] = self.team_stats[team_a]['FTM'] / self.team_stats[team_a]['FTA']
        # FTA: Free Throws Attempted
        x_features[20] = self.team_stats[team_h]['FTA'] / game_cnt_h
        x_features[21] = self.team_stats[team_a]['FTA'] / game_cnt_a
        # FTM: Free Throws Made
        x_features[22] = self.team_stats[team_h]['FTM'] / game_cnt_h
        x_features[23] = self.team_stats[team_a]['FTM'] / game_cnt_a
        # OREB: Offense Rebounds
        x_features[24] = self.team_stats[team_h]['ORB'] / game_cnt_h
        x_features[25] = self.team_stats[team_a]['ORB'] / game_cnt_a
        # PF: Fouls
        x_features[26] = self.team_stats[team_h]['PF'] / game_cnt_h
        x_features[27] = self.team_stats[team_a]['PF'] / game_cnt_a
        # Plus_Minus: difference in score
        x_features[28] = self.team_stats[team_h]['PM'] / game_cnt_h
        x_features[29] = self.team_stats[team_a]['PM'] / game_cnt_a
        # PTS: Number of points
        x_features[30] = self.team_stats[team_h]['PTS'] / game_cnt_h
        x_features[31] = self.team_stats[team_a]['PTS'] / game_cnt_a
        # REB: Rebounds
        x_features[32] = self.team_stats[team_h]['RB'] / game_cnt_h
        x_features[33] = self.team_stats[team_a]['RB'] / game_cnt_a
        # STL: Number of Steals
        x_features[34] = self.team_stats[team_h]['STL'] / game_cnt_h
        x_features[35] = self.team_stats[team_a]['STL'] / game_cnt_a
        # TO: Number of Turnovers
        x_features[36] = self.team_stats[team_h]['TOV'] / game_cnt_h
        x_features[37] = self.team_stats[team_a]['TOV'] / game_cnt_a

        # ADVANCED STATISTICS
        # AST: Assist
        x_features[38] = self.team_stats[team_h]['AST'] / game_cnt_h
        x_features[39] = self.team_stats[team_a]['AST'] / game_cnt_a
        # AST_TOV: Number of Assists for every turnover
        x_features[40] = self.team_stats[team_h]['AST'] / self.team_stats[team_h]['TOV']
        x_features[41] = self.team_stats[team_a]['AST'] / self.team_stats[team_a]['TOV']
        # DREB_PCT:
        x_features[42] = self.team_stats[team_h]['DRB'] / self.team_stats[team_h]['RB']
        x_features[43] = self.team_stats[team_a]['DRB'] / self.team_stats[team_a]['RB']
        # EFG_PCT:
        fg3m_h = self.team_stats[team_h]['FG3M']
        fg3m_a = self.team_stats[team_a]['FG3M']
        x_features[44] = (0.5*fg3m_h + self.team_stats[team_h]['FGM']) / self.team_stats[team_h]['FGA']
        x_features[45] = (0.5*fg3m_a + self.team_stats[team_a]['FGM']) / self.team_stats[team_a]['FGA']
        # OREB_PCT
        x_features[46] = self.team_stats[team_h]['ORB'] / self.team_stats[team_h]['RB']
        x_features[47] = self.team_stats[team_a]['ORB'] / self.team_stats[team_a]['RB']
        # DREB_PCT
        x_features[48] = self.team_stats[team_h]['DRB'] / self.team_stats[team_h]['RB']
        x_features[49] = self.team_stats[team_a]['DRB'] / self.team_stats[team_a]['RB']
        # PIE
        x_features[50] = self.__calculate_pie(team_h)
        x_features[51] = self.__calculate_pie(team_a)

        # FOUR FACTORS
        # FTA_RATE
        x_features[52] = self.team_stats[team_h]['FTA'] / self.team_stats[team_h]['FGA']
        x_features[53] = self.team_stats[team_a]['FTA'] / self.team_stats[team_a]['FGA']
        # OPP_EFG_PCT
        opp_fg3m_h = self.team_stats[team_h]['OPP_FG3M']
        opp_fg3m_a = self.team_stats[team_a]['OPP_FG3M']
        x_features[54] = (0.5 * opp_fg3m_h + self.team_stats[team_h]['OPP_FGM']) / self.team_stats[team_h]['OPP_FGA']
        x_features[55] = (0.5 * opp_fg3m_a + self.team_stats[team_a]['OPP_FGM']) / self.team_stats[team_a]['OPP_FGA']
        # OPP_FTA_RATE
        x_features[56] = self.team_stats[team_h]['OPP_FTA'] / self.team_stats[team_h]['OPP_FGA']
        x_features[57] = self.team_stats[team_a]['OPP_FTA'] / self.team_stats[team_a]['OPP_FGA']
        # OPP_OREB_PCT
        x_features[58] = self.team_stats[team_h]['OPP_ORB'] / self.team_stats[team_h]['OPP_RB']
        x_features[59] = self.team_stats[team_a]['OPP_ORB'] / self.team_stats[team_a]['OPP_RB']

        # Bookmaker's odds
        x_features[60] = odd_h
        x_features[61] = odd_a
        x_features[62] = self.team_stats[team_h]['WIN'] / game_cnt_h
        x_features[63] = self.team_stats[team_a]['WIN'] / game_cnt_a

        return x_features

    def get_data(self, opps: pd.DataFrame) -> (np.array, list):
        x_data = np.empty((opps.shape[0], self.features_n))
        skipped = []

        for i, row in enumerate(opps.iterrows()):
            match = row[1]
            team_h, team_a = match['HID'], match['AID']

            # if there is no info about team, skip
            if team_a not in self.team_stats or team_h not in self.team_stats:
                skipped.append(i)
                continue
            if self.team_stats[team_a]['GM CNT'] <= self.minimal_games or self.team_stats[team_h][
                'GM CNT'] <= self.minimal_games:
                skipped.append(i)
                continue

            x_data[i, :] = self.get_features(team_h, team_a, match['OddsH'], match['OddsA'])

        return x_data, skipped

    def train_model(self):
        x_train = np.concatenate(self.train_data[0], axis=0)
        y_train = np.concatenate(self.train_data[1], axis=0)
        # print(len(self.train_data[0]))
        train_model(self.model, x_train, y_train)
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
        if self.__is_new_season(opps, inc[0]):
            self.__prepare_train_data()
            if self.__is_enough_data():
                self.train_model()

        # skip betting on training seasons
        if not self.has_model or opps.empty:
            bets = pd.DataFrame(data=bets, columns=['BetH', 'BetA'], index=opps.index)
            return bets

        # make prediction
        x, no_info = self.get_data(opps)
        probs = classify(self.model, x)
        # confidence threshold
        confidence_threshold = np.where(abs(probs - 0.5) < + self.cnf_threshold)[0]
        no_bet = np.union1d(no_info, confidence_threshold).astype(int)

        # chose bets
        prev_bets = opps[['BetH', 'BetA']].to_numpy()
        # fractions = fractional_kelly_betting_strategy(probs, opps[['OddsH', 'OddsA']])
        # budget_per_match = (bankroll * 0.1) / n
        fractions = sharp_betting_strategy(probs)
        budget_per_match = bankroll * 0.1
        my_bets = fractions * budget_per_match
        my_bets -= prev_bets
        my_bets[my_bets < min_bet] = 0
        my_bets[my_bets > max_bet] = max_bet

        # place bets
        bets = my_bets
        bets[no_bet, :] = 0
        bets = pd.DataFrame(data=bets, columns=['BetH', 'BetA'], index=opps.index)
        return bets
