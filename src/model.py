from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

class Model:
    def __init__(self):
        self.model = None
        self.cumulative_games = pd.DataFrame()
        self.cumulative_players = pd.DataFrame()
        self.current_season = None

    def train_model(self):
        if len(self.cumulative_games) < 500:  # Ensure enough data for training
            return None

        # Prepare features and labels for training
        features = []
        labels = []
        for _, row in self.cumulative_games.iterrows():
            home_stats = self.cumulative_players[self.cumulative_players['Team'] == row['HID']].mean(numeric_only=True)
            away_stats = self.cumulative_players[self.cumulative_players['Team'] == row['AID']].mean(numeric_only=True)
            
            feature_vector = [
                row['OddsH'], row['OddsA'],
                home_stats['PTS'], away_stats['PTS'],
                home_stats['AST'], away_stats['AST'],
                home_stats['RB'], away_stats['RB']
            ]
            features.append(feature_vector)
            labels.append(row['H'])  # Home win indicator

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

        # Create a pipeline with scaling and gradient boosting
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', GradientBoostingClassifier(n_estimators=100, random_state=42))
        ])

        # Train the model
        self.model = pipeline.fit(X_train, y_train)


    def place_bets(self, summary: pd.DataFrame, opps: pd.DataFrame, inc: tuple[pd.DataFrame, pd.DataFrame]):
        bankroll = summary.iloc[0]['Bankroll']
        min_bet = summary.iloc[0]['Min_bet']
        max_bet = summary.iloc[0]['Max_bet']
        month = summary.iloc[0]['Date'].month
        n = len(opps)
        opp_start_index = opps.index
        bets = np.zeros((n, 2))
        games, players = inc

        if opps.empty:
            bets = pd.DataFrame(data=bets, columns=["BetH", "BetA"], index=opp_start_index)
            return bets

        current_season = opps.iloc[0]['Season']
        if self.current_season != current_season:
            self.cumulative_games = pd.DataFrame()
            self.cumulative_players = pd.DataFrame()
            self.current_season = current_season

        self.cumulative_games = pd.concat([self.cumulative_games, games], ignore_index=True)
        self.cumulative_players = pd.concat([self.cumulative_players, players], ignore_index=True)

        if len(self.cumulative_games) < 500:
            bets = pd.DataFrame(data=bets, columns=["BetH", "BetA"], index=opp_start_index)
            return bets

        if len(self.cumulative_games) % 250 <= 10:
            self.train_model()

        confidence_threshold = 0.8
        ev_threshold = 0.1

        bets = np.zeros((n, 2))
        prp = -1
        for i, row in opps.iterrows():
            prp += 1
            # Prepare features for prediction
            home_stats = self.cumulative_players[self.cumulative_players['Team'] == row['HID']].mean(numeric_only=True)
            away_stats = self.cumulative_players[self.cumulative_players['Team'] == row['AID']].mean(numeric_only=True)
            feature_vector = [
                row['OddsH'], row['OddsA'],
                home_stats['PTS'], away_stats['PTS'],
                home_stats['AST'], away_stats['AST'],
                home_stats['RB'], away_stats['RB']
            ]
            feature_vector = np.array(feature_vector).reshape(1, -1)

            # Predict win probability for the home team
            home_prob = self.model.predict_proba(feature_vector)[0, 1]
            away_prob = 1 - home_prob

            # Skip if confidence is below threshold
            if max(home_prob, away_prob) < confidence_threshold:
                continue

            # Calculate expected value
            ev_home = home_prob * row['OddsH'] - (1 - home_prob)
            ev_away = away_prob * row['OddsA'] - (1 - away_prob)

            # Place bets based on EV
            if ev_home > ev_threshold:
                bet_home = min(max_bet, max(min_bet, bankroll * 0.01 * home_prob))
                bets[prp, 0] = bet_home
                bankroll -= bet_home

            if ev_away > ev_threshold:
                bet_away = min(max_bet, max(min_bet, bankroll * 0.01 * away_prob))
                bets[prp, 1] = bet_away
                bankroll -= bet_away

        bets = pd.DataFrame(data=bets, columns=["BetH", "BetA"], index=opp_start_index)
        # print(bets)
        return bets
