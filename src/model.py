from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from typing import Tuple

def train_model(games: pd.DataFrame, players: pd.DataFrame, prev_model=None) -> Tuple[object, StandardScaler]:
    """
    Trains a machine learning model to predict match outcomes using features consistent with `opps`.

    Parameters:
        games (pd.DataFrame): Historical games data with match and team statistics.
        players (pd.DataFrame): Historical player statistics.
        prev_model (Tuple[object, StandardScaler], optional): Previously trained model and scaler to fall back on.

    Returns:
        model (object): Trained machine learning model.
        scaler (StandardScaler): Scaler used to normalize input features.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer

    # Add target variable
    games['Outcome'] = np.where(games['H'] == 1, 0, 1)  # 0 for Home Win, 1 for Away Win

    # Check class distribution
    class_counts = games['Outcome'].value_counts()
    if class_counts.min() < 2:
        print("Insufficient data for one of the classes. Using previous model if available.")
        return prev_model if prev_model else (None, None)

    # Define overlapping features with opps
    features = ['N', 'POFF', 'OddsH', 'OddsA']

    X = games[features]
    y = games['Outcome']

    # Train-test split
    try:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError as e:
        print(f"Error during train-test split: {e}")
        return prev_model if prev_model else (None, None)

    # Preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, features)
        ]
    )

    # Model pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"Train Accuracy: {train_score:.2f}, Test Accuracy: {test_score:.2f}")

    # Extract scaler for future scaling
    scaler = model.named_steps['preprocessor'].named_transformers_['num'].named_steps['scaler']

    return model, scaler



def calculate_bets(opps: pd.DataFrame, model, scaler, historical_data: pd.DataFrame, bankroll: float, min_bet: float, max_bet: float, kelly_fraction: float = 0.1, confidence_threshold: float = 0.90):
    """
    Calculates bets for upcoming matches using features consistent with training, with better risk management and confidence filtering.

    Parameters:
        opps (pd.DataFrame): DataFrame containing upcoming games.
        model: Trained classifier model.
        scaler: Fitted scaler for preprocessing features.
        historical_data (pd.DataFrame): Stored data containing historical match stats.
        bankroll (float): The available bankroll for betting.
        min_bet (float): Minimum bet size.
        max_bet (float): Maximum bet size.
        kelly_fraction (float): Fraction of Kelly Criterion to use for bet size. E.g., 0.1 means 10% of Kelly.
        confidence_threshold (float): Minimum probability to consider a bet (e.g., 0.80 means 80% confidence).

    Returns:
        pd.DataFrame: Bets DataFrame with columns "BetH" and "BetA".
    """
    bets = pd.DataFrame(data=0, columns=["BetH", "BetA"], index=opps.index)

    # Features consistent with training
    features = ['N', 'POFF', 'OddsH', 'OddsA']
    
    # Check if all features are present in opps
    if not all(feature in opps.columns for feature in features):
        raise ValueError(f"Missing required columns in opps DataFrame: {set(features) - set(opps.columns)}")
    
    X = opps[features]

    # Check for empty DataFrame
    if X.empty:
        print("No valid data in opps DataFrame to process.")
        return bets  # Return empty bets DataFrame

    # Preprocess and scale features
    try:
        X_scaled = pd.DataFrame(
            model.named_steps['preprocessor'].transform(X), 
            columns=features,
            index=X.index
        )
    except ValueError as e:
        print(f"Error during preprocessing: {e}")
        return bets  # Return empty bets DataFrame

    # Predict probabilities
    probs = model.predict_proba(X_scaled)  # [P(Home win), P(Away win)]

    # Calculate expected value and Kelly Criterion for each bet
    for i, (prob, row) in enumerate(zip(probs, opps.itertuples())):
        # Calculate expected values for home and away bets
        ev_home = prob[0] * row.OddsH - (1 - prob[0])
        ev_away = prob[1] * row.OddsA - (1 - prob[1])

        # Kelly Criterion: f* = (bp - q) / b
        def kelly_criterion(odds, p):
            q = 1 - p
            return (odds * p - q) / odds if odds > 1 else 0  # Ensure non-negative bet fraction
        
        kelly_home = kelly_criterion(row.OddsH, prob[0])
        kelly_away = kelly_criterion(row.OddsA, prob[1])

        # Apply Kelly fraction (e.g., 0.1 for 10% of Kelly)
        kelly_home *= kelly_fraction
        kelly_away *= kelly_fraction

        # Apply confidence threshold: only place bet if confidence > threshold (e.g., 80%)
        if prob[0] < confidence_threshold and prob[1] < confidence_threshold:
            continue  # Skip betting on this match as the confidence is not high enough
        
        # Scale the bet fraction based on Kelly Criterion and bankroll
        bet_size_home = kelly_home * bankroll
        bet_size_away = kelly_away * bankroll

        # Cap bet sizes between min_bet and max_bet
        bet_size_home = min_bet # min(max(min_bet, bet_size_home), max_bet)
        bet_size_away = min_bet # min(max(min_bet, bet_size_away), max_bet)

        # Place bets if the Kelly Criterion bet size is above zero
        if kelly_home > 0 and prob[0] >= confidence_threshold:
            bets.at[row.Index, "BetH"] = bet_size_home
        
        if kelly_away > 0 and prob[1] >= confidence_threshold:
            bets.at[row.Index, "BetA"] = bet_size_away

    return bets









class Model:
    def __init__(self) -> None:
        self.all_games = pd.DataFrame()
        self.all_players = pd.DataFrame()
        self.prev_model = None
        self.prev_scaler = None

    def store_inc(self, inc: tuple[pd.DataFrame, pd.DataFrame]):
        """
        Accumulates data from the inc tuple (Games and Players DataFrames).

        Parameters:
            inc (tuple): A tuple containing Games and Players DataFrames.
        """
        games, players = inc
        self.all_games = pd.concat([self.all_games, games], ignore_index=True)
        self.all_players = pd.concat([self.all_players, players], ignore_index=True)

    def train_model(self):
        """
        Trains the model using all accumulated data.
        """
        # Call the previously defined train_model function with accumulated data
        model, scaler = train_model(self.all_games, self.all_players, self.prev_model)
        self.prev_model = model
        self.prev_scaler = scaler
    
    




    def place_bets(self, summary: pd.DataFrame, opps: pd.DataFrame, inc: tuple[pd.DataFrame, pd.DataFrame]):
        bankroll = summary.iloc[0]['Bankroll']
        min_bet = summary.iloc[0]['Min_bet']
        max_bet = summary.iloc[0]['Max_bet']
        month = summary.iloc[0]['Date'].month
        n = len(opps)
        bets = np.zeros((n, 2))
        games, players = inc

        # Update accumulated data
        self.store_inc(inc)

        # Check if there's enough data to train/retrain the model
        if len(self.all_games) > 20:  # Train after sufficient data is accumulated
            self.train_model()

        # Use the trained model and scaler to calculate bets
        if self.prev_model and self.prev_scaler:
            bets = calculate_bets(opps, self.prev_model, self.prev_scaler, self.all_games, bankroll, min_bet, max_bet)
        else:
            # If no model is trained yet, return zero bets
            bets = pd.DataFrame(data=0, columns=["BetH", "BetA"], index=opps.index)

        return bets
    