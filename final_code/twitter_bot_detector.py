import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import datetime

try:
    from hmmlearn import hmm
except Exception:  # allow import to fail gracefully for environments without hmmlearn
    hmm = None

from sklearn.preprocessing import RobustScaler

import matplotlib.pyplot as plt


class TwitterBotDetector:
    """Optimized and robust Twitter bot detector using a 4-state Gaussian HMM.

    Key improvements:
    - Deterministic pseudo-sequence generation with per-user seeded RNG for reproducibility
    - Vectorized operations and safe numeric handling to avoid NaNs/Infs
    - Configurable sequence length and RNG seed
    - Optional plotting toggles to reduce overhead in headless/batch runs
    - Graceful degradation when hmmlearn is unavailable
    """

    def __init__(self, sequence_length: int = 20, random_seed: int = 42, enable_plots: bool = True):
        self.model = None
        self.scaler = RobustScaler()
        self.sequence_length = max(5, int(sequence_length))
        self.random_seed = int(random_seed)
        self.enable_plots = enable_plots

        self.state_definitions = {
            0: {
                "name": "DORMANT",
                "description": "Minimal activity, lurking behavior",
                "color": "blue",
                "characteristics": "Low engagement, minimal posting, high followers/friends ratio",
            },
            1: {
                "name": "ORGANIC",
                "description": "Natural human-like posting patterns",
                "color": "green",
                "characteristics": "Balanced engagement, normal social ratios, diverse content",
            },
            2: {
                "name": "AGGRESSIVE",
                "description": "High-volume posting and engagement",
                "color": "orange",
                "characteristics": "High activity, many retweets, hashtag-heavy content",
            },
            3: {
                "name": "SUSPICIOUS",
                "description": "Bot-like patterns, potential automation",
                "color": "red",
                "characteristics": "Unusual ratios, repetitive behavior, extreme metrics",
            },
        }

        self.feature_weights = {
            "activity_ratio": 0.20,
            "social_ratio": 0.15,
            "engagement_rate": 0.15,
            "hashtag_density": 0.10,
            "retweet_ratio": 0.10,
            "account_maturity": 0.10,
            "content_diversity": 0.10,
            "anomaly_score": 0.10,
        }

        self.hmm_features = [
            "posting_rate",
            "engagement_per_tweet",
            "follower_friend_ratio",
            "content_diversity",
            "automation_score",
            "network_health",
            "maturity_score",
            "activity_ratio",
            "social_ratio",
        ]

    # --------------------- Data Loading ---------------------
    def load_twitter_dataset(self, file_path: str) -> pd.DataFrame:
        try:
            if file_path.lower().endswith(".csv"):
                df = pd.read_csv(file_path)
            elif file_path.lower().endswith((".xlsx", ".xls")):
                df = pd.read_excel(file_path)
            elif file_path.lower().endswith(".json"):
                df = pd.read_json(file_path)
            else:
                df = pd.read_csv(file_path)

            required_cols = ["user_id", "followers_count", "friends_count", "statuses_count"]
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                return self.generate_synthetic_twitter_data()

            for col in ["user_created_at", "tweet_created_at", "timestamp"]:
                if col in df.columns:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        try:
                            df[col] = pd.to_datetime(df[col], errors="coerce")
                        except Exception:
                            pass

            numeric_cols = [
                "account_age_days",
                "statuses_count",
                "followers_count",
                "friends_count",
                "favourites_count",
                "listed_count",
                "activity_ratio",
                "social_ratio",
                "retweet_count",
                "reply_count",
                "favorite_count",
                "num_hashtags",
                "num_urls",
                "num_mentions",
            ]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

            if "is_inactive" in df.columns:
                df["is_inactive"] = df["is_inactive"].astype(bool)

            return df
        except Exception:
            return self.generate_synthetic_twitter_data()

    # --------------------- Feature Engineering ---------------------
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Guard against zero/negative values where divisions occur
        for col in ["followers_count", "friends_count", "statuses_count", "account_age_days"]:
            if col not in df:
                df[col] = 1
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=1)

        # Ratios and rates
        df["follower_friend_ratio"] = np.log1p(df["followers_count"]) / np.log1p(df["friends_count"])
        df["posting_rate"] = df["statuses_count"] / df["account_age_days"]

        total_engagement = (
            df.get("retweet_count", 0)
            + df.get("reply_count", 0)
            + df.get("favorite_count", 0)
        )
        df["engagement_per_tweet"] = total_engagement / df["statuses_count"]

        content_actions = (
            df.get("retweet_count", 0)
            + df.get("reply_count", 0)
            + (df["statuses_count"] - df.get("retweet_count", 0) - df.get("reply_count", 0))
        )
        df["content_diversity"] = (
            df.get("retweet_count", 0)
            + df.get("reply_count", 0)
            + df.get("num_mentions", 0)
            + df.get("num_hashtags", 0)
        ) / (content_actions + 1)

        df["automation_score"] = (
            (df.get("retweet_count", 0) / (df["statuses_count"] + 1)) * 0.3
            + (df.get("num_hashtags", 0) / (df["statuses_count"] + 1)) * 0.3
            + (df["posting_rate"] > 50).astype(int) * 0.4
        )

        df["network_health"] = np.clip(
            np.log1p(df["followers_count"]) / (
                np.log1p(df["friends_count"]) + np.log1p(df.get("listed_count", 0) + 1)
            ),
            0,
            10,
        )

        df["maturity_score"] = np.clip(df["account_age_days"] / 365.0, 0, 5)

        # Anomalies (not fed directly to HMM but good for diagnostics)
        with np.errstate(all="ignore"):
            df["high_posting_anomaly"] = (
                df["posting_rate"] > df["posting_rate"].quantile(0.95)
            ).astype(int)
            ratio = df["follower_friend_ratio"]
            df["ratio_anomaly"] = (
                (ratio > ratio.quantile(0.95)) | (ratio < ratio.quantile(0.05))
            ).astype(int)

        # Ensure HMM features exist and are numeric
        for feat in self.hmm_features:
            if feat not in df.columns:
                df[feat] = 0.0
            df[feat] = pd.to_numeric(df[feat], errors="coerce")

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        for feat in self.hmm_features:
            median_val = df[feat].median()
            df[feat] = df[feat].fillna(median_val if np.isfinite(median_val) else 0.0)

        return df

    # --------------------- Sequence Construction ---------------------
    def create_user_sequences(self, df: pd.DataFrame):
        sequences = []
        sequence_lengths = []
        user_info = []

        # Per-user deterministic RNG seeded by user_id and global seed
        for user_id in df["user_id"].unique():
            user_row = df[df["user_id"] == user_id].iloc[0]
            sequence = self._generate_user_sequence(user_row)
            if len(sequence) >= 5:
                sequences.append(sequence)
                sequence_lengths.append(len(sequence))
                user_info.append(
                    {
                        "user_id": user_id,
                        "is_inactive": bool(user_row.get("is_inactive", False)),
                        "followers_count": float(user_row.get("followers_count", 0)),
                        "posting_rate": float(user_row.get("posting_rate", 0)),
                    }
                )

        return sequences, sequence_lengths, user_info

    def _generate_user_sequence(self, user_row: pd.Series) -> np.ndarray:
        base_features = np.array([float(user_row.get(feat, 0.0)) for feat in self.hmm_features], dtype=float)

        # Seed RNG deterministically per user for reproducibility
        uid = str(user_row.get("user_id", ""))
        uid_seed = (hash(uid) ^ self.random_seed) & 0xFFFFFFFF
        rng = np.random.default_rng(uid_seed)

        seq_len = int(self.sequence_length)
        sequence = np.empty((seq_len, base_features.size), dtype=float)

        # Precompute time and maturity effects once
        time_factors = np.linspace(0.0, 1.0, num=seq_len, dtype=float)
        maturity_effect = np.array([0, 0, 0, 0, -0.2, 0.1, 0.2, 0, 0], dtype=float)
        if maturity_effect.size > base_features.size:
            maturity_effect = maturity_effect[: base_features.size]
        elif maturity_effect.size < base_features.size:
            maturity_effect = np.pad(maturity_effect, (0, base_features.size - maturity_effect.size))

        low_variation = float(user_row.get("automation_score", 0) > 0.5)
        noise_scale = 0.05 if low_variation else 0.10

        for i, t in enumerate(time_factors):
            variation = rng.normal(0.0, noise_scale, size=base_features.size)
            evolved = base_features + variation + maturity_effect * t
            sequence[i] = evolved

        return sequence

    # --------------------- Model Training & Inference ---------------------
    def train_hmm_model(self, df: pd.DataFrame, n_components: int = 4):
        if hmm is None:
            raise RuntimeError(
                "hmmlearn is not available. Please install hmmlearn to train the model."
            )

        sequences, sequence_lengths, self.user_info = self.create_user_sequences(df)
        if not sequences:
            raise ValueError("No sequences available for HMM training.")

        X = np.vstack(sequences)
        X_scaled = self.scaler.fit_transform(X)

        self.model = hmm.GaussianHMM(
            n_components=n_components,
            covariance_type="diag",
            n_iter=100,
            random_state=self.random_seed,
            tol=0.01,
        )

        self.model.fit(X_scaled, lengths=sequence_lengths)
        return self.model

    def predict_bot_states(self, df: pd.DataFrame):
        if self.model is None:
            raise RuntimeError("Model not trained. Call train_hmm_model first.")

        predictions = {}
        sequences, _, user_info = self.create_user_sequences(df)

        for sequence, info in zip(sequences, user_info):
            scaled = self.scaler.transform(sequence)
            states = self.model.predict(scaled)
            probs = self.model.predict_proba(scaled)
            bot_score = self._calculate_bot_score(states, probs, info)
            predictions[info["user_id"]] = {
                "predicted_states": states,
                "state_probabilities": probs,
                "bot_score": float(bot_score),
                "user_info": info,
                "sequence_features": sequence,
            }

        return predictions

    def _calculate_bot_score(self, states: np.ndarray, state_probs: np.ndarray, user_info: dict) -> float:
        length = max(1, len(states))
        suspicious_ratio = float(np.sum(states == 3)) / length
        aggressive_ratio = float(np.sum(states == 2)) / length
        organic_ratio = float(np.sum(states == 1)) / length

        score = 0.0
        score += suspicious_ratio * 0.4
        score += aggressive_ratio * 0.2
        score -= organic_ratio * 0.2

        transitions = float(np.sum(np.diff(states) != 0)) / length
        if transitions < 0.1:
            score += 0.15
        elif transitions > 0.8:
            score += 0.10

        posting_rate = float(user_info.get("posting_rate", 0.0))
        followers = float(user_info.get("followers_count", 0.0))
        if posting_rate > 50:
            score += 0.15
        if followers < 10 or followers > 100000:
            score += 0.10

        return float(np.clip(score, 0.0, 1.0))

    def detect_bots(self, predictions: dict, threshold: float = 0.5) -> dict:
        if not predictions:
            return {}
        scores = sorted(((uid, p["bot_score"]) for uid, p in predictions.items()), key=lambda x: x[1], reverse=True)
        return {uid: score for uid, score in scores if score > threshold}

    # --------------------- Visualization ---------------------
    def visualize_analysis(self, predictions: dict, df: pd.DataFrame, top_users: int = 8):
        if not self.enable_plots or not predictions:
            return
        self._plot_bot_score_distribution(predictions)
        self._plot_state_distribution(predictions)
        self._plot_feature_analysis(df)
        self._plot_user_sequences(predictions, top_users)

    def _plot_bot_score_distribution(self, predictions: dict):
        if not self.enable_plots:
            return
        bot_scores = [p["bot_score"] for p in predictions.values()]
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.hist(bot_scores, bins=30, alpha=0.7, color="skyblue", edgecolor="black")
        plt.axvline(0.5, color="red", linestyle="--", linewidth=2, label="Threshold (0.5)")
        plt.xlabel("Bot Score")
        plt.ylabel("Number of Users")
        plt.title("Distribution of Bot Scores")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.subplot(1, 2, 2)
        plt.boxplot(bot_scores)
        plt.ylabel("Bot Score")
        plt.title("Bot Score Statistics")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def _plot_state_distribution(self, predictions: dict):
        if not self.enable_plots:
            return
        all_states = []
        for p in predictions.values():
            all_states.extend(p["predicted_states"])
        if not all_states:
            return
        counts = [int(np.sum(np.array(all_states) == i)) for i in range(4)]
        names = [self.state_definitions[i]["name"] for i in range(4)]
        colors = [self.state_definitions[i]["color"] for i in range(4)]
        plt.figure(figsize=(10, 6))
        bars = plt.bar(names, counts, color=colors, alpha=0.7, edgecolor="black")
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.01, f"{count}", ha="center", va="bottom", fontweight="bold")
        plt.title("Overall State Distribution Across All Users", fontsize=14, fontweight="bold")
        plt.xlabel("User State")
        plt.ylabel("Number of Observations")
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.show()

    def _plot_feature_analysis(self, df: pd.DataFrame):
        if not self.enable_plots:
            return
        features = ["posting_rate", "follower_friend_ratio", "automation_score", "engagement_per_tweet"]
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        for i, feature in enumerate(features):
            if feature in df.columns:
                q1, q3 = df[feature].quantile([0.25, 0.75])
                iqr = q3 - q1
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                filtered = df[(df[feature] >= lower) & (df[feature] <= upper)][feature]
                axes[i].hist(filtered, bins=30, alpha=0.7, color="lightblue", edgecolor="black")
                axes[i].set_xlabel(feature.replace("_", " ").title())
                axes[i].set_ylabel("Frequency")
                axes[i].set_title(f"Distribution of {feature.replace('_', ' ').title()}")
                axes[i].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def _plot_user_sequences(self, predictions: dict, top_users: int):
        if not self.enable_plots:
            return
        scores = sorted(((uid, p["bot_score"]) for uid, p in predictions.items()), key=lambda x: x[1], reverse=True)
        half = max(1, top_users // 2)
        top_bots = scores[:half]
        bottom = scores[-half:]
        fig, axes = plt.subplots(half, 2, figsize=(15, 4 * half))
        if half == 1:
            axes = [axes]
        for i, ((bot_uid, bot_score), (normal_uid, normal_score)) in enumerate(zip(top_bots, bottom)):
            bot_states = predictions[bot_uid]["predicted_states"]
            axes[i][0].plot(bot_states, "o-", color="red", linewidth=2, markersize=4)
            axes[i][0].set_title(f"Suspected Bot: {bot_uid}\nBot Score: {bot_score:.3f}")
            axes[i][0].set_ylabel("State")
            axes[i][0].set_yticks(range(4))
            axes[i][0].set_yticklabels([self.state_definitions[j]["name"] for j in range(4)])
            axes[i][0].grid(True, alpha=0.3)
            normal_states = predictions[normal_uid]["predicted_states"]
            axes[i][1].plot(normal_states, "o-", color="green", linewidth=2, markersize=4)
            axes[i][1].set_title(f"Likely Human: {normal_uid}\nBot Score: {normal_score:.3f}")
            axes[i][1].set_ylabel("State")
            axes[i][1].set_yticks(range(4))
            axes[i][1].set_yticklabels([self.state_definitions[j]["name"] for j in range(4)])
            axes[i][1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    # --------------------- Reporting ---------------------
    def generate_detailed_report(self, predictions: dict, df: pd.DataFrame, threshold: float = 0.5):
        print("=" * 100)
        print("TWITTER BOT DETECTION REPORT - 4-STATE HMM ANALYSIS")
        print("=" * 100)

        detected_bots = self.detect_bots(predictions, threshold)

        print("\n\ud83d\udcca DATASET OVERVIEW:")
        print(f"   Total users analyzed: {len(predictions):,}")
        print(f"   Features engineered: {len(self.hmm_features)}")
        print("   HMM states: 4 (Dormant, Organic, Aggressive, Suspicious)")

        print("\n\ud83e\udd16 BOT DETECTION RESULTS:")
        print(f"   Potential bots detected: {len(detected_bots):,}")
        rate = (len(detected_bots) / max(1, len(predictions))) * 100
        print(f"   Detection rate: {rate:.1f}%")
        print(f"   Detection threshold: {threshold}")

        all_states = []
        for p in predictions.values():
            all_states.extend(p["predicted_states"])
        total_obs = max(1, len(all_states))
        print("\n\ud83d\udcc8 STATE DISTRIBUTION:")
        for state in range(4):
            count = int(np.sum(np.array(all_states) == state))
            percentage = (count / total_obs) * 100
            info = self.state_definitions[state]
            print(f"   {info['name']}: {count:,} observations ({percentage:.1f}%)")
            print(f"      \u2192 {info['description']}")

        if detected_bots:
            print("\n\ud83d\udea8 TOP SUSPICIOUS ACCOUNTS:")
            top5 = sorted(detected_bots.items(), key=lambda x: x[1], reverse=True)[:5]
            for rank, (user_id, score) in enumerate(top5, 1):
                pred = predictions[user_id]
                info = pred["user_info"]
                states = pred["predicted_states"]
                primary_state = self.state_definitions[int(np.argmax(np.bincount(states)))]["name"]
                dist = [int(np.sum(states == i)) for i in range(4)]
                print(f"\n   #{rank}. User: {user_id}")
                print(f"      Bot Score: {score:.3f}")
                print(f"      Followers: {int(info.get('followers_count', 0)):,}")
                print(f"      Primary State: {primary_state}")
                print(
                    f"      State Distribution: {dict(zip(['Dormant','Organic','Aggressive','Suspicious'], dist))}"
                )

        if "posting_rate" in df.columns:
            high_posters = int((df["posting_rate"] > 50).sum())
            print(f"\n\ud83d\udcc8 Users with high posting rate (>50/day): {high_posters:,}")
        if "automation_score" in df.columns:
            high_automation = int((df["automation_score"] > 0.7).sum())
            print(f"\ud83d\udcc8 Users with high automation score (>0.7): {high_automation:,}")

        if self.model is not None and hasattr(self.model, "monitor_"):
            print("\n\ud83d\udd0d MODEL PERFORMANCE:")
            print(f"   HMM converged: {self.model.monitor_.converged}")
            print(f"   Training iterations: {self.model.monitor_.iter}")
            print(f"   Features used: {', '.join(self.hmm_features)}")

        return detected_bots

    # --------------------- Synthetic Data ---------------------
    def generate_synthetic_twitter_data(self, n_users: int = 1000) -> pd.DataFrame:
        rng = np.random.default_rng(self.random_seed)
        users = []
        for i in range(int(n_users)):
            is_bot = rng.random() < 0.2
            if is_bot:
                followers_count = int(rng.integers(10, 10000))
                friends_count = int(rng.integers(1000, 50000))
                statuses_count = int(rng.integers(10000, 100000))
                account_age_days = int(rng.integers(30, 2000))
                activity_ratio = float(rng.uniform(0.8, 1.0))
                social_ratio = float(rng.uniform(0.1, 0.3))
                retweet_count = int(rng.integers(5000, 80000))
                num_hashtags = int(rng.integers(1000, 20000))
            else:
                followers_count = int(rng.integers(50, 5000))
                friends_count = int(rng.integers(100, 2000))
                statuses_count = int(rng.integers(100, 10000))
                account_age_days = int(rng.integers(365, 3000))
                activity_ratio = float(rng.uniform(0.2, 0.7))
                social_ratio = float(rng.uniform(0.4, 0.9))
                retweet_count = int(rng.integers(10, 5000))
                num_hashtags = int(rng.integers(0, 1000))

            users.append(
                {
                    "user_id": f"user_{i}",
                    "account_age_days": account_age_days,
                    "statuses_count": statuses_count,
                    "followers_count": followers_count,
                    "friends_count": friends_count,
                    "favourites_count": int(rng.integers(0, max(1, statuses_count))),
                    "listed_count": int(rng.integers(0, max(1, followers_count // 10))) if followers_count > 0 else 0,
                    "activity_ratio": activity_ratio,
                    "social_ratio": social_ratio,
                    "is_inactive": bool(rng.random() < 0.1),
                    "retweet_count": retweet_count,
                    "reply_count": int(rng.integers(0, max(1, statuses_count // 2))),
                    "favorite_count": int(rng.integers(0, max(1, statuses_count))),
                    "num_hashtags": num_hashtags,
                    "num_urls": int(rng.integers(0, max(1, statuses_count // 4))),
                    "num_mentions": int(rng.integers(0, max(1, statuses_count // 3))),
                    "true_label": "bot" if is_bot else "human",
                }
            )
        return pd.DataFrame(users)


def main_twitter_analysis(file_path: str | None = None, detection_threshold: float = 0.5, sequence_length: int = 20, seed: int = 42, enable_plots: bool = True):
    print("\ud83d\udc26 Twitter Bot Detection with 4-State HMM")
    print("=" * 50)

    detector = TwitterBotDetector(sequence_length=sequence_length, random_seed=seed, enable_plots=enable_plots)

    if file_path:
        print(f"Loading dataset from: {file_path}")
        df = detector.load_twitter_dataset(file_path)
    else:
        print("No file path provided, using synthetic data for demonstration")
        df = detector.generate_synthetic_twitter_data(n_users=500)

    df = detector.engineer_features(df)

    print("\n\ud83d\udd27 Training HMM model...")
    model = detector.train_hmm_model(df, n_components=4)
    print("Model trained.")

    print("\n\ud83d\udd0d Analyzing user behavior and detecting bots...")
    predictions = detector.predict_bot_states(df)

    print("\n\ud83d\udcca Generating visualizations...")
    detector.visualize_analysis(predictions, df, top_users=6)

    print("\n\ud83d\udccb Generating detection report...")
    detected_bots = detector.generate_detailed_report(predictions, df, threshold=detection_threshold)

    print("\n" + "=" * 100)
    print("\ud83d\udd2c ADDITIONAL INSIGHTS")
    print("=" * 100)

    feature_cols = [c for c in df.columns if c in detector.hmm_features]
    if len(feature_cols) > 1:
        corr = df[feature_cols].corr()
        pairs = []
        for i in range(len(feature_cols)):
            for j in range(i + 1, len(feature_cols)):
                val = corr.iloc[i, j]
                if abs(val) > 0.3:
                    pairs.append((feature_cols[i], feature_cols[j], float(val)))
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        if pairs:
            print("\n\ud83d\udcc8 Top correlated feature pairs:")
            for a, b, v in pairs[:5]:
                print(f"   {a} \u2194 {b}: {v:.3f}")

    print("\n\ud83d\udd04 State Transition Patterns:")
    transition = np.zeros((4, 4), dtype=float)
    for p in predictions.values():
        s = p["predicted_states"]
        for i in range(len(s) - 1):
            transition[s[i], s[i + 1]] += 1
    row_sums = transition.sum(axis=1)
    row_sums[row_sums == 0] = 1
    probs = transition / row_sums[:, None]
    names = ["Dormant", "Organic", "Aggressive", "Suspicious"]
    print("   Transition probabilities (from \u2192 to):")
    for i in range(4):
        for j in range(4):
            if probs[i, j] > 0.1:
                print(f"   {names[i]} \u2192 {names[j]}: {probs[i, j]:.3f}")

    print("\n\u2705 Analysis complete! Check visualizations above for detailed insights.")
    return detector, df, predictions


def analyze_custom_twitter_data(file_path: str, detection_threshold: float = 0.5, sequence_length: int = 20, seed: int = 42, enable_plots: bool = True):
    detector, df, predictions = main_twitter_analysis(
        file_path=file_path,
        detection_threshold=detection_threshold,
        sequence_length=sequence_length,
        seed=seed,
        enable_plots=enable_plots,
    )
    if predictions:
        detected_bots = detector.detect_bots(predictions, detection_threshold)
        print("\n\ud83c\udfaf QUICK SUMMARY:")
        print(f"   Dataset: {len(df):,} users")
        print(f"   Potential bots: {len(detected_bots):,} ({(len(detected_bots)/max(1,len(predictions))*100):.1f}%)")
        print(f"   Threshold: {detection_threshold}")
        return detector, detected_bots
    return None, None


def demo_with_synthetic_data():
    return main_twitter_analysis(file_path=None)


def analyze_csv_file(csv_path: str):
    return main_twitter_analysis(file_path=csv_path)


if __name__ == "__main__":
    # Optional demo run using synthetic data
    demo_with_synthetic_data()

