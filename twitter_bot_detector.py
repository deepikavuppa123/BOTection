import argparse
import logging
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Silence overly chatty libraries
warnings.filterwarnings("ignore")

try:
    from hmmlearn import hmm
except Exception as import_error:  # pragma: no cover - handled at runtime
    hmm = None  # type: ignore

try:
    import matplotlib
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - headless environments
    matplotlib = None  # type: ignore
    plt = None  # type: ignore

from sklearn.preprocessing import RobustScaler


# --------------------------------------------------------------------------------------
# Configuration and Utilities
# --------------------------------------------------------------------------------------


def configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


@dataclass(frozen=True)
class StateInfo:
    name: str
    description: str
    color: str
    characteristics: str


class TwitterBotDetector:
    def __init__(self, random_state: int = 42) -> None:
        self.model: Optional[Any] = None
        self.scaler = RobustScaler()
        self.random_state = random_state

        self.state_definitions: Dict[int, StateInfo] = {
            0: StateInfo(
                name="DORMANT",
                description="Minimal activity, lurking behavior",
                color="blue",
                characteristics="Low engagement, minimal posting, high followers/friends ratio",
            ),
            1: StateInfo(
                name="ORGANIC",
                description="Natural human-like posting patterns",
                color="green",
                characteristics="Balanced engagement, normal social ratios, diverse content",
            ),
            2: StateInfo(
                name="AGGRESSIVE",
                description="High-volume posting and engagement",
                color="orange",
                characteristics="High activity, many retweets, hashtag-heavy content",
            ),
            3: StateInfo(
                name="SUSPICIOUS",
                description="Bot-like patterns, potential automation",
                color="red",
                characteristics="Unusual ratios, repetitive behavior, extreme metrics",
            ),
        }

        self.feature_weights: Dict[str, float] = {
            "activity_ratio": 0.20,
            "social_ratio": 0.15,
            "engagement_rate": 0.15,
            "hashtag_density": 0.10,
            "retweet_ratio": 0.10,
            "account_maturity": 0.10,
            "content_diversity": 0.10,
            "anomaly_score": 0.10,
        }

        self.hmm_features: List[str] = []
        self.user_info: List[Dict[str, Any]] = []

    # ----------------------------------------------------------------------------------
    # Data Loading
    # ----------------------------------------------------------------------------------
    def load_twitter_dataset(self, file_path: str) -> pd.DataFrame:
        logging.info("Loading dataset from %s", file_path)
        try:
            if file_path.lower().endswith(".csv"):
                df = pd.read_csv(file_path)
            elif file_path.lower().endswith((".xlsx", ".xls")):
                df = pd.read_excel(file_path)
            elif file_path.lower().endswith(".json"):
                df = pd.read_json(file_path)
            else:
                df = pd.read_csv(file_path)

            logging.info("Dataset shape: %s", df.shape)
            logging.debug("Columns: %s", list(df.columns))

            required_cols = ["user_id", "followers_count", "friends_count", "statuses_count"]
            missing_cols = [c for c in required_cols if c not in df.columns]
            if missing_cols:
                logging.warning("Missing required columns: %s. Generating synthetic data.", missing_cols)
                return self.generate_synthetic_twitter_data()

            for col in ["user_created_at", "tweet_created_at", "timestamp"]:
                if col in df.columns:
                    try:
                        df[col] = pd.to_datetime(df[col])
                        logging.debug("Converted %s to datetime", col)
                    except Exception:
                        logging.debug("Could not convert %s to datetime", col)

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

            logging.info("Processed dataset shape: %s", df.shape)
            if "user_id" in df.columns:
                logging.info("Unique users: %s", df["user_id"].nunique())
            return df

        except Exception as exc:  # pragma: no cover - runtime safeguard
            logging.error("Error loading file: %s", exc)
            logging.info("Generating synthetic Twitter data as fallback...")
            return self.generate_synthetic_twitter_data()

    # ----------------------------------------------------------------------------------
    # Feature Engineering
    # ----------------------------------------------------------------------------------
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Engineering features...")

        # Avoid division by zero
        for col in ["followers_count", "friends_count", "statuses_count", "account_age_days"]:
            if col in df.columns:
                df[col] = df[col].clip(lower=1)

        # Ratios and rates
        df["follower_friend_ratio"] = np.log1p(df["followers_count"]) / np.log1p(df["friends_count"])
        df["posting_rate"] = df["statuses_count"] / df["account_age_days"]

        total_engagement = df["retweet_count"] + df["reply_count"] + df["favorite_count"]
        df["engagement_per_tweet"] = total_engagement / df["statuses_count"]

        content_actions = (
            df["retweet_count"]
            + df["reply_count"]
            + (df["statuses_count"] - df["retweet_count"] - df["reply_count"])
        )
        df["content_diversity"] = (
            df["retweet_count"] + df["reply_count"] + df["num_mentions"] + df["num_hashtags"]
        ) / (content_actions + 1)

        df["automation_score"] = (
            (df["retweet_count"] / (df["statuses_count"] + 1)) * 0.3
            + (df["num_hashtags"] / (df["statuses_count"] + 1)) * 0.3
            + (df["posting_rate"] > 50).astype(int) * 0.4
        )

        df["network_health"] = np.clip(
            np.log1p(df["followers_count"]) / (
                np.log1p(df["friends_count"]) + np.log1p(df["listed_count"] + 1)
            ),
            0,
            10,
        )

        df["maturity_score"] = np.clip(df["account_age_days"] / 365.0, 0, 5)

        df["high_posting_anomaly"] = (df["posting_rate"] > df["posting_rate"].quantile(0.95)).astype(int)
        ratio_q = df["follower_friend_ratio"].quantile([0.05, 0.95])
        df["ratio_anomaly"] = (
            (df["follower_friend_ratio"] > ratio_q.loc[0.95])
            | (df["follower_friend_ratio"] < ratio_q.loc[0.05])
        ).astype(int)

        hmm_features = [
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

        for feat in hmm_features:
            if feat not in df.columns:
                df[feat] = 0
            df[feat] = pd.to_numeric(df[feat], errors="coerce").replace([np.inf, -np.inf], np.nan)
            df[feat] = df[feat].fillna(df[feat].median())

        self.hmm_features = hmm_features
        logging.info("Prepared %d features for HMM", len(hmm_features))
        return df

    # ----------------------------------------------------------------------------------
    # Sequence Construction
    # ----------------------------------------------------------------------------------
    def create_user_sequences(self, df: pd.DataFrame, sequence_length: int = 20) -> Tuple[List[np.ndarray], List[int], List[Dict[str, Any]]]:
        logging.info("Creating user behavior sequences...")

        sequences: List[np.ndarray] = []
        sequence_lengths: List[int] = []
        user_info: List[Dict[str, Any]] = []

        feature_values = df[self.hmm_features].to_numpy()
        user_ids = df["user_id"].tolist() if "user_id" in df.columns else list(range(len(df)))

        # Iterate rows efficiently
        for row_index in range(len(df)):
            base_features = feature_values[row_index]
            seq = self._generate_user_sequence(base_features, sequence_length=sequence_length,
                                               automation_score=float(df.iloc[row_index].get("automation_score", 0.0)))

            if len(seq) >= 5:
                sequences.append(seq)
                sequence_lengths.append(len(seq))
                user_info.append(
                    {
                        "user_id": user_ids[row_index],
                        "is_inactive": bool(df.iloc[row_index].get("is_inactive", False)),
                        "followers_count": int(df.iloc[row_index].get("followers_count", 0)),
                        "posting_rate": float(df.iloc[row_index].get("posting_rate", 0.0)),
                    }
                )

        return sequences, sequence_lengths, user_info

    def _generate_user_sequence(
        self,
        base_features: Sequence[float],
        sequence_length: int = 20,
        automation_score: float = 0.0,
    ) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)
        base = np.asarray(base_features, dtype=float)

        sequence: List[np.ndarray] = []
        for i in range(sequence_length):
            time_factor = i / max(sequence_length, 1)
            variation = rng.normal(0.0, 0.1, size=base.shape[0])

            maturity_effect = np.array([0, 0, 0, 0, -0.2 * (1 - time_factor), 0.1 * time_factor, 0.2 * time_factor, 0, 0])
            maturity_effect = maturity_effect[: base.shape[0]]

            if automation_score > 0.5:
                variation *= 0.5

            evolved = base + variation + maturity_effect
            sequence.append(evolved)

        return np.vstack(sequence)

    # ----------------------------------------------------------------------------------
    # Modeling
    # ----------------------------------------------------------------------------------
    def train_hmm_model(self, df: pd.DataFrame, n_components: int = 4) -> Optional[Any]:
        if hmm is None:
            logging.error("hmmlearn is not installed. Please install it to train the model.")
            return None

        sequences, sequence_lengths, self.user_info = self.create_user_sequences(df)
        if not sequences:
            logging.error("No valid sequences found for training.")
            return None

        X = np.vstack(sequences)
        X_scaled = self.scaler.fit_transform(X)

        self.model = hmm.GaussianHMM(
            n_components=n_components,
            covariance_type="diag",
            n_iter=100,
            random_state=self.random_state,
            tol=0.01,
        )

        try:
            self.model.fit(X_scaled, lengths=sequence_lengths)
            logging.info("HMM trained. Converged: %s | Iterations: %s", self.model.monitor_.converged, self.model.monitor_.iter)
            logging.debug("Training log likelihood: %.2f", self.model.score(X_scaled, lengths=sequence_lengths))
            return self.model
        except Exception as exc:  # pragma: no cover
            logging.error("Error training HMM model: %s", exc)
            return None

    def predict_bot_states(self, df: pd.DataFrame) -> Dict[Any, Dict[str, Any]]:
        if self.model is None:
            logging.error("Model not trained yet.")
            return {}

        predictions: Dict[Any, Dict[str, Any]] = {}
        sequences, _, user_info = self.create_user_sequences(df)

        for seq, info in zip(sequences, user_info):
            try:
                scaled_sequence = self.scaler.transform(seq)
                predicted_states = self.model.predict(scaled_sequence)
                state_probabilities = self.model.predict_proba(scaled_sequence)

                bot_score = self._calculate_bot_score(predicted_states, state_probabilities, info)

                predictions[info["user_id"]] = {
                    "predicted_states": predicted_states,
                    "state_probabilities": state_probabilities,
                    "bot_score": bot_score,
                    "user_info": info,
                    "sequence_features": seq,
                }
            except Exception as exc:  # pragma: no cover
                logging.debug("Prediction failed for %s: %s", info.get("user_id"), exc)
                continue

        return predictions

    def _calculate_bot_score(self, states: np.ndarray, state_probs: np.ndarray, user_info: Dict[str, Any]) -> float:
        score = 0.0

        suspicious_state_ratio = float(np.mean(states == 3))
        aggressive_state_ratio = float(np.mean(states == 2))
        organic_state_ratio = float(np.mean(states == 1))

        score += suspicious_state_ratio * 0.4
        score += aggressive_state_ratio * 0.2
        score -= organic_state_ratio * 0.2

        transitions = np.mean(np.diff(states) != 0) if len(states) > 1 else 0.0
        if transitions < 0.1:
            score += 0.15
        elif transitions > 0.8:
            score += 0.1

        posting_rate = float(user_info.get("posting_rate", 0.0))
        followers = int(user_info.get("followers_count", 0))

        if posting_rate > 50:
            score += 0.15
        if followers < 10 or followers > 100000:
            score += 0.1

        return float(np.clip(score, 0, 1))

    def detect_bots(self, predictions: Dict[Any, Dict[str, Any]], threshold: float = 0.5) -> Dict[Any, float]:
        if not predictions:
            logging.warning("No predictions available for bot detection.")
            return {}

        user_scores: List[Tuple[Any, float]] = [(uid, float(p["bot_score"])) for uid, p in predictions.items()]
        user_scores.sort(key=lambda x: x[1], reverse=True)
        detected_bots = {uid: score for uid, score in user_scores if score > threshold}

        logging.info(
            "Detection: %d/%d users flagged (threshold=%.2f)",
            len(detected_bots),
            len(predictions),
            threshold,
        )
        return detected_bots

    # ----------------------------------------------------------------------------------
    # Visualization
    # ----------------------------------------------------------------------------------
    def visualize_analysis(self, predictions: Dict[Any, Dict[str, Any]], df: pd.DataFrame, top_users: int = 8, show: bool = True, save_dir: Optional[str] = None) -> None:
        if not predictions:
            logging.warning("No predictions available for visualization.")
            return

        if plt is None:
            logging.warning("Matplotlib not available; skipping plots.")
            return

        self._plot_bot_score_distribution(predictions, show=show, save_dir=save_dir)
        self._plot_state_distribution(predictions, show=show, save_dir=save_dir)
        self._plot_feature_analysis(df, show=show, save_dir=save_dir)
        self._plot_user_sequences(predictions, top_users=top_users, show=show, save_dir=save_dir)

    def _plot_bot_score_distribution(self, predictions: Dict[Any, Dict[str, Any]], show: bool, save_dir: Optional[str]) -> None:
        bot_scores = [float(pred["bot_score"]) for pred in predictions.values()]

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
        if save_dir:
            plt.savefig(f"{save_dir}/bot_score_distribution.png", dpi=150)
        if show:
            plt.show()
        plt.close()

    def _plot_state_distribution(self, predictions: Dict[Any, Dict[str, Any]], show: bool, save_dir: Optional[str]) -> None:
        all_states: List[int] = []
        for pred in predictions.values():
            all_states.extend(list(pred["predicted_states"]))
        if not all_states:
            return

        state_counts = [int(np.sum(np.array(all_states) == i)) for i in range(4)]
        state_names = [self.state_definitions[i].name for i in range(4)]
        colors = [self.state_definitions[i].color for i in range(4)]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(state_names, state_counts, color=colors, alpha=0.7, edgecolor="black")

        for bar, count in zip(bars, state_counts):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(state_counts) * 0.01,
                f"{count}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        plt.title("Overall State Distribution Across All Users", fontsize=14, fontweight="bold")
        plt.xlabel("User State")
        plt.ylabel("Number of Observations")
        plt.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        if save_dir:
            plt.savefig(f"{save_dir}/state_distribution.png", dpi=150)
        if show:
            plt.show()
        plt.close()

    def _plot_feature_analysis(self, df: pd.DataFrame, show: bool, save_dir: Optional[str]) -> None:
        features_to_plot = ["posting_rate", "follower_friend_ratio", "automation_score", "engagement_per_tweet"]
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()

        for i, feature in enumerate(features_to_plot):
            if feature in df.columns:
                q1, q3 = df[feature].quantile([0.25, 0.75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                filtered = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)][feature]

                axes[i].hist(filtered, bins=30, alpha=0.7, color="lightblue", edgecolor="black")
                axes[i].set_xlabel(feature.replace("_", " ").title())
                axes[i].set_ylabel("Frequency")
                axes[i].set_title(f"Distribution of {feature.replace('_', ' ').title()}")
                axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_dir:
            plt.savefig(f"{save_dir}/feature_distributions.png", dpi=150)
        if show:
            plt.show()
        plt.close()

    def _plot_user_sequences(self, predictions: Dict[Any, Dict[str, Any]], top_users: int, show: bool, save_dir: Optional[str]) -> None:
        user_scores: List[Tuple[Any, float]] = [(uid, float(pred["bot_score"])) for uid, pred in predictions.items()]
        user_scores.sort(key=lambda x: x[1], reverse=True)

        half = max(top_users // 2, 1)
        top_bots = user_scores[:half]
        bottom_normal = user_scores[-half:]

        fig, axes = plt.subplots(half, 2, figsize=(15, 4 * half))
        if half == 1:
            axes = np.array([axes])  # shape (1, 2)

        for i, ((bot_uid, bot_score), (normal_uid, normal_score)) in enumerate(zip(top_bots, bottom_normal)):
            bot_states = predictions[bot_uid]["predicted_states"]
            axes[i][0].plot(bot_states, "o-", color="red", linewidth=2, markersize=4)
            axes[i][0].set_title(f"Suspected Bot: {bot_uid}\nBot Score: {bot_score:.3f}")
            axes[i][0].set_ylabel("State")
            axes[i][0].set_yticks(range(4))
            axes[i][0].set_yticklabels([self.state_definitions[j].name for j in range(4)])
            axes[i][0].grid(True, alpha=0.3)

            normal_states = predictions[normal_uid]["predicted_states"]
            axes[i][1].plot(normal_states, "o-", color="green", linewidth=2, markersize=4)
            axes[i][1].set_title(f"Likely Human: {normal_uid}\nBot Score: {normal_score:.3f}")
            axes[i][1].set_ylabel("State")
            axes[i][1].set_yticks(range(4))
            axes[i][1].set_yticklabels([self.state_definitions[j].name for j in range(4)])
            axes[i][1].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_dir:
            plt.savefig(f"{save_dir}/user_sequences.png", dpi=150)
        if show:
            plt.show()
        plt.close()

    # ----------------------------------------------------------------------------------
    # Reporting and Synthetic Data
    # ----------------------------------------------------------------------------------
    def generate_detailed_report(self, predictions: Dict[Any, Dict[str, Any]], df: pd.DataFrame, threshold: float = 0.5) -> Dict[Any, float]:
        print("=" * 100)
        print("TWITTER BOT DETECTION REPORT - 4-STATE HMM ANALYSIS")
        print("=" * 100)

        detected_bots = self.detect_bots(predictions, threshold)

        print(f"\n\ud83d\udcca DATASET OVERVIEW:")
        print(f"   Total users analyzed: {len(predictions):,}")
        print(f"   Features engineered: {len(self.hmm_features)}")
        print(f"   HMM states: 4 (Dormant, Organic, Aggressive, Suspicious)")

        print(f"\n\ud83e\udd16 BOT DETECTION RESULTS:")
        print(f"   Potential bots detected: {len(detected_bots):,}")
        print(f"   Detection rate: {len(detected_bots)/max(len(predictions),1)*100:.1f}%")
        print(f"   Detection threshold: {threshold}")

        all_states: List[int] = []
        for pred in predictions.values():
            all_states.extend(list(pred["predicted_states"]))

        print(f"\n\ud83d\udcc8 STATE DISTRIBUTION:")
        total_obs = max(len(all_states), 1)
        for state in range(4):
            count = int(np.sum(np.array(all_states) == state))
            percentage = (count / total_obs) * 100
            info = self.state_definitions[state]
            print(f"   {info.name}: {count:,} observations ({percentage:.1f}%)")
            print(f"      \u2192 {info.description}")

        if detected_bots:
            print(f"\n\ud83d\udea8 TOP SUSPICIOUS ACCOUNTS:")
            top = sorted(detected_bots.items(), key=lambda x: x[1], reverse=True)[:5]
            for rank, (user_id, score) in enumerate(top, 1):
                pred = predictions[user_id]
                user_inf = pred["user_info"]
                states = pred["predicted_states"]
                primary_state = np.argmax(np.bincount(states)) if len(states) else 0
                state_dist = [int(np.sum(states == i)) for i in range(4)]
                print(f"\n   #{rank}. User: {user_id}")
                print(f"      Bot Score: {score:.3f}")
                print(f"      Followers: {user_inf.get('followers_count', 'N/A'):,}")
                print(f"      Primary State: {self.state_definitions[int(primary_state)].name}")
                print(
                    f"      State Distribution: {dict(zip(['Dormant','Organic','Aggressive','Suspicious'], state_dist))}"
                )

        print(f"\n\ud83d\udccb KEY FEATURE INSIGHTS:")
        if "posting_rate" in df.columns:
            high_posters = int((df["posting_rate"] > 50).sum())
            print(f"   Users with high posting rate (>50/day): {high_posters:,}")
        if "automation_score" in df.columns:
            high_automation = int((df["automation_score"] > 0.7).sum())
            print(f"   Users with high automation score (>0.7): {high_automation:,}")

        if self.model is not None:
            print(f"\n\ud83d\udd0d MODEL PERFORMANCE:")
            print(f"   HMM converged: {self.model.monitor_.converged}")
            print(f"   Training iterations: {self.model.monitor_.iter}")
            print(f"   Features used: {', '.join(self.hmm_features)}")

        return detected_bots

    def generate_synthetic_twitter_data(self, n_users: int = 1000) -> pd.DataFrame:
        logging.info("Generating synthetic Twitter data (%d users)...", n_users)
        rng = np.random.default_rng(self.random_state)

        users: List[Dict[str, Any]] = []
        for i in range(n_users):
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
                    "favourites_count": int(rng.integers(0, max(statuses_count, 1))),
                    "listed_count": int(rng.integers(0, max(followers_count // 10, 1))),
                    "activity_ratio": activity_ratio,
                    "social_ratio": social_ratio,
                    "is_inactive": bool(rng.random() < 0.1),
                    "retweet_count": retweet_count,
                    "reply_count": int(rng.integers(0, max(statuses_count // 2, 1))),
                    "favorite_count": int(rng.integers(0, max(statuses_count, 1))),
                    "num_hashtags": num_hashtags,
                    "num_urls": int(rng.integers(0, max(statuses_count // 4, 1))),
                    "num_mentions": int(rng.integers(0, max(statuses_count // 3, 1))),
                    "true_label": "bot" if is_bot else "human",
                }
            )

        return pd.DataFrame(users)


# --------------------------------------------------------------------------------------
# High-level API
# --------------------------------------------------------------------------------------


def main_twitter_analysis(
    file_path: Optional[str] = None,
    n_components: int = 4,
    visualize: bool = True,
    top_users: int = 6,
    random_state: int = 42,
    plots_dir: Optional[str] = None,
) -> Tuple[TwitterBotDetector, pd.DataFrame, Dict[Any, Dict[str, Any]]]:
    print("\ud83d\udc26 Twitter Bot Detection with 4-State HMM")
    print("=" * 50)

    detector = TwitterBotDetector(random_state=random_state)

    if file_path:
        print(f"Loading dataset from: {file_path}")
        df = detector.load_twitter_dataset(file_path)
    else:
        print("No file path provided, using synthetic data for demonstration")
        df = detector.generate_synthetic_twitter_data(n_users=500)

    df = detector.engineer_features(df)

    print("\n\ud83d\udd27 Training HMM model...")
    model = detector.train_hmm_model(df, n_components=n_components)
    if model is None:
        print("\u274c Failed to train model. Exiting.")
        return detector, df, {}

    print("\n\ud83d\udd0d Analyzing user behavior and detecting bots...")
    predictions = detector.predict_bot_states(df)
    if not predictions:
        print("\u274c Failed to generate predictions. Exiting.")
        return detector, df, {}

    if visualize:
        print("\n\ud83d\udcca Generating visualizations...")
        detector.visualize_analysis(predictions, df, top_users=top_users, show=True, save_dir=plots_dir)

    print("\n\ud83d\udccb Generating detection report...")
    detector.generate_detailed_report(predictions, df, threshold=0.5)

    # Additional insights
    print("\n" + "=" * 100)
    print("\ud83d\udd2c ADDITIONAL INSIGHTS")
    print("=" * 100)

    feature_cols = [c for c in df.columns if c in detector.hmm_features]
    if len(feature_cols) > 1:
        corr = df[feature_cols].corr()
        print("\n\ud83d\udcc8 Feature Correlation Analysis (|r|>0.3):")
        pairs: List[Tuple[str, str, float]] = []
        for i in range(len(feature_cols)):
            for j in range(i + 1, len(feature_cols)):
                r = float(corr.iloc[i, j])
                if abs(r) > 0.3:
                    pairs.append((feature_cols[i], feature_cols[j], r))
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        for a, b, r in pairs[:5]:
            print(f"   {a} \u2194 {b}: {r:.3f}")

    print(f"\n\ud83d\udd04 State Transition Patterns:")
    transition_matrix = np.zeros((4, 4))
    for pred in predictions.values():
        states = pred["predicted_states"]
        for i in range(len(states) - 1):
            transition_matrix[states[i], states[i + 1]] += 1
    row_sums = transition_matrix.sum(axis=1)
    with np.errstate(invalid="ignore"):
        transition_probs = transition_matrix / np.where(row_sums[:, None] == 0, 1, row_sums[:, None])

    state_names = ["Dormant", "Organic", "Aggressive", "Suspicious"]
    print("   Transition probabilities (from → to):")
    for i in range(4):
        for j in range(4):
            if transition_probs[i, j] > 0.1:
                print(f"   {state_names[i]} → {state_names[j]}: {transition_probs[i, j]:.3f}")

    print("\n\u2705 Analysis complete!\n")
    return detector, df, predictions


def analyze_custom_twitter_data(file_path: str, detection_threshold: float = 0.5) -> Tuple[Optional[TwitterBotDetector], Optional[Dict[Any, float]]]:
    detector, df, predictions = main_twitter_analysis(file_path)
    if predictions:
        detected_bots = detector.detect_bots(predictions, detection_threshold)
        print(f"\n\ud83c\udfaf QUICK SUMMARY:")
        print(f"   Dataset: {len(df):,} users")
        print(f"   Potential bots: {len(detected_bots):,} ({len(detected_bots)/len(predictions)*100:.1f}%)")
        print(f"   Threshold: {detection_threshold}")
        return detector, detected_bots
    return None, None


def demo_with_synthetic_data() -> Tuple[TwitterBotDetector, pd.DataFrame, Dict[Any, Dict[str, Any]]]:
    return main_twitter_analysis()


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Twitter Bot Detection using a 4-state HMM")
    parser.add_argument("--file", dest="file_path", type=str, default=None, help="Path to dataset (.csv/.json/.xlsx)")
    parser.add_argument("--threshold", dest="threshold", type=float, default=0.5, help="Bot detection threshold [0-1]")
    parser.add_argument("--no-plots", dest="no_plots", action="store_true", help="Disable plotting (headless mode)")
    parser.add_argument("--states", dest="n_components", type=int, default=4, help="Number of HMM states")
    parser.add_argument("--top-users", dest="top_users", type=int, default=6, help="Top users to visualize")
    parser.add_argument("--plots-dir", dest="plots_dir", type=str, default=None, help="Directory to save plots")
    parser.add_argument("-v", "--verbose", dest="verbosity", action="count", default=0, help="Increase logging verbosity (-v, -vv)")
    parser.add_argument("--seed", dest="seed", type=int, default=42, help="Random seed")
    return parser


def main() -> None:  # pragma: no cover - CLI entry
    args = build_arg_parser().parse_args()
    configure_logging(args.verbosity)

    if matplotlib is not None and args.no_plots:
        try:
            matplotlib.use("Agg", force=True)
        except Exception:
            pass

    if args.file_path:
        detector, df, predictions = main_twitter_analysis(
            file_path=args.file_path,
            n_components=args.n_components,
            visualize=not args.no_plots,
            top_users=args.top_users,
            random_state=args.seed,
            plots_dir=args.plots_dir,
        )
        if predictions:
            detected_bots = detector.detect_bots(predictions, args.threshold)
            print(f"\nPotential bots detected: {len(detected_bots)} (threshold={args.threshold})")
    else:
        print("Running Twitter Bot Detection Demo with synthetic data...")
        detector, df, predictions = demo_with_synthetic_data()
        if predictions:
            print("\n\ud83c\udf89 Demo completed successfully!")
            print(f"   Trained model on {len(df):,} users")
            print("   Analyzed behavioral patterns using 4-state HMM")
            print("   Check the visualizations above for detailed results!")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()

