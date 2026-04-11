import sqlite3
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, roc_auc_score,
    ConfusionMatrixDisplay, RocCurveDisplay,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
import matplotlib.pyplot as plt

WINDOW_SIZES = [5, 10, 20]

def load_logs(db_path: str, limit: int | None = 2000, null: bool = False) -> pd.DataFrame:
    """Load memory_logs from SQLite into a DataFrame."""
    conn = sqlite3.connect(db_path)
    query = f"SELECT * FROM memory_logs {f'WHERE attack_scenario IS NOT NULL' if null else ''} ORDER BY sequence_num {f' LIMIT {limit}' if limit else ''}"
    df = pd.read_sql_query(query, conn)
    conn.close()
 
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values(["agent_id", "sequence_num"]).reset_index(drop=True)
    return df

class MLClassifier:
    OPERATION_COLUMN_NAME = "operation"
    SYS_MEM_OPS  = {"core_memory_append", "core_memory_write_persona", "core_memory_write_human", "core_memory_read_human", "core_memory_read_persona", "core_memory_replace" }
    WRITE_OPS    = {"recall_insert", "recall_insert_many", "archival_insert", "archival_search", "archival_delete"}
    READ_OPS     = {"recall_text_search", "recall_date_search"}


    def __init__(self, window_sizes: list[int] = [5, 10, 20]):
        self.window_sizes = window_sizes
        self.data_df = None
        self.feature_df = None
        self.pipe = None
        self.eval_dict = None

    def _op_flags(self, op: str) -> dict:
        op_lower = op.lower()
        return {
            "is_write": int(op_lower in self.WRITE_OPS),
            "is_read": int(op_lower in self.READ_OPS),
            "is_sys_mem": int(op_lower in self.SYS_MEM_OPS)
        }
    
    def _reset(self):
        self.data_df = None
        self.feature_df = None
        self.pipe = None
        self.eval_dict = None

    def _weak_label(self, df: pd.DataFrame, init_skip: int = 6) -> pd.Series:
        """
        Generate weak (heuristic) labels for anomaly detection.
        Returns a binary series (0 = normal, 1 = suspicious).
        """
        df_agent_grouped = df.groupby("agent_id")

        init_mask = df_agent_grouped.cumcount() >= init_skip

        # Rule 1: High context pressure (near full context window)
        high_ctx = df["context_window_pct"] > 0.9

        # Rule 2: System memory operations (often sensitive)
        sys_mem = df["is_sys_mem"] == 1

        # Rule 3: Sudden content changes
        content_delta = (df["content_len"] - df["prev_val_len"]).abs()
        threshold = df_agent_grouped["content_changed"].transform(
            lambda x: x.quantile(0.95)
        )
        large_content_change = content_delta > threshold

        # Rule 4: Large token jumps (requires feature already computed)
        large_token_jump = (
            df_agent_grouped["token_offset"]
            .diff().abs().fillna(0) > 1000
        )

        # burst of sys memory writes
        sys_mem_burst = (
            df_agent_grouped["is_sys_mem"]
            .transform(lambda x: x.rolling(5, min_periods=1).sum()) >= 3
        )

        # Combine rules into a score
        score = (
            high_ctx.astype(int) +
            (sys_mem & ~sys_mem_burst).astype(int) +
            sys_mem_burst.astype(int) * 2 + 
            large_content_change.astype(int) * 2 +
            large_token_jump.astype(int) * 2
        )

        return ((score >= 3) & init_mask).astype(int)

    def _aggregate_features(self, group: pd.DataFrame):
        group = group.copy()
        for window_size in self.window_sizes:
            rolling = group.rolling(window_size, min_periods=1)

            col_pfx = f"w{window_size}_"
            # Operation rate
            group[f"{col_pfx}op_rate"] = rolling["op_encoded"].mean()
 
            # Write / read ratio
            write_cnt = rolling["is_write"].sum()
            read_cnt  = rolling["is_read"].sum()
            group[f"{col_pfx}write_read_ratio"] = write_cnt / (read_cnt + 1e-6)
 
            # Sys-mem access rate
            group[f"{col_pfx}sys_mem_rate"] = rolling["is_sys_mem"].mean()
 
            # Token offset drift (mean absolute change)
            group[f"{col_pfx}token_drift"]  = (
                group["token_offset"]
                .diff()
                .abs()
                .rolling(window_size, min_periods=1)
                .mean()
            )
 
            # Context-window pressure
            group[f"{col_pfx}ctx_pct_mean"] = rolling["context_window_pct"].mean()
            group[f"{col_pfx}ctx_pct_std"]  = rolling["context_window_pct"].std()

             # Op-type entropy over window (diversity of operation types)
            def op_entropy(series):
                vc = series.value_counts(normalize=True)
                return -(vc * np.log2(vc + 1e-9)).sum()\
                
            group[f"{col_pfx}op_entropy"] = (
                group["op_encoded"]
                .rolling(window_size, min_periods=1)
                .apply(op_entropy, raw=False)
            )
 
            # Content burst (spike in content length)
            group[f"{col_pfx}content_burst"] = (
                rolling["content_len"].std().fillna(0)
            )
        group["seq_gap"] = group["sequence_num"].diff().fillna(0)
        return group

    def _build_feature_df(self, feature_rows: list[pd.DataFrame]):
        target_feature_df = pd.concat(feature_rows, ignore_index=True)
 
        scalar_cols = [
            "agent_id", "op_encoded", "is_write", "is_read", "is_sys_mem",
            "token_offset", "context_window_pct", "context_window",
            "content_len", "prev_val_len", "content_changed", "seq_gap",
        ]
    
        rolling_cols = [
            c for c in target_feature_df.columns
            if any(c.startswith(f"w{w}_") for w in self.window_sizes)
        ]
    
        feature_cols = scalar_cols + rolling_cols
        return target_feature_df[feature_cols + ["ground_truth_label", "attack_scenario"]]
    
    def _build_features(self, target_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        op_encoder = LabelEncoder()
        target_df["op_encoded"] = op_encoder.fit_transform(target_df["operation"].fillna("unknown"))

        flag_df = target_df[self.OPERATION_COLUMN_NAME].apply(lambda o: pd.Series(self._op_flags(o)))
        target_df = pd.concat([target_df, flag_df], axis=1)

        target_df["content_len"] = target_df["content"].fillna("").str.len()
        target_df["prev_val_len"] = target_df["previous_value"].fillna("").str.len()
        target_df["content_changed"] = (target_df["content_len"] != target_df["prev_val_len"]).astype(int)
        # target_df["ground_truth_label"] = self._weak_label(target_df)

        feature_rows = []
        for _, group in target_df.groupby("agent_id", sort=False):
            group = group.sort_values("sequence_num").reset_index(drop=True)
            feature_rows.append(self._aggregate_features(group))

        return target_df, self._build_feature_df(feature_rows)
    
    def build_features(self, df: pd.DataFrame):
        self._reset()
        self.data_df, self.feature_df = self._build_features(df)
        return self
    
    def build_classifier(
        self,
        label_col: str = "ground_truth_label",
        test_size: float = 0.2,
        random_state: int = 42,
        n_estimators: int = 200,
        class_weight: str = "balanced",   # important for imbalanced attack logs)
    ):
        """
        Train a Random Forest on feature_df.
        Returns (fitted_pipeline, eval_dict).
        """

        if self.feature_df is None:
            raise Exception("Feature df not initialized. Call build_features")
        
        groups = self.feature_df["agent_id"]
        X = self.feature_df.drop(columns={"ground_truth_label", "attack_scenario", "agent_id"}, errors="ignore")
        y = self.feature_df[label_col].fillna(0).astype(int)

        gss = GroupShuffleSplit(test_size=test_size, random_state=random_state)

        train_idx, test_idx = next(gss.split(X, y, groups))

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
        self.pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),  # handles NaN from rolling
            ("clf", RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=None,
                min_samples_leaf=2,
                class_weight=class_weight,
                random_state=random_state,
                n_jobs=-1,
            )),
        ])
        print("Train labels:", np.unique(y_train, return_counts=True))
        print("Test labels:", np.unique(y_test, return_counts=True))
        self.pipe.fit(X_train, y_train)

        y_pred  = self.pipe.predict(X_test)
        y_proba = self.pipe.predict_proba(X_test)[:, 1]
    
        n_classes = len(np.unique(y))
        if n_classes == 2:
            auc = roc_auc_score(y_test, y_proba)
        else:
            auc = roc_auc_score(y_test, self.pipe.predict_proba(X_test), multi_class="ovr")
    
        report = classification_report(y_test, y_pred, output_dict=True)

        num_agents = 3  # change to match actual num of agents
        cv_scores = cross_val_score(self.pipe, X, y, cv=StratifiedGroupKFold(n_splits=num_agents).split(X, y, groups), scoring="f1_weighted", n_jobs=-1)
    
        self.eval_dict = {
            "report": report,
            "auc": auc,
            "cv_f1_mean": cv_scores.mean(),
            "cv_f1_std":  cv_scores.std(),
            "X_test": X_test,
            "y_test": y_test,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "feature_names": list(X.columns),
        }
    
        print("\n── Classification Report ──")
        print(classification_report(y_test, y_pred))
        print(f"ROC-AUC : {auc:.4f}")
        print(f"CV F1   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        return self
    
    def plot_evaluation(self, top_n: int = 20):
        if not self.eval_dict and not self.pipe:
            raise Exception("eval_dict and pipe not set for plot_evaluation")
        
        clf = self.pipe.named_steps["clf"]
        feature_names = self.eval_dict["feature_names"]
        importances   = clf.feature_importances_
    
        top_idx  = np.argsort(importances)[-top_n:]
        top_names = [feature_names[i] for i in top_idx]
        top_vals  = importances[top_idx]
    
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("MemGPT Anomaly Classifier — Evaluation", fontsize=13)
    
        ConfusionMatrixDisplay.from_predictions(
            self.eval_dict["y_test"], self.eval_dict["y_pred"], ax=axes[0], colorbar=False
        )
        axes[0].set_title("Confusion matrix")
    
        # ROC curve (binary only)
        if len(np.unique(self.eval_dict["y_test"])) == 2:
            RocCurveDisplay.from_predictions(
                self.eval_dict["y_test"], self.eval_dict["y_proba"], ax=axes[1]
            )
            axes[1].set_title(f"ROC curve  (AUC={self.eval_dict['auc']:.3f})")
        else:
            axes[1].text(0.5, 0.5, "Multi-class ROC\nnot plotted here",
                        ha="center", va="center", transform=axes[1].transAxes)
            axes[1].set_title("ROC curve")
    
        # Feature importances
        axes[2].barh(top_names, top_vals, color="#5DCAA5")
        axes[2].set_xlabel("Mean decrease in impurity")
        axes[2].set_title(f"Top {top_n} features")
        axes[2].tick_params(labelsize=8)
    
        plt.tight_layout()
        plt.savefig("classifier_eval.png", dpi=150, bbox_inches="tight")
        plt.show()
        print("Saved in classifier_eval.png")
        return self
    
    def predict_new_session(self, pipe: Pipeline, raw_rows: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        """
        Score a batch of raw log rows.
        raw_rows must have the same columns as memory_logs (minus ground_truth_label).
        Returns the input rows with added columns:
            attack_prob  — model's P(attack)
            predicted    — 0/1 at given threshold
        """
        _, feature_df = self._build_features(raw_rows.copy())
        X = feature_df.drop(columns=["ground_truth_label", "attack_scenario", "agent_id"], errors="ignore")
        proba = pipe.predict_proba(X)[:, 1]
        raw_rows = raw_rows.copy()
        raw_rows["attack_prob"] = proba
        raw_rows["predicted"]   = (proba >= threshold).astype(int)
        return raw_rows

 
if __name__ == "__main__": 
    db_path = sys.argv[1] if len(sys.argv) > 1 else "memgpt.db"
 
    print(f"Loading logs from {db_path} …")
    df = load_logs(db_path)
    print(f"  Rows: {len(df):,}  |  Agents: {df['agent_id'].nunique()}")

    classifier = (MLClassifier(WINDOW_SIZES)
        .build_features(df)
        .build_classifier()
        .plot_evaluation(50)
    )

    print(f"  Feature matrix: {classifier.feature_df.shape}")
    print(f"  Label distribution:\n{classifier.feature_df['ground_truth_label'].value_counts()}")
 
    scored = classifier.predict_new_session(classifier.pipe, df.head(50))
    print("\nSample scored rows:")
    print(scored[["id", "agent_id", "operation", "attack_prob", "predicted"]].head(5).to_string())
