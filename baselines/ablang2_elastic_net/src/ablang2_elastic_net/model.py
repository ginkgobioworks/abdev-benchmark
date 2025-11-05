# baselines/ablang2_elastic_net/src/ablang2_elastic_net/model.py

from __future__ import annotations
from pathlib import Path
import pickle
import re
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import ablang2

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score
from abdev_core import BaseModel, PROPERTY_LIST


# Keep 20 canonical AAs; drop gaps and stops
AA = set("ACDEFGHIKLMNPQRSTVWY")
def _clean_seq(s: str) -> str:
    s = str(s).upper().strip()
    s = re.sub(r"[^A-Z]", "", s)            # remove non-letters
    s = s.replace("*", "").replace("-", "") # remove stop/gap
    return "".join(ch for ch in s if ch in AA)


class Ablang2ElasticNetModel(BaseModel):
    """ElasticNet regression on AbLang2 paired (VH|VL) mean-pooled embeddings.

    - Requires: 'vh_protein_sequence', 'vl_protein_sequence' in both train/predict
    - Embedding: AbLang2-paired with input 'VH|VL' 
    - Pooling: mean over last hidden states
    - Head: one ElasticNet per property in PROPERTY_LIST 
    """

    ALPHA = 0.1
    L1_RATIO = 0.5
    MAX_ITER = 2000
    RANDOM_STATE = 42

    def __init__(self) -> None:
        # Use cuda if available; load AbLang2 onto the same device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.m = None  # lazy-loaded AbLang2 model

    # ---------- embedding ----------
    def _init_ablang2(self) -> None:
        if self.m is not None:
            return
        self.m = ablang2.pretrained(
            model_to_use="ablang2-paired",
            random_init=False,
            ncpu=1,
            device=str(self.device),  
        )

    def _embed_pairs(self, vh_list: List[str], vl_list: List[str]) -> np.ndarray:
        """Return array [N, H] of mean-pooled hidden states over VH|VL tokens."""
        self._init_ablang2()

        # Clean sequences; avoid empty strings (fallback to "A" to keep row counts aligned)
        paired = []
        for vh, vl in zip(vh_list, vl_list):
            vh_c, vl_c = _clean_seq(vh), _clean_seq(vl)
            if not vh_c: vh_c = "A"
            if not vl_c: vl_c = "A"
            paired.append(f"{vh_c}|{vl_c}")

        tok = self.m.tokenizer(paired, pad=True, w_extra_tkns=False, device=str(self.device))
        with torch.no_grad():
            reps = self.m.AbRep(tok).last_hidden_states  # (B, L, H)
            pooled = reps.mean(dim=1).cpu().numpy()      # (B, H)
        return pooled

    # ---------- training ----------
    def train(self, df: pd.DataFrame, run_dir: Path, *, seed: int = 42) -> None:
        """Fit one ElasticNet per property in PROPERTY_LIST."""
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        required = ["vh_protein_sequence", "vl_protein_sequence"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        vh = df["vh_protein_sequence"].astype(str).tolist()
        vl = df["vl_protein_sequence"].astype(str).tolist()

        embeddings = self._embed_pairs(vh, vl)

        models: Dict[str, ElasticNet] = {}
        rng_state = self.RANDOM_STATE if seed is None else seed

        for prop in PROPERTY_LIST:
            if prop not in df.columns:
                continue

            # ---- target cleanup / guards ----
            y = pd.to_numeric(df[prop], errors="coerce")
            mask = y.notna().values
            if mask.sum() < 2:
                # Not enough labels to train a regressor; skip
                continue

            yp = y.values[mask].astype(float)
            # Skip near-constant targets (prevents meaningless CV & collapsed preds)
            if np.nanstd(yp) < 1e-8:
                print(f"[skip] {prop}: target variance ~0, cannot train.")
                continue
                

            Xp = embeddings[mask].astype(np.float64)

            # Safer PCA size for small-N
            pca_k = min( min(48, Xp.shape[1]), max(2, Xp.shape[0] - 2) )

            cv = RepeatedKFold(
                n_splits=min(5, max(2, Xp.shape[0] // 5)),
                n_repeats=3,
                random_state=rng_state,
            )

            # IMPORTANT: no 0.0 here to avoid the warning
            enet = ElasticNetCV(
                l1_ratio=[0.05, 0.1, 0.2, 0.3],
                alphas=np.logspace(-5, 1, 20),
                cv=cv,
                max_iter=50000,
                tol=1e-4,
                random_state=rng_state,
            )

            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=pca_k, svd_solver="auto", whiten=False)),
                ("enet", enet),
            ])

            pipe.fit(Xp, yp)

            # Fallback to true ridge if ENet is effectively zeroed or performs terribly
            use_ridge = False
            enet_fitted = pipe.named_steps["enet"]
            coef = getattr(enet_fitted, "coef_", None)
            if coef is None or np.allclose(coef, 0.0):
                use_ridge = True
            else:
                # sanity check in-sample R2; if <= 0, try ridge
                try:
                    if len(yp) > 1 and r2_score(yp, pipe.predict(Xp)) <= 0:
                        use_ridge = True
                except Exception:
                    pass

            if use_ridge:
                ridge = RidgeCV(alphas=np.logspace(-5, 3, 30), store_cv_values=False)
                pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("pca", PCA(n_components=pca_k, svd_solver="auto", whiten=False)),
                    ("ridge", ridge),
                ])
                pipe.fit(Xp, yp)

            models[prop] = pipe

            # --- keep your small diagnostics ---
            try:
                yhat = pipe.predict(Xp)
                uniq = len(np.unique(np.round(yhat, 6)))
                std  = float(np.std(yhat, ddof=1)) if len(yhat) > 1 else 0.0
                r2   = r2_score(yp, yhat) if len(yp) > 1 else float("nan")
                head = "ridge" if "ridge" in pipe.named_steps else "enet"
                chosen_alpha = (pipe.named_steps["ridge"].alpha_
                                if head == "ridge" else getattr(enet_fitted, "alpha_", None))
                chosen_l1 = (None if head == "ridge" else getattr(enet_fitted, "l1_ratio_", None))
                n_iter    = (None if head == "ridge" else getattr(enet_fitted, "n_iter_", None))
                print(f"[fit] {prop}: head={head} alpha={chosen_alpha} l1={chosen_l1} n_iter={n_iter} | "
                    f"uniq={uniq:4d} std={std:.6g} R2={r2:.4f} n={len(yp)}")
            except Exception as _diag_err:
                print(f"[fit] {prop}: diagnostics skipped ({_diag_err})")


        # Save artifacts like other accepted baselines
        with open(run_dir / "models.pkl", "wb") as f:
            pickle.dump(models, f)
        np.save(run_dir / "embeddings.npy", embeddings)

    # ---------- prediction ----------
    def predict(self, df: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
        """Predict all available trained properties; always return required columns."""
        run_dir = Path(run_dir)
        models_path = run_dir / "models.pkl"
        if not models_path.exists():
            raise FileNotFoundError(f"Models not found: {models_path}")

        with open(models_path, "rb") as f:
            models: Dict[str, ElasticNet] = pickle.load(f)

        required = ["vh_protein_sequence", "vl_protein_sequence"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        vh = df["vh_protein_sequence"].astype(str).tolist()
        vl = df["vl_protein_sequence"].astype(str).tolist()

        embeddings = self._embed_pairs(vh, vl)

        # Ensure the 3 required columns are present even if antibody_name is absent in input
        df_out = pd.DataFrame({
            "antibody_name": (
                df["antibody_name"].astype(str).values
                if "antibody_name" in df.columns else
                np.array([f"ab_{i}" for i in range(len(df))], dtype=str)
            ),
            "vh_protein_sequence": df["vh_protein_sequence"].astype(str).values,
            "vl_protein_sequence": df["vl_protein_sequence"].astype(str).values,
        })

        for prop, model in models.items():
            df_out[prop] = model.predict(embeddings)

        return df_out