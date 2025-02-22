import os
import pandas as pd
import tqdm
from pathlib import Path

from src.conf import settings, logger, COL_NAMES


class InteractionsPreprocessor:
    _tqdm_update_each = 10

    def __init__(self, data_path: Path = settings.paths.interactions_raw) -> None:
        self.data_path = data_path
        self.inter_fnames = os.listdir(data_path)
        self.n_inter_fnames = len(self.inter_fnames)

    def _generate_user_ids(self, inter_df: pd.DataFrame) -> pd.DataFrame:
        user_name_df = inter_df[COL_NAMES.user_name].unique()
        user_ids = range(1, len(user_name_df) + 1)
        user_name_id_dict = dict(zip(user_name_df, user_ids))
        user_name_id_df = pd.DataFrame(
            user_name_id_dict.items(), columns=[COL_NAMES.user_name, COL_NAMES.user_id]
        )
        inter_df[COL_NAMES.user_id] = inter_df[COL_NAMES.user_name].map(
            user_name_id_dict
        )
        inter_df = inter_df.drop(COL_NAMES.user_name, axis=1)
        return inter_df, user_name_id_df

    def preprocess_raw_interactions(
        self,
        verbose: bool = True,
    ) -> pd.DataFrame:
        inter_df_list = []
        bar = tqdm.tqdm(
            total=self.n_inter_fnames,
            desc="Preprocessing",
            position=0,
            leave=True,
        )
        for idx, fname in enumerate(self.inter_fnames):
            if verbose and idx % self._tqdm_update_each == 0:
                bar.update(self._tqdm_update_each)
            inter_init_df = pd.read_csv(
                self.data_path / fname,
                delimiter="\t",
                usecols=[
                    COL_NAMES.user_id,
                    COL_NAMES.anime_id,
                    COL_NAMES.score,
                    COL_NAMES.favorite,
                    COL_NAMES.status,
                    COL_NAMES.progress,
                ],
                dtype={
                    COL_NAMES.user_id: str,
                    COL_NAMES.anime_id: "Int32",
                    COL_NAMES.score: "Int32",
                    COL_NAMES.favorite: "Int32",
                    COL_NAMES.status: str,
                    COL_NAMES.progress: "Int32",
                },
            )
            inter_df_list.append(inter_init_df)
            del inter_init_df

        inter_full_df = pd.concat(inter_df_list)
        inter_full_df = inter_full_df.rename(
            columns={
                COL_NAMES.user_id: COL_NAMES.user_name,
                COL_NAMES.anime_id: COL_NAMES.item_id,
            }
        )
        inter_full_df[COL_NAMES.score] = inter_full_df[COL_NAMES.score].fillna(0)
        logger.info("Interactions preprocessed, generating user ids...")
        inter_full_df, user_name_id_df = self._generate_user_ids(inter_full_df)
        inter_full_df.to_parquet(
            settings.paths.interactions_score_favorite,
            engine="pyarrow",
            index=False,
        )
        user_name_id_df.to_parquet(
            settings.paths.user_name_id,
            engine="pyarrow",
            index=False,
        )
        return inter_full_df, user_name_id_df
