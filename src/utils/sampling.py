from src.utils import db_api


def sample_users_by_activity(n: int, min_rev: int, max_rev: int) -> list[str]:
    sql = f"""
      SELECT user_id FROM (
        SELECT user_id, COUNT(*) AS cnt
        FROM reviews
        GROUP BY user_id
      ) sub
      WHERE cnt BETWEEN {min_rev} AND {max_rev}
      ORDER BY RANDOM()
      LIMIT {n};
    """
    return db_api.conn.execute(sql).fetchdf()["user_id"].tolist()


def sample_users_stratified(n_total: int, strata: dict) -> list[str]:
    """
    Samples users from different activity strata.

    Args:
        n_total (int): The total number of users to sample.
        strata (dict): A dictionary where keys are strata names and values are
                       tuples of (min_reviews, max_reviews, proportion).
                       Proportions should sum to 1.

    Returns:
        list[str]: A list of sampled user IDs.
    """
    all_user_ids = []
    for stratum, (min_rev, max_rev, proportion) in strata.items():
        n_sample = int(n_total * proportion)
        if n_sample == 0:
            continue

        user_ids = sample_users_by_activity(n_sample, min_rev, max_rev)
        all_user_ids.extend(user_ids)

    return all_user_ids
