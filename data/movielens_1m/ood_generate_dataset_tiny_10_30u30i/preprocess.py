from tqdm import tqdm
import random
import pandas as pd

random.seed(61)

window_size = 10
num_neg_train = 4
num_neg_test = 99
buffer_train = []
buffer_test = []

df_ratings = pd.read_csv('./ml-1m/ratings.dat',
                         sep='::',
                         names=['user_id', 'item_id', 'rating', 'timestamp'],
                         engine='python')
df_ratings[["user_id", "item_id", "rating", "timestamp"
            ]] = df_ratings[["user_id", "item_id", "rating",
                             "timestamp"]].astype(int)
df_ratings = df_ratings.sort_values(by=["user_id", "timestamp"])
df_movies = pd.read_csv('./ml-1m/movies.dat',
                        sep="::",
                        names=['item_id', 'name', 'genre'],
                        encoding='ISO-8859-1',
                        engine='python')
all_movies = set(df_movies['item_id'])

for user_id, group in tqdm(df_ratings.groupby("user_id")):
    user_clicks = list(group["item_id"])
    if len(user_clicks) < 3:
        continue
    trigger_seq = [0] * window_size
    click_seq = [user_clicks[0]] + [0] * (window_size - 1)
    all_neg_samples = all_movies - set(user_clicks)
    for i in range(len(user_clicks) - 1):
        target_id = user_clicks[i + 1]
        trigger_seq_str = ",".join(map(str, trigger_seq))
        click_seq_str = ",".join(map(str, click_seq))
        if i < len(user_clicks) - 2:
            buffer_train.append(
                f"{user_id};{target_id};{trigger_seq_str};{click_seq_str};{1}\n"
            )
            neg_samples_train = random.sample(all_neg_samples, num_neg_train)
            for neg_sample in neg_samples_train:
                buffer_train.append(
                    f"{user_id};{neg_sample};{trigger_seq_str};{click_seq_str};{0}\n"
                )
        else:
            buffer_test.append(
                f"{user_id};{target_id};{trigger_seq_str};{click_seq_str};{1}\n"
            )
            neg_samples_test = random.sample(all_neg_samples, num_neg_test)
            for neg_sample in neg_samples_test:
                buffer_test.append(
                    f"{user_id};{neg_sample};{trigger_seq_str};{click_seq_str};{0}\n"
                )
        trigger_seq = click_seq[:]
        if i < window_size - 1:
            click_seq = trigger_seq[:i + 1] + [target_id
                                               ] + [0] * (window_size - i - 2)
        else:
            click_seq = trigger_seq[1:] + [target_id]
    with open("train.txt", "a") as f:
        f.writelines(buffer_train)
    with open("test.txt", "a") as f:
        f.writelines(buffer_test)
    buffer_train.clear()  # 清空缓存
    buffer_test.clear()
