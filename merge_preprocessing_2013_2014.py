import warnings
import pandas as pd
import os

pd.set_option('display.max_columns', None)

warnings.filterwarnings('ignore')

# Loading the datasets
data_path = "data/"
player_stat = pd.read_excel(os.path.join(
    data_path, "raw_data", "players stat.xlsx"))
player_cv = pd.read_excel(os.path.join(
    data_path, "raw_data", "players cv.xlsx"))
player_salary = pd.read_excel(os.path.join(
    data_path, "raw_data", "players salary.xlsx"))

# 2013-2014 season for player_stat
player_stat_13_14 = player_stat[player_stat['Season'] == '2013-14']

player_stat_13_14 = player_stat_13_14.reset_index(drop=True)

# There is one duplicated name (Tony Mitchell) for season 2013-2014.
# We distinguish them by adding their ages on names.
player_stat_13_14.iloc[308, 0] = 'Tony Mitchell 24'
player_stat_13_14.iloc[309, 0] = 'Tony Mitchell 21'

# Get 2013-2014 data for player_cv
player_cv_13_14 = player_cv[(player_cv['From'] <= 2014)
                            & (player_cv['To'] >= 2013)]
player_cv_13_14 = player_cv_13_14.reset_index(drop=True)

# There are 3 duplicated name:
# Chris Johnson, Tony Mitchell and Chris Wright.
# After checking Wikipedia,
# we only keep the players that are common for both tables.

player_cv_13_14.iloc[363, 0] = 'Tony Mitchell 24'
player_cv_13_14.iloc[364, 0] = 'Tony Mitchell 21'

player_cv_13_14 = player_cv_13_14.drop(274)
player_cv_13_14 = player_cv_13_14.reset_index(drop=True)

player_cv_13_14 = player_cv_13_14.drop(564)
player_cv_13_14 = player_cv_13_14.reset_index(drop=True)

# Merge these two tables

merge_stat_cv = pd.merge(player_stat_13_14, player_cv_13_14,
                         left_on='Player', right_on='Player', how='inner')

# Get 2013-2014 data for player_salary

player_salary_13_14 = player_salary[player_salary['SEASON'] == '2013-2014']
player_salary_13_14 = player_salary_13_14.reset_index(drop=True)

# We check the special player Tony Mitchell
player_salary_13_14.iloc[114, 2] = 'Tony Mitchell 21, PF'

# Preprocess the name of table player_salary

# We noticed that in player_salary_14_15,
# these five players' names are not in good format:
# Jeff Taylor, Louis Williams,
# Luc Richard Mbah A Moute, Patty Mills and erik jay murphy.
# For example, in the table merge_stat_cv,
# Jeff Taylor is called Jeffery Taylor.

false_names = ['jeff taylor', 'louis williams',
               'luc richard mbah a moute', 'patty mills', 'erik jay murphy']
true_names = ['jeffery taylor', 'lou williams',
              'luc mbah a moute', 'patrick mills', 'erik murphy']


def preprocess_name(name):
    '''
    This function will be used to preprocess the player names in
    player_salary_13_14 and merge_stat_cv. We have checked manually
    the errors that exist in the original names and have considered them
    in this function.
    '''
    if ',' in name:
        ind = name.find(',')
        name = name[:ind]
    # We remove the Jr.
    if "Jr." in name:
        name = name.replace(" Jr.", "")
    if "Jr" in name:
        name = name.replace(" Jr.", "")
    if "III" in name:
        name = name.replace(" III", "")
    while '.' in name:
        ind = name.find('.')
        name = name.replace('.', '')

    name = name.lower()
    name = name.strip()

    if name in false_names:
        name = true_names[false_names.index(name)]

    return name

# Build final dataframe to work on for season 2013-2014


player_salary_13_14['Player'] = player_salary_13_14['NAME'].apply(
    lambda s: preprocess_name(s))

merge_stat_cv['Player'] = merge_stat_cv['Player'].apply(
    lambda s: preprocess_name(s))

merge = pd.merge(merge_stat_cv, player_salary_13_14,
                 left_on='Player', right_on='Player', how='inner')

# Save to csv
merge.to_csv(os.path.join(data_path, "preprocessed_data",
                          "merge_13_14.csv"), index=False)
