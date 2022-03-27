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

# Get 2014-2015 season data for player_stat
player_stat_14_15 = player_stat[player_stat['Season'] == '2014-15']

# There is no duplicated name for player_stat_14_15 after manual check.

# Get 2014-2015 season data for player_cv
player_cv_14_15 = player_cv[(player_cv['From'] <= 2015)
                            & (player_cv['To'] >= 2014)]

# Merge these two tables
merge_stat_cv = pd.merge(player_stat_14_15, player_cv_14_15,
                         left_on='Player', right_on='Player', how='inner')

# Get 2014-2015 season data for player_salary
player_salary_14_15 = player_salary[player_salary['SEASON'] == '2014-2015']

# preprocess the player names

# We noticed that in player_salary_14_15,
# these four players' names are not in good format:
# Jeff Taylor, Louis Williams, Luc Richard Mbah A Moute and Patty Mills.
# For example, in the table merge_stat_cv,
# Jeff Taylor is called Jeffery Taylor.

false_names = ['jeff taylor', 'louis williams',
               'luc richard mbah a moute', 'patty mills']
true_names = ['jeffery taylor', 'lou williams',
              'luc mbah a moute', 'patrick mills']


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


# Build final dataframe to work on for 2014-2015
player_salary_14_15['Player'] = player_salary_14_15['NAME'].apply(
    lambda s: preprocess_name(s))
merge_stat_cv['Player'] = merge_stat_cv['Player'].apply(
    lambda s: preprocess_name(s))

merge_14_15 = pd.merge(merge_stat_cv, player_salary_14_15,
                       left_on='Player', right_on='Player', how='inner')
# Save to csv
merge_14_15.to_csv(os.path.join(
    data_path, "preprocessed_data", "merge_14_15.csv"), index=False)
