
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)

df = pd.read_csv('raw_total_fight_data.csv', sep = ';') # ; For fixing formatting

df.head()


df.shape

to_drop = ['R_REV','B_REV','R_HEAD', 'B_HEAD', 'R_BODY', 'B_BODY', 'R_LEG', 'B_LEG', 'R_DISTANCE', 'B_DISTANCE', 'R_CLINCH', 'B_CLINCH']
df = df.drop(to_drop, axis=1)

to_drop2 = ['Referee','location']
df = df.drop(to_drop2, axis=1)

df['Format'] = df['Format'].str.extract(r'(\d+)')
df = df.rename(columns={'Format': 'No_of_rounds'})
df = df.rename(columns={'R_KD': 'R_Knockdown'})
df = df.rename(columns={'B_KD': 'B_Knockdown'})
df = df.rename(columns={'R_SIG_STR.': 'R_Significant_Strikes'})
df = df.rename(columns={'B_SIG_STR.': 'B_Significant_Strikes'})
df = df.rename(columns={'R_SIG_STR_pct': 'R_Significant_Strike_Percent'})
df = df.rename(columns={'B_SIG_STR_pct': 'B_Significant_Strike_Percent'})
df = df.rename(columns={'R_TOTAL_STR.': 'R_Total_Strikes'})
df = df.rename(columns={'B_TOTAL_STR.': 'B_Total_Strikes'})
df = df.rename(columns={'R_TD': 'R_Takedowns'})
df = df.rename(columns={'B_TD': 'B_Takedowns'})
df = df.rename(columns={'R_TD_pct': 'R_Takedown_Percent'})
df = df.rename(columns={'B_TD_pct': 'B_Takedown_Percent'})
df = df.rename(columns={'R_SUB_ATT': 'R_Submission_Attempt'})
df = df.rename(columns={'B_SUB_ATT': 'B_Submission_Attempt'})
df = df.rename(columns={'R_CTRL': 'R_Ground_Control'})
df = df.rename(columns={'B_CTRL': 'B_Ground_Control'})
df = df.rename(columns={'R_GROUND': 'R_Ground_Strikes'})
df = df.rename(columns={'B_GROUND': 'B_Ground_Strikes'})
df.head()
df[['R_Significant_Strikes_Landed', 'R_Significant_Strikes_Attempted']] = df['R_Significant_Strikes'].str.split(' of ', expand=True)
df[['B_Significant_Strikes_Landed', 'B_Significant_Strikes_Attempted']] = df['B_Significant_Strikes'].str.split(' of ', expand=True)

to_drop3 = ['R_Significant_Strikes','B_Significant_Strikes']
df = df.drop(to_drop3, axis=1)

df[['R_Total_Strikes_Landed', 'R_Total_Strikes_Attempted']] = df['R_Total_Strikes'].str.split(' of ', expand=True)
df[['B_Total_Strikes_Landed', 'B_Total_Strikes_Attempted']] = df['B_Total_Strikes'].str.split(' of ', expand=True)

to_drop3 = ['R_Total_Strikes','B_Total_Strikes']
df = df.drop(to_drop3, axis=1)

df[['R_Takedowns_Landed', 'R_Takedowns_Attempted']] = df['R_Takedowns'].str.split(' of ', expand=True)
df[['B_Takedowns_Landed', 'B_Takedowns_Attempted']] = df['B_Takedowns'].str.split(' of ', expand=True)

to_drop3 = ['R_Takedowns','B_Takedowns']
df = df.drop(to_drop3, axis=1)

df[['R_Ground_Strikes_Landed', 'R_Ground_Strikes_Attempted']] = df['R_Ground_Strikes'].str.split(' of ', expand=True)
df[['B_Ground_Strikes_Landed', 'B_Ground_Strikes_Attempted']] = df['B_Ground_Strikes'].str.split(' of ', expand=True)

to_drop3 = ['R_Ground_Strikes','B_Ground_Strikes']
df = df.drop(to_drop3, axis=1)

df['R_Significant_Strike_Percent'] = df['R_Significant_Strike_Percent'].str.replace('%', '')
df['B_Significant_Strike_Percent'] = df['B_Significant_Strike_Percent'].str.replace('%', '')
df['R_Takedown_Percent'] = df['R_Takedown_Percent'].str.replace('%', '')
df['B_Takedown_Percent'] = df['B_Takedown_Percent'].str.replace('%', '')

df['R_Significant_Strike_Percent'] = df['R_Significant_Strike_Percent'].str.replace('---', '0')
df['B_Significant_Strike_Percent'] = df['B_Significant_Strike_Percent'].str.replace('---', '0')
df['R_Takedown_Percent'] = df['R_Takedown_Percent'].str.replace('---', '0')
df['B_Takedown_Percent'] = df['B_Takedown_Percent'].str.replace('---', '0')

df.shape

print(df['date'].dtype)

df['date'] = pd.to_datetime(df['date'], format="%B %d, %Y").dt.strftime("%Y-%m-%d")

limit_date = '2001-04-01'
df = df[(df['date'] > limit_date)]
print(df.shape)

df.head()


print(df['win_by'].unique())

df = df.dropna(subset=['Winner'])

print("Total NaN in dataframe :" , df.isna().sum().sum())
print("Total NaN in each column of the dataframe")
na = []
for index, col in enumerate(df):
    na.append((index, df[col].isna().sum())) 
na_sorted = na.copy()
na_sorted.sort(key = lambda x: x[1], reverse = True) 

for i in range(len(df.columns)):
    print(df.columns[na_sorted[i][0]],":", na_sorted[i][1], "NaN")

def calculate_total_duration(value):
    parts = value.split(':')
    minutes = int(parts[0])
    seconds = int(parts[1])
    total_seconds = minutes * 60 + seconds
    return total_seconds


df['last_round_time'] = df['last_round_time'].apply(calculate_total_duration)

df['time_fought'] = (df['last_round'] - 1) * 5 * 60 + df['last_round_time']

to_drop = ['last_round','last_round_time']
df = df.drop(to_drop, axis=1)

df.head()

df_DQ = df[df['win_by'] == 'DQ']

df_DQ.head(2)

df = df.drop(df[df['win_by'] == 'DQ'].index)

df_DQ = df[df['win_by'] == 'DQ']
df_DQ.head(2)

R_subset = ['R_fighter','R_Knockdown','R_Significant_Strike_Percent','R_Takedown_Percent', 'R_Submission_Attempt','R_Ground_Control', 'win_by', 'No_of_rounds', 'date', 'Fight_type',
            'R_Significant_Strikes_Landed', 'R_Significant_Strikes_Attempted', 'R_Total_Strikes_Landed', 'R_Total_Strikes_Attempted', 'R_Takedowns_Landed', 'R_Takedowns_Attempted', 
            'R_Ground_Strikes_Landed', 'time_fought','Winner']
R_df = df[R_subset]
R_df = R_df.rename(columns=lambda x: x.replace('R_', ''))

R_df['Winner'] = np.where(R_df['Winner'] == R_df['fighter'], 1, 0)

R_df.head()

B_subset = ['B_fighter', 'B_Knockdown', 'B_Significant_Strike_Percent', 'B_Takedown_Percent', 'B_Submission_Attempt',
            'B_Ground_Control', 'win_by', 'No_of_rounds', 'date', 'Fight_type', 'B_Significant_Strikes_Landed',
            'B_Significant_Strikes_Attempted', 'B_Total_Strikes_Landed', 'B_Total_Strikes_Attempted', 
            'B_Takedowns_Landed', 'B_Takedowns_Attempted', 'B_Ground_Strikes_Landed', 'time_fought', 'Winner']
B_df = df[B_subset]
B_df = B_df.rename(columns=lambda x: x.replace('B_', ''))
B_df['Winner'] = np.where(B_df['Winner'] == B_df['fighter'], 1, 0)


B_df.head()

new_df = pd.concat([R_df, B_df])

# sort the combined dataframe by date
ufc = new_df.sort_values(by='date', ascending=False)
# reset the index
ufc = ufc.reset_index(drop=True)

ufc['Ground_Control'] = ufc['Ground_Control'].apply(calculate_total_duration)

ufc.head()

ufc['win_by'].unique()

mapping = {
    'KO/TKO': 10,
    'Submission': 9,
    "TKO - Doctor's Stoppage": 9,
    'Decision - Unanimous': 9,
    'Decision - Majority': 8,
    'Decision - Split': 7
}

ufc['win_by'] = ufc['win_by'].replace(mapping)

ufc.head()

ufc['Fight_type'].unique()

ufc['Fight_type'] = ufc['Fight_type'].str.replace('Bout', '')
ufc['Fight_type'] = ufc['Fight_type'].str.replace('Title', '')
ufc['Fight_type'] = ufc['Fight_type'].str.replace('Tournament', '')
ufc['Fight_type'] = ufc['Fight_type'].str.replace('Ultimate Fighter', '')
ufc['Fight_type'] = ufc['Fight_type'].str.replace('UFC', '')
ufc['Fight_type'] = ufc['Fight_type'].str.replace('Interim', '')
ufc['Fight_type'] = ufc['Fight_type'].str.replace('Brazil', '')
ufc['Fight_type'] = ufc['Fight_type'].str.replace('America', '')
ufc['Fight_type'] = ufc['Fight_type'].str.replace('China', '')
ufc['Fight_type'] = ufc['Fight_type'].str.replace('TUF', '')
ufc['Fight_type'] = ufc['Fight_type'].str.replace('Australia', '')
ufc['Fight_type'] = ufc['Fight_type'].str.replace('Nations', '')
ufc['Fight_type'] = ufc['Fight_type'].str.replace('Canada', '')
ufc['Fight_type'] = ufc['Fight_type'].str.replace('vs.', '')
ufc['Fight_type'] = ufc['Fight_type'].str.replace('UK', '')
ufc['Fight_type'] = ufc['Fight_type'].str.replace('Latin', '')
ufc['Fight_type'] = ufc['Fight_type'].str.replace("Women's", 'W')

ufc['Fight_type'] = ufc['Fight_type'].apply(lambda x: re.sub(r'\d+', ' ', x))

ufc['Fight_type'] = ufc['Fight_type'].str.strip()

ufc.head()

column_dtypes = ufc.dtypes
print(column_dtypes)

columns_to_convert = ['Significant_Strike_Percent', 'Takedown_Percent', 'win_by',
                     'No_of_rounds','Significant_Strikes_Landed','Significant_Strikes_Attempted',
                     'Total_Strikes_Landed','Total_Strikes_Attempted','Takedowns_Landed',
                      'Takedowns_Attempted','Ground_Strikes_Landed']

ufc[columns_to_convert] = ufc[columns_to_convert].astype('int64')

plt.figure(figsize=(25, 20))
corr_matrix = ufc.corr(method='pearson', numeric_only=True).abs()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)

sol = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1)
                 .astype(bool))  # Use `bool` instead of `np.bool`
                 .stack()
                 .sort_values(ascending=False))

# Print the top 10 highest correlations
print(sol[0:10])

ufc.isna().sum()

def assign_gender(row):
    women_weights = ['W Strawweight', 'W Flyweight', 'W Bantamweight', 'W Featherweight']
    if row['Fight_type'] in women_weights:
        return 0
    else:
        return 1


ufc.insert(len(ufc.columns) - 1, 'Gender', ufc.apply(assign_gender, axis=1))

ufc = ufc[ufc['Fight_type'] != 'Catch Weight']

mapping = {
    'W Strawweight': 115,
    'W Flyweight': 125,
    'W Bantamweight': 135,
    'W Featherweight': 145,
    'Flyweight': 125,
    'Bantamweight': 135,
    'Featherweight': 145,
    'Lightweight': 155,
    'Welterweight': 170,
    'Middleweight': 185,
    'Light Heavyweight': 205,
    'Heavyweight': 265,
}
pd.set_option('future.no_silent_downcasting', True)
ufc['Fight_type'] = ufc['Fight_type'].replace(mapping)

ufc.rename(columns={'Fight_type': 'Weight Division'}, inplace=True)


ufc.head()


ufc.describe()


date_column = ufc['date']  # Extract the 'date' column
ufc.drop(columns=['date'], inplace=True)  # Remove the 'date' column from its current position
ufc.insert(0, 'date', date_column)  # Insert the 'date' column at the first position

ufc2= ufc.copy()

print(ufc2.columns[2:-1].tolist())

columns_to_standardize = ufc2.columns[2:-1].tolist()

scaler = StandardScaler()
scaler.fit(ufc2[columns_to_standardize])

ufc2[columns_to_standardize] = scaler.transform(ufc2[columns_to_standardize])

ufc2.head()

# Define the output path for the cleaned CSV
CLEANED_DATA_PATH = './data/avg_attributes.csv'  # You can change the path as needed

# Export the cleaned DataFrame to CSV
ufc2.to_csv(CLEANED_DATA_PATH, index=False)


# Define the path where the scaler will be saved
SCALER_PATH = './models/scaler.joblib'  # You can change the path as needed

# Ensure the 'models' directory exists
import os
os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)

joblib.dump(scaler, SCALER_PATH)

print(f"Scaler successfully saved to {SCALER_PATH}")

with open('./data/raw_total_fight_data.csv', 'r') as file:
    for _ in range(5):
        line = file.readline()
        print(line)

X = ufc2.iloc[:, 2:-1].values
y = ufc2.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LogisticRegression
lr_classifier = LogisticRegression(random_state = 0)
lr_classifier.fit(X_train, y_train)

y_pred = lr_classifier.predict(X_test)
print(y_pred)

new_y_pred = y_pred.reshape(len(y_pred),1)
new_y_test = y_test.reshape(len(y_test),1)
np.concatenate((new_y_pred,new_y_test),1)

from sklearn.metrics import confusion_matrix, accuracy_score
accuracy_score(y_test, y_pred)
cm1 = confusion_matrix(y_pred, y_test)
print(cm1)
print(accuracy_score(y_test, y_pred))

from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators = 50, criterion = 'entropy', random_state = 0)
rf_classifier.fit(X_train, y_train)

y_pred = rf_classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

cm2 = confusion_matrix(y_test, y_pred)
print(cm2)
accuracy_score(y_test, y_pred)

from sklearn.tree import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier(criterion = 'gini', random_state = 0)
dt_classifier.fit(X_train, y_train)

y_pred = dt_classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


cm3 = confusion_matrix(y_test, y_pred)
print(cm3)
accuracy_score(y_test, y_pred)

from sklearn.naive_bayes import GaussianNB
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)


y_pred = nb_classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


cm4 = confusion_matrix(y_test, y_pred)
print(cm4)
accuracy_score(y_test, y_pred)

from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

knn_classifier.fit(X_train, y_train)


y_pred = knn_classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

cm5 = confusion_matrix(y_test, y_pred)
print(cm5)
accuracy_score(y_test, y_pred)

from sklearn.svm import SVC
svm_classifier = SVC(kernel = 'linear', random_state = 0)
svm_classifier.fit(X_train, y_train)


y_pred = svm_classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


cm6 = confusion_matrix(y_test, y_pred)
print(cm6)
accuracy_score(y_test, y_pred)

ufc3 = ufc2.copy()

to_drop = ['Winner']

ufc3 = ufc3.drop(to_drop, axis=1)

ufc3.head()

ufc3.columns.tolist()


def calculate_avg_attributes(dataframe, years):
    df_temp = dataframe.sort_values('date', ascending=False)  # Sort by date in descending order

    grouped_df = df_temp.groupby('fighter').head(years)  # Select up to 'years' number of rows for each fighter

    exclude_columns = ['Weight Division', 'Gender']
    avg_attributes = grouped_df.groupby('fighter').apply(lambda x: x.mean(numeric_only=True) if len(x) > 0 else pd.Series())
    
    # Update 'Weight Division' and 'Gender' columns with the last (latest) value
    last_values = grouped_df.groupby('fighter')[exclude_columns].last()
    avg_attributes[exclude_columns] = last_values
    
    avg_attributes = avg_attributes.reset_index()
    return avg_attributes

avg_attributes_df = calculate_avg_attributes(ufc3, 3)


avg_attributes_df.head()

def get_fighter_row(dataframe, fighter_name):
    row = dataframe.loc[dataframe['fighter'] == fighter_name]
    row_values = row.values.flatten()[1:]  # Exclude the 'fighter' column
    return row_values

def get_win_probability(fighter_name):
    fighter_name = fighter_name
    fighter_row = get_fighter_row(avg_attributes_df, fighter_name)
    win_prob = rf_classifier.predict_proba([fighter_row])[0][1]
    return win_prob

get_win_probability('Stipe Miocic')

def match_probability(fighter_A, fighter_B):
    fighter_A = fighter_A
    fighter_B = fighter_B
    prob_A = get_win_probability(fighter_A)
    prob_B = get_win_probability(fighter_B)
    total_prob = prob_A + prob_B
    probability_A = round(prob_A / total_prob, 2)
    probability_B = round(prob_B / total_prob, 2)
    return probability_A, probability_B


match_probability('Conor McGregor', "Sean O'Malley")


import joblib

model_path = 'models/rf_model.joblib'

joblib.dump(rf_classifier, model_path)

print(f"Model successfully saved to {model_path}")


