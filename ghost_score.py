##Script to run Ghost Writer Score
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class MultiClassRegression(nn.Module):
    def __init__(self, dim, num_classes):
        super(MultiClassRegression, self).__init__()
        self.layer1 = nn.Linear(dim, 128)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        probs = self.layer1(x)
        probs = self.relu(probs)
        probs = self.layer2(probs)
        return probs
    
def get_ghost_score(row, predicted):
    get_art = hip.loc[row.name,'artist']
    ref_id = artist_mapping[get_art]
    ref_prob = predicted[ref_id]
    
    diff = predicted - ref_prob 
    sim = torch.tensor([1/(1+abs(d)) for d in diff])
    
    #since the sim for our reference artist will always be 1
    total = sum(sim) - 1
    
    p_ghost = torch.tensor([s/total for s in sim])
    
    for i, p in enumerate(p_ghost):
        if i == ref_id:
            p_ghost[i] = -1
        p_ghost[i] = p
    
    return torch.max(p_ghost).tolist(), torch.argmax(p_ghost).tolist()

#loading in our best model
model_state_dict = torch.load('models/model_parameters_sgd.pth',weights_only=True)

hip = pd.read_csv('data/cleaned_hip_dat.csv')

all_artists = hip['artist'].unique().tolist()
artist_mapping = {artist: idx for idx, artist in enumerate(all_artists)}
id_to_name = {v: k for k, v in artist_mapping.items()}

train_data = hip[hip['pot_ghost'] == 0]
test_data = hip[hip['pot_ghost'] == 1]

tfidf = TfidfVectorizer(
    lowercase=True,
    max_features=50000,     # recommended cap
    ngram_range=(1, 2)      # big improvement for lyrics
)

#getting our predict values 
train_data['artist_id'] = train_data['artist'].map(artist_mapping)
train_data = train_data.loc[:,['main_artist_lyrics_joined', 'artist_id']]

test_data['artist_id'] = test_data['artist'].map(artist_mapping)
test_data_clean = test_data.loc[:,['main_artist_lyrics_joined', 'artist_id']]

train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42, stratify=train_data['artist_id'])

X_train = tfidf.fit_transform(train_data['main_artist_lyrics_joined'])
x_train = torch.tensor(X_train.toarray(), dtype=torch.float32)
y_train = torch.tensor(train_data['artist_id'].values, dtype = torch.long)

X_val = tfidf.transform(val_data['main_artist_lyrics_joined'])
x_val = torch.tensor(X_val.toarray(), dtype=torch.float32)
y_val = torch.tensor(val_data['artist_id'].values,dtype=torch.long)


X_test = tfidf.transform(test_data_clean['main_artist_lyrics_joined'])
x_test = torch.tensor(X_test.toarray(), dtype=torch.float32)
y_test = torch.tensor(test_data_clean['artist_id'].values, dtype = torch.long)

model = MultiClassRegression(x_train.shape[1], len(artist_mapping))

model.load_state_dict(model_state_dict)

#getting the predicted labels on test
pred_test = model(x_test)
test_label = []


for i in range(len(pred_test)):
    test_label.append(torch.argmax(pred_test[i]))
    
#getting the predicted labels on val 
pred_val = model(x_val)
val_label = []

for i in range(len(pred_val)):
    val_label.append(torch.argmax(pred_val[i]))
    
#getting the simiarity scores from these predictions
ghost_p_test = []
ghost_lab_test = []

for i in range(len(test_data)):
    p_ghost, ghost_label = get_ghost_score(test_data.iloc[i,],pred_test[i])

    ghost_p_test.append(p_ghost)
    ghost_lab_test.append(id_to_name[ghost_label])
    
ghost_p_val = []
ghost_lab_val = []

for i in range(len(val_data)):
    p_ghost, ghost_label = get_ghost_score(val_data.iloc[i,],pred_val[i])

    ghost_p_val.append(p_ghost)
    ghost_lab_val.append(id_to_name[ghost_label])
    
## performing bootstrapping on val ghost scores to get distribution for artist to artist simiarities
num_sim = 5000

keys = ['Tyler, The Creator-Mac Miller', 'Tyler, The Creator-Kendrick Lamar', 'Tyler, The Creator-Quentin Miller', 'Tyler, The Creator-Drake', 'Tyler, The Creator-J. Cole', 'Tyler, The Creator-A Tribe Called Quest', 'Tyler, The Creator-Kanye West', 'Tyler, The Creator-Soulja Boy', 'Tyler, The Creator-Jay-Z', 'Tyler, The Creator-Big L', 'Tyler, The Creator-2Pac', 'Tyler, The Creator-Joey Bada$$', 
        'A Tribe Called Quest-Mac Miller', 'A Tribe Called Quest-Kendrick Lamar', 'A Tribe Called Quest-Quentin Miller', 'A Tribe Called Quest-Drake', 'A Tribe Called Quest-J. Cole', 'A Tribe Called Quest-Kanye West', 'A Tribe Called Quest-Soulja Boy', 'A Tribe Called Quest-Jay-Z', 'A Tribe Called Quest-Big L', 'A Tribe Called Quest-Tyler, The Creator', 'A Tribe Called Quest-2Pac', 'A Tribe Called Quest-Joey Bada$$', 
        '2Pac-Mac Miller', '2Pac-Kendrick Lamar', '2Pac-Quentin Miller', '2Pac-Drake', '2Pac-J. Cole', '2Pac-A Tribe Called Quest', '2Pac-Kanye West', '2Pac-Soulja Boy', '2Pac-Jay-Z', '2Pac-Big L', '2Pac-Tyler, The Creator', '2Pac-Joey Bada$$', 
        'J. Cole-Mac Miller', 'J. Cole-Kendrick Lamar', 'J. Cole-Quentin Miller', 'J. Cole-Drake', 'J. Cole-A Tribe Called Quest', 'J. Cole-Kanye West', 'J. Cole-Soulja Boy', 'J. Cole-Jay-Z', 'J. Cole-Big L', 'J. Cole-Tyler, The Creator', 'J. Cole-2Pac', 'J. Cole-Joey Bada$$', 
        'Drake-Mac Miller', 'Drake-Kendrick Lamar', 'Drake-Quentin Miller', 'Drake-J. Cole', 'Drake-A Tribe Called Quest', 'Drake-Kanye West', 'Drake-Soulja Boy', 'Drake-Jay-Z', 'Drake-Big L', 'Drake-Tyler, The Creator', 'Drake-2Pac', 'Drake-Joey Bada$$', 
        'Kendrick Lamar-Mac Miller', 'Kendrick Lamar-Quentin Miller', 'Kendrick Lamar-Drake', 'Kendrick Lamar-J. Cole', 'Kendrick Lamar-A Tribe Called Quest', 'Kendrick Lamar-Kanye West', 'Kendrick Lamar-Soulja Boy', 'Kendrick Lamar-Jay-Z', 'Kendrick Lamar-Big L', 'Kendrick Lamar-Tyler, The Creator', 'Kendrick Lamar-2Pac', 'Kendrick Lamar-Joey Bada$$', 
        'Mac Miller-Kendrick Lamar', 'Mac Miller-Quentin Miller', 'Mac Miller-Drake', 'Mac Miller-J. Cole', 'Mac Miller-A Tribe Called Quest', 'Mac Miller-Kanye West', 'Mac Miller-Soulja Boy', 'Mac Miller-Jay-Z', 'Mac Miller-Big L', 'Mac Miller-Tyler, The Creator', 'Mac Miller-2Pac', 'Mac Miller-Joey Bada$$', 
        'Soulja Boy-Mac Miller', 'Soulja Boy-Kendrick Lamar', 'Soulja Boy-Quentin Miller', 'Soulja Boy-Drake', 'Soulja Boy-J. Cole', 'Soulja Boy-A Tribe Called Quest', 'Soulja Boy-Kanye West', 'Soulja Boy-Jay-Z', 'Soulja Boy-Big L', 'Soulja Boy-Tyler, The Creator', 'Soulja Boy-2Pac', 'Soulja Boy-Joey Bada$$', 
        'Kanye West-Mac Miller', 'Kanye West-Kendrick Lamar', 'Kanye West-Quentin Miller', 'Kanye West-Drake', 'Kanye West-J. Cole', 'Kanye West-A Tribe Called Quest', 'Kanye West-Soulja Boy', 'Kanye West-Jay-Z', 'Kanye West-Big L', 'Kanye West-Tyler, The Creator', 'Kanye West-2Pac', 'Kanye West-Joey Bada$$', 
        'Big L-Mac Miller', 'Big L-Kendrick Lamar', 'Big L-Quentin Miller', 'Big L-Drake', 'Big L-J. Cole', 'Big L-A Tribe Called Quest', 'Big L-Kanye West', 'Big L-Soulja Boy', 'Big L-Jay-Z', 'Big L-Tyler, The Creator', 'Big L-2Pac', 'Big L-Joey Bada$$', 
        'Joey Bada$$-Mac Miller', 'Joey Bada$$-Kendrick Lamar', 'Joey Bada$$-Quentin Miller', 'Joey Bada$$-Drake', 'Joey Bada$$-J. Cole', 'Joey Bada$$-A Tribe Called Quest', 'Joey Bada$$-Kanye West', 'Joey Bada$$-Soulja Boy', 'Joey Bada$$-Jay-Z', 'Joey Bada$$-Big L', 'Joey Bada$$-Tyler, The Creator', 'Joey Bada$$-2Pac', 
        'Quentin Miller-Mac Miller', 'Quentin Miller-Kendrick Lamar', 'Quentin Miller-Drake', 'Quentin Miller-J. Cole', 'Quentin Miller-A Tribe Called Quest', 'Quentin Miller-Kanye West', 'Quentin Miller-Soulja Boy', 'Quentin Miller-Jay-Z', 'Quentin Miller-Big L', 'Quentin Miller-Tyler, The Creator', 'Quentin Miller-2Pac', 'Quentin Miller-Joey Bada$$', 
        'Jay-Z-Mac Miller', 'Jay-Z-Kendrick Lamar', 'Jay-Z-Quentin Miller', 'Jay-Z-Drake', 'Jay-Z-J. Cole', 'Jay-Z-A Tribe Called Quest', 'Jay-Z-Kanye West', 'Jay-Z-Soulja Boy', 'Jay-Z-Big L', 'Jay-Z-Tyler, The Creator', 'Jay-Z-2Pac', 'Jay-Z-Joey Bada$$']

sim_est = {k: [] for k in keys}

artist_to_indices = {
    aid: val_data.reset_index().index[val_data["artist_id"] == aid].tolist()
    for aid in range(13)
}

for sim in range(num_sim):
    samples = {k: [] for k in keys}

    for ref_artist_id in range(13):
        ref_name = id_to_name[ref_artist_id]

        real_inds = artist_to_indices[ref_artist_id]
        n_real = len(real_inds)

        # bootstrap indices with replacement
        boot_inds = np.random.choice(real_inds, size=n_real, replace=True)

        for ind in boot_inds:

            p_for_ref = pred_val[ind][ref_artist_id]
            diff = pred_val[ind] - p_for_ref

            similar_score = torch.tensor([1/(1+abs(d)) for d in diff])
            total = sum(similar_score) - 1
            p_ghost = torch.tensor([s/total for s in similar_score])

            for i, p in enumerate(p_ghost):
                tar_name = id_to_name[i]
                if tar_name == ref_name:
                    continue

                key = f"{ref_name}-{tar_name}"
                samples[key].append(float(p))
                
    #calculate mean and std for each art from each sample
    for key, value in samples.items():
        sim_est[key].append([np.mean(value), np.std(value)])


#creating final bootstrap estimate
df = pd.DataFrame(sim_est)

boot_est = {k: 0 for k in keys}

for key in keys:
    boot_est[key] = [np.mean(df[key].str[0]), np.std(df[key].str[1])]

#use these bootstrap samples to calculate mean and std for each art to art distribution
art_to_art_info = {}
for key, value in boot_est.items():
    art_to_art_info[key] = [np.mean(value), np.std(value)]
    
#now will use these values to create z-scores for our test ghost scores
z_scores_sgd_boot = []

for i in range(len(test_data)):
    row_ref_art = test_data.iloc[i,0]
    row_tar_art_sgd = ghost_lab_test[i]
    
    tar_art_mean_sgd, tar_art_std_sgd = art_to_art_info[f'{row_ref_art}-{row_tar_art_sgd}']
    z_scores_sgd_boot.append((ghost_p_test[i]-tar_art_mean_sgd)/tar_art_std_sgd)

test_copy = test_data.copy()
test_copy['predicted ghost writer'] = ghost_lab_test
test_copy['z score sgd boot'] = z_scores_sgd_boot
print(test_copy.loc[:,['song_title','artist','predicted ghost writer','z score sgd boot']])


print(ghost_lab_test)


##trying score on more songs
ghost_test = pd.read_csv('data/cleaned_ghost_test.csv')

ghost_test['artist_id'] = ghost_test['artist'].map(artist_mapping)
ghost_test = ghhost_test.loc[:,['main_artist_lyrics_joined', 'artist_id']]

X_test = tfidf.transform(ghost_test['main_artist_lyrics_joined'])
x_test = torch.tensor(X_test.toarray(), dtype=torch.float32)
y_test = torch.tensor(ghost_test['artist_id'].values, dtype = torch.long)

#getting the predicted labels on test
pred_test = model(x_test)
test_label = []

for i in range(len(pred_test)):
    test_label.append(torch.argmax(pred_test[i]))

#getting the simiarity scores from these predictions
ghost_p_test = []
ghost_lab_test = []

for i in range(len(ghost_test)):
    p_ghost, ghost_label = get_ghost_score(ghost_test.iloc[i,],pred_test[i])

    ghost_p_test.append(p_ghost)
    ghost_lab_test.append(id_to_name[ghost_label])

#now will use these values to create z-scores for our test ghost scores
z_scores_sgd_boot = []

for i in range(len(ghost_test)):
    row_ref_art = test_data.iloc[i,0]
    row_tar_art_sgd = ghost_lab_test[i]
    
    tar_art_mean_sgd, tar_art_std_sgd = art_to_art_info[f'{row_ref_art}-{row_tar_art_sgd}']
    z_scores_sgd_boot.append((ghost_p_test[i]-tar_art_mean_sgd)/tar_art_std_sgd)

test_copy = ghost_test.copy()
test_copy['predicted ghost writer'] = ghost_lab_test
test_copy['z score sgd boot'] = z_scores_sgd_boot
print(test_copy.loc[:,['song_title','artist','predicted ghost writer','z score sgd boot']])
