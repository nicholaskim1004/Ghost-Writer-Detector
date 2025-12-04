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

samples = {}

for sim in range(num_sim):
    rand_ind = np.random.randint(0,len(val_data))
    
    ind_to_main = val_data.index[rand_ind]

    ref_artist = hip['artist'][ind_to_main]
    ref_artist_id = artist_mapping[ref_artist]
    
    p_for_ref = pred_val[rand_ind][ref_artist_id]
        
    diff = pred_val[rand_ind] - p_for_ref

    similar_score = torch.tensor([1/(1+abs(d)) for d in diff])

    total = sum(similar_score) - 1
    
    p_ghost = torch.tensor([s/total for s in similar_score])
    
    for i, p in enumerate(p_ghost):
        tar_artist = id_to_name[i]
        if tar_artist == ref_artist:
            continue
        
        key = f"{ref_artist}-{tar_artist}"
        
        samples.setdefault(key, []).append(float(p))

#use these bootstrap samples to calculate mean and std for each art to art distribution
art_to_art_info = {}
for key, value in samples.items():
    art_to_art_info[key] = [np.mean(value), np.std(value)]
    
#now will use these values to create z-scores for our test ghost scores
z_scores_sgd_boot = []

for i in range(len(test_data)):
    row_ref_art = test_data.iloc[i,0]
    row_tar_art_sgd = ghost_lab_test[i]
    
    tar_art_mean_sgd, tar_art_std_sgd = art_to_art_info[f'{row_ref_art}-{row_tar_art_sgd}']
    z_scores_sgd_boot.append((ghost_p_test-tar_art_mean_sgd)/tar_art_std_sgd)

test_copy = test_data.copy()
test_copy['predicted ghost writer'] = ghost_lab_test
test_copy['z score sgd boot'] = z_scores_sgd_boot
print(test_copy.loc[:,['song_title','artist','predicted ghost writer','z score sgd boot']])


