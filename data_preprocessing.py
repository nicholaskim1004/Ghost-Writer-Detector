## Data Preprocessing Script
import pandas as pd
import numpy as np
import ast
import re
import unicodedata

# function to standarize input of artist names
def normalize_artist_name(name):
    if not name:
        return ""
    # 1. unicode normalize (NFKD) -> removes accents/diacritics
    name = unicodedata.normalize("NFKD", name)
    # 2. lowercase and trim
    name = name.lower().strip()
    # 3. turn one-or-more dollar signs into a single 's'
    name = re.sub(r'\$+', 's', name)
    # 4. unify common separators to spaces (hyphens/underscores/ampersand/at/slash)
    name = re.sub(r'[-_&@/]+', ' ', name)
    # 5. remove any remaining punctuation (keep letters, numbers, spaces)
    #    note: \w includes underscore; we've already removed underscores above
    name = re.sub(r'[^a-z0-9\s]', '', name)
    # 6. collapse whitespace and strip
    name = re.sub(r'\s+', ' ', name).strip()
    if name in ['tyler','the creator']:
        return 'tyler the creator'
    
    return name

#function to get artist name from headers in lyrics
#will only keep if it exists
def extract_artists(section):
    """
    Extracts artists from a bracketed song section like:
    [Intro: Bēkon & Kid Capri]
    
    Returns:
        list of artist names OR None if no artists exist.
    """
    if section is None:
        return None

    # Step 1: detect pattern like: [Intro: artist(s)]
    match = re.match(r"\[(?:[^\]:]+)(?::\s*(.*?))?\]$", section.strip())
    if not match:
        return None

    artists_str = match.group(1)  # captures everything after the colon
    
    # Step 2: return None if no artist portion exists
    if not artists_str or artists_str.strip() == "":
        return None

    # Step 3: split artists by “&” or “,” and clean
    artists = re.split(r"&|,|\band\b", artists_str)
    artists = [a.strip() for a in artists if a.strip() != ""]

    return artists if artists else None

#function to get all artists with a speaking part in the song
#will skip if the artist who the song is stored under as
def get_featured_artist(row):
    blocks = re.split(r'(\[[^\]]+\])', row['lyrics'])
    prim_art = normalize_artist_name(row['artist'])
    
    featured_art = []
    seen = set()
    
    for i in range(len(blocks)):
        artist = extract_artists(blocks[i])
        if artist is not None:
            for a in artist:
                art = normalize_artist_name(a)
                
                if art in ["q tip", "phife dawg", "ali shaheed muhammad"]:
                    art = "A Tribe Called Quest"
                if art not in seen and art != prim_art:
                    seen.add(art)
                    featured_art.append(a)
    
    if len(featured_art)==0:
        return None
    
    return featured_art

#function to get only the verses that the artist the song its stored under sang
def get_main_artist_verses(lyrics, main_artist, corpus_artists):
    
    # Normalize everything to lowercase
    main_artist = normalize_artist_name(main_artist)
    corpus_artists = [normalize_artist_name(c) for c in corpus_artists]

    # Split lyrics into [Header][Text]
    blocks = re.split(r'(\[[^\]]+\])', lyrics)
    verses = []

    for i in range(len(blocks)):
        header = blocks[i]

        # Only process bracketed headers
        if not re.match(r"\[.*?\]", header):
            continue

        text = blocks[i+1] if i + 1 < len(blocks) else ""

        # CASE 1 — No colon → no artist info → keep automatically
        if ":" not in header:
            cleaned = re.sub(r"\(.*?\)", "", text, flags=re.DOTALL)
            verses.append(cleaned.strip())
            continue

        # CASE 2 — Colon exists → try to extract artists
        artists = extract_artists(header)

        # If colon exists but extract_artists returns None → fallback to keep
        if artists is None:
            cleaned = re.sub(r"\(.*?\)", "", text, flags=re.DOTALL)
            verses.append(cleaned.strip())
            continue

        # Normalize
        artists = [normalize_artist_name(a) for a in artists]

        # If artist not listed → skip
        if main_artist not in artists:
            continue

        # Now artist is in (e.g., [Verse: Kendrick Lamar])
        art_index = artists.index(main_artist)

        if art_index == 0:
            # main voice generally in first index → remove adlibs
            cleaned = re.sub(r"\(.*?\)", "", text, flags=re.DOTALL)
            verses.append(cleaned.strip())
        else:
            # after is generally set as background artist → grab only the parentheses lines
            adlibs = re.findall(r"\((.*?)\)", text, flags=re.DOTALL)
            verses.extend([a.strip() for a in adlibs])

    return verses

#helper function to find potential 'ghost written' cases
def flag_missing_header_artists(lyrics, main_artist, writers, corpus_artists, group_members=None):
    """
    Returns 1 if the lyric header does NOT include the main artist,
    and also does NOT include any other contributing artists from our corpus.
    """
    # Determine allowed names for the main artist
    allowed_artists = group_members.get(main_artist, [main_artist]) if group_members else [main_artist]
    allowed_artists = [a.lower() for a in allowed_artists]
    
    # Get all lyric headers in the song
    headers = re.findall(r'\[(?:[^\]:]*?:\s*)([^\]]+)\]', lyrics)
    
    # Split header artists by & / and, flatten all headers
    header_artists = []
    for h in headers:
        header_artists.extend([a.strip().lower() for a in re.split(r'\s*(?:&|and)\s*', h)])
    
    # Check if main artist is in any header
    main_in_header = any(a in allowed_artists for a in header_artists)
    
    # Get all other contributing artists from writers who are in our corpus
    other_contributors = [w for w in writers if w in corpus_artists and w != main_artist]
    
    # Check if any of these contributors appear in the headers
    contributor_in_header = any(a in other_contributors for a in header_artists)
    
    # Flag = 1 if main artist not in header and none of the other contributors are in header
    if not main_in_header and not contributor_in_header and len(other_contributors) > 0:
        return 1, other_contributors
    return 0, None

## Starting data cleaning
hip = pd.read_csv('data/hiphop_corpus_with_lyrics.csv')

#convert writers to be a list rather than a string
hip['writers'] = hip['writers'].apply(ast.literal_eval)

#manual data cleaning 
#these songs were realeased by other artist and the artist its stored under doesn't have a speaking part
hip.loc[316,'artist'] = 'Kendrick Lamar'
hip.loc[1227:1233,'artist'] = "Kanye West"
hip.loc[1235,'artist'] = "Kanye West"
hip.loc[1242:1249,'artist'] = "Drake"
hip.loc[1480,'artist'] = "Drake"
hip.loc[1574,'artist'] = "Drake"
hip.loc[1675,'artist'] = "Kendrick Lamar"
hip.loc[1710,'artist'] = "Jay-Z"

hip['featured_artist'] = hip.apply(
    lambda row: get_featured_artist(row),
    axis=1
)
all_artists = hip['artist'].unique().tolist()

featured = hip['featured_artist']

need_row = []
new_rows = []

for row in hip.itertuples(index=True):
    idx = row.Index
    feat_list = row.featured_artist
    
    if feat_list is None:
        continue
    for f in feat_list:
        if f in all_artists:
            song_feat_in = row.title_clean
            art_discogrophy = hip[hip['artist']==f]['title_clean']
            
            if song_feat_in not in art_discogrophy:
                need_row.append(idx)
                
                #swapping featured list to be correct
                new_feat = []
                for cur_f in row.featured_artist:
                    if cur_f == f:
                        new_feat.append(row.artist)
                    else:
                        new_feat.append(cur_f)
                        
                new_rows.append({'artist': f, 
                               'album': row.album,
                               'song_title': row.song_title,
                               'date': row.date,
                               'writers': row.writers,
                               'lyrics': row.lyrics,
                               'title_clean': row.title_clean,
                               'featured_artist': new_feat})
                
hip = pd.concat([hip, pd.DataFrame(new_rows)], ignore_index=True)

hip['main_artist_lyrics'] = hip.apply(
    lambda row: get_main_artist_verses(row['lyrics'], row['artist'], all_artists),
    axis=1
)

# Join the verses back into a single string per song
hip['main_artist_lyrics_joined'] = hip['main_artist_lyrics'].apply(lambda x: "\n".join(x))

#cleaning by removing \n
hip['main_artist_lyrics_joined'] = hip['main_artist_lyrics_joined'].str.replace('\n', ' ')

#want to count these artist to be treated from coming from same artist
group_members = {
    "A Tribe Called Quest": ["Q-Tip", "Phife Dawg", "Ali Shaheed Muhammad"]
}

hip[['pot_ghost', 'pot_ghost_name']] = hip.apply(
    lambda row: pd.Series(flag_missing_header_artists(
        row['lyrics'],
        row['artist'],
        row['writers'],
        all_artists,
        group_members
    )),
    axis=1
)

hip = hip[hip['main_artist_lyrics_joined'].fillna('').str.strip() != '']
hip = hip.reset_index(drop=True)

#manually removing these songs as they aren't legit
row_to_rm = [1029,1030,1032,1033,1035,1037,1041,1042,1043,1067,1068,1069,1070,1071,
             1072,1073,1075,1076,1077,1078,1079,1080,1081,1082,1083,1084,1085,1086,
             1087,1091,1092,1093,1094,1095,1096,1097,1658,1119,1173,1196,1202,1203,
             1205,1226,1638,1231,1232,1247,1250,1252,1287,1288,1289,1290,1291,1292,
             1293,1294,1295,1296,1297,1298,1299,1300,1301,1302,1305,1308,1309,1327,
             1652,1346,1348,1361,1370,1376,1382,1385,1386,1388,1389,1390,1391,1392,
             1397,1399,1400,1403,1422,1423,1424,1425,1426,1427,1428,1429,1430,1431,
             1432,1433,1434,1435,1436,1437,1438,1476,1480,1496,1499,1500,1501,1502,
             1503,1504,1505,1506,1507,1508,1509,1510,1511,1512,1513,1514,1515,1516,
             1517,1518,1519,1521,1522,1526,1534,1542,1545,1558,1560,1571,1574,1576,
             1577,1578,1579,1580,1581,1582,1583,1601]

hip.drop(index=row_to_rm, inplace=True)
hip = hip.reset_index(drop=True)

hip.to_csv('data/cleaned_hip_dat.csv', index = False)

