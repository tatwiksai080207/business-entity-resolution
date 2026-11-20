import pandas as pd

# Load training data
s1 = pd.read_csv('dataset/train/train_source1.tsv', sep='\t')
s2 = pd.read_csv('dataset/train/train_source2.tsv', sep='\t')
s3 = pd.read_csv('dataset/train/train_source3.tsv', sep='\t')
gt = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t')

print('=== TRAINING DATA SHAPES ===')
print(f'Source 1: {s1.shape}')
print(f'Source 2: {s2.shape}')
print(f'Source 3: {s3.shape}')
print(f'Ground Truth: {gt.shape}')

print('\n=== COLUMNS ===')
print(f'S1 columns: {list(s1.columns)}')
print(f'GT columns: {list(gt.columns)}')

print('\n=== COUNTRY DISTRIBUTION ===')
print('Source 1:', s1['country'].value_counts().to_dict())
print('Source 2:', s2['country'].value_counts().to_dict())
print('Source 3:', s3['country'].value_counts().to_dict())

print('\n=== MISSING VALUES ===')
print('S1 nulls:', s1.isnull().sum().to_dict())
print('S2 nulls:', s2.isnull().sum().to_dict())
print('S3 nulls:', s3.isnull().sum().to_dict())

print('\n=== SAMPLE S1 RECORDS ===')
for i, row in s1.head(5).iterrows():
    eid = row["entity_id"]
    name = str(row["business_name"])[:80]
    addr = str(row["business_address"])[:80]
    country = row["country"]
    print(f'  {eid} | {name} | {addr} | {country}')

print('\n=== SAMPLE S2 RECORDS ===')
for i, row in s2.head(5).iterrows():
    eid = row["entity_id"]
    name = str(row["business_name"])[:80]
    addr = str(row["business_address"])[:80]
    country = row["country"]
    print(f'  {eid} | {name} | {addr} | {country}')

print('\n=== SAMPLE S3 RECORDS ===')
for i, row in s3.head(5).iterrows():
    eid = row["entity_id"]
    name = str(row["business_name"])[:80]
    addr = str(row["business_address"])[:80]
    country = row["country"]
    print(f'  {eid} | {name} | {addr} | {country}')

print('\n=== GROUND TRUTH STATS ===')
print(f'Total S1 entities in GT: {len(gt)}')
singletons = gt['matched_entity_ids'].isna().sum()
print(f'Singletons (no match): {singletons} ({singletons/len(gt)*100:.1f}%)')
non_singletons = gt[gt['matched_entity_ids'].notna()]
if len(non_singletons) > 0:
    match_counts = non_singletons['matched_entity_ids'].str.split(',').apply(len)
    print(f'Non-singletons: {len(non_singletons)}')
    print(f'Avg matches per non-singleton: {match_counts.mean():.2f}')
    print(f'Max matches for a single entity: {match_counts.max()}')
    print(f'Match count distribution:')
    for k, v in match_counts.value_counts().sort_index().head(15).items():
        print(f'  {k} matches: {v}')

# Check S2 vs S3 distribution in matches
all_matches = non_singletons['matched_entity_ids'].str.cat(sep=',').split(',')
s2_count = sum(1 for m in all_matches if m.startswith('S2-'))
s3_count = sum(1 for m in all_matches if m.startswith('S3-'))
print(f'\nMatched IDs from S2: {s2_count}, from S3: {s3_count}')

print('\n=== SAMPLE GROUND TRUTH ===')
for i, row in gt.head(15).iterrows():
    sid = row["source1_entity_id"]
    mids = row["matched_entity_ids"]
    print(f'  {sid} -> {mids}')

# Show some actual matching pairs to understand noise
print('\n=== EXAMPLE MATCHING PAIRS (showing name/addr differences) ===')
s2_dict = s2.set_index('entity_id').to_dict('index')
s3_dict = s3.set_index('entity_id').to_dict('index')
s1_dict = s1.set_index('entity_id').to_dict('index')

count = 0
for i, row in non_singletons.iterrows():
    if count >= 10:
        break
    s1_id = row['source1_entity_id']
    if s1_id not in s1_dict:
        continue
    s1_rec = s1_dict[s1_id]
    matched_ids = row['matched_entity_ids'].split(',')
    for mid in matched_ids[:2]:
        mid = mid.strip()
        if mid.startswith('S2-') and mid in s2_dict:
            rec = s2_dict[mid]
            src = 'S2'
        elif mid.startswith('S3-') and mid in s3_dict:
            rec = s3_dict[mid]
            src = 'S3'
        else:
            continue
        print(f'\n  S1: {s1_rec["business_name"][:60]} | {str(s1_rec["business_address"])[:60]} | {s1_rec["country"]}')
        print(f'  {src}: {rec["business_name"][:60]} | {str(rec["business_address"])[:60]} | {rec["country"]}')
        count += 1
