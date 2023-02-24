# Description: Balanced frames generator
# Must be executed as a script and must return two list of frames, train and validation

def run():
    with open("utils/data/extensions/balanced_frames.txt", "r") as f:
        content = f.read().splitlines()
        
    keywords = ["TRAIN_BALANCED", "VAL_BALANCED"]

    ids = {keyword: [] for keyword in keywords}
    current_keyword = None

    for row in content:
        if row[0] == "%" and row[1:] in keywords:
            current_keyword = row[1:]
        else:
            ids[current_keyword].append(row)
    
    train_ids = ids["TRAIN_BALANCED"]
    val_ids = ids["VAL_BALANCED"]
    
    return train_ids, val_ids

if __name__ == '__main__':
    # If executed as a script, run the generator
    run()

