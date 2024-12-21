import os

map_seqs = ['seq-01', 'seq-02']
dataset_dir = 'fire/map'
output_file = 'matches.txt'
temporal_window = 5

def get_sorted_files(directory, suffix):
    return sorted([f for f in os.listdir(directory) if f.endswith(suffix)])

with open(output_file, 'w') as f:
    for seq in map_seqs:
        seq_dir = os.path.join(dataset_dir, seq)
        images = get_sorted_files(seq_dir, '.color.png')
        
        num_images = len(images)
        for idx, img1 in enumerate(images):
            # window bounds
            start = max(0, idx - temporal_window)
            end = min(num_images, idx + temporal_window + 1)
            
            for j in range(start, end):
                if j == idx:  # skip self
                    continue
                
                img2 = images[j]
                img1_path = os.path.join(seq_dir, img1)
                img2_path = os.path.join(seq_dir, img2)

                f.write(f"{img1_path} {img2_path}\n")

print(f"Success.")
