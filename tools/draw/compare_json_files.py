import json
import re
import os
import sys
from typing import Dict, List, Tuple

def preprocess_filename(filename: str) -> str:
    """Preprocess filename according to the existing logic"""
    match = re.search(r'(images)', filename)
    if match:
        date_index = match.start()
        return filename[date_index:]
    return filename

def load_jsonl_file(filepath: str) -> List[Dict]:
    """Load and parse a JSONL file"""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                data.append(json.loads(line.strip()))
    return data

def compare_json_files(file_a: str, file_b: str) -> List[Dict]:
    """Compare two JSONL files and summarize differences in correctness"""
    data_a = load_jsonl_file(file_a)
    data_b = load_jsonl_file(file_b)
    
    differences = []
    
    # Create lookup dictionaries using preprocessed filenames
    samples_a = {preprocess_filename(item['filename']): item for item in data_a}
    samples_b = {preprocess_filename(item['filename']): item for item in data_b}
    
    # Find all unique filenames
    all_filenames = set(samples_a.keys()) | set(samples_b.keys())
    
    # Compare each sample
    for filename in sorted(all_filenames):
        sample_a = samples_a.get(filename)
        sample_b = samples_b.get(filename)
        
        if sample_a and sample_b:
            if sample_a['correct'] != sample_b['correct']:
                differences.append({
                    'filename': filename,
                    'tag': sample_a['tag'],
                    'prompt': sample_a['prompt'],
                    'file_a': sample_a['filename'],
                    'file_b': sample_b['filename'],
                    'file_a_correct': sample_a['correct'],
                    'file_b_correct': sample_b['correct'],
                    'file_a_reason': sample_a.get('reason', ''),
                    'file_b_reason': sample_b.get('reason', '')
                })
    
    return differences

def summarize_differences(differences: List[Dict], file_a: str, file_b: str, tag: str) -> None:
    """Create a PDF summary of the differences with images using matplotlib"""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from PIL import Image as PILImage
    import numpy as np
    from tqdm import tqdm
    
    # Create PDF with matplotlib
    with PdfPages(f'comparison_{tag}.pdf') as pdf:
        # Create first page with input file information
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.7, "Comparison Results", 
                horizontalalignment='center', verticalalignment='center', 
                fontsize=16, weight='bold')
        plt.text(0.5, 0.5, f"Left : {os.path.dirname(file_a)}", 
                horizontalalignment='center', verticalalignment='center', 
                fontsize=10)
        plt.text(0.5, 0.4, f"Right: {os.path.dirname(file_b)}", 
                horizontalalignment='center', verticalalignment='center', 
                fontsize=10)
        plt.text(0.5, 0.2, f"Total differences found: {len(differences)}", 
                horizontalalignment='center', verticalalignment='center', 
                fontsize=14)
        plt.axis('off')
        pdf.savefig()
        plt.close()

        # Process each difference
        for diff in tqdm(differences, desc="Generating PDF pages"):
            # Create a new figure for each comparison
            plt.figure(figsize=(16, 10))
            
            # Set title and subtitle
            plt.suptitle(f"Sample: {diff['filename']}", fontsize=16, y=0.95)
            plt.figtext(0.5, 0.90, f"Tag: {diff['tag']}, Prompt: {diff['prompt']}", 
                       wrap=True, horizontalalignment='center', fontsize=10)
            
            try:
                # Load and display both images
                img_a = PILImage.open(diff['file_a'])
                img_b = PILImage.open(diff['file_b'])
                
                # Create subplots for the two images
                plt.subplot(1, 2, 1)
                plt.imshow(np.array(img_a))
                plt.axis('off')
                title_a = f"{'✓ Correct' if diff['file_a_correct'] else '✗ Incorrect'}\n{diff['file_a_reason']}"
                plt.title(title_a, wrap=True)
                
                plt.subplot(1, 2, 2)
                plt.imshow(np.array(img_b))
                plt.axis('off')
                title_b = f"{'✓ Correct' if diff['file_b_correct'] else '✗ Incorrect'}\n{diff['file_b_reason']}"
                plt.title(title_b, wrap=True)
                
            except Exception as e:
                plt.figtext(0.5, 0.5, f"Error loading images: {str(e)}", 
                          wrap=True, horizontalalignment='center')
            
            # Adjust layout and save page
            plt.tight_layout()
            pdf.savefig()
            plt.close()
    
    print(f"PDF report generated as 'comparison_results.pdf'")

def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_json_files.py <file_a> <file_b>")
        sys.exit(1)
    
    file_a = sys.argv[1]
    file_b = sys.argv[2]
    
    if not os.path.exists(file_a) or not os.path.exists(file_b):
        print("Error: One or both files do not exist!")
        sys.exit(1)
    
    differences = compare_json_files(file_a, file_b)

    tagged_diff = {}
    for d in differences:
        if d['tag'] not in tagged_diff:
            tagged_diff[d['tag']] = [d]
        else:
            tagged_diff[d['tag']].append(d)

    tag_of_interest = 'two_object'
    summarize_differences(tagged_diff[tag_of_interest], file_a, file_b, tag_of_interest)
    
    # for tag, diff in tagged_diff.items():
    #     summarize_differences(diff, file_a, file_b, tag)

if __name__ == "__main__":
    main()