import os

DATA_DIR = 'data'

def find_missing_files():
    """Finds missing numbers in the sequence of files."""
    
    if not os.path.exists(DATA_DIR):
        print(f"Error: Directory '{DATA_DIR}' not found.")
        return

    file_numbers = set()
    file_list = os.listdir(DATA_DIR)

    for filename in file_list:
        if filename.endswith('.json'):
            try:
                # Extract the numeric prefix from the filename (e.g., '001' from '001_title.json')
                number_str = filename.split('_')[0]
                file_numbers.add(int(number_str))
            except (ValueError, IndexError):
                print(f"Could not parse number from filename: {filename}")
    
    if not file_numbers:
        print("No valid numbered files found.")
        return
        
    print(f"Found {len(file_numbers)} files.")
    print(f"Highest file number is: {max(file_numbers)}")

    # Create the full set of expected numbers from 1 to the maximum found
    expected_numbers = set(range(1, max(file_numbers) + 1))
    
    # Determine which numbers are missing
    missing_numbers = expected_numbers - file_numbers

    if missing_numbers:
        print("\nMissing file numbers are:")
        for num in sorted(list(missing_numbers)):
            print(num)
    else:
        print("\nNo missing file numbers found in the sequence.")

if __name__ == '__main__':
    find_missing_files()