from rapidfuzz import fuzz, process
import pandas as pd

def find_link(file_path, header_name, threshold=80):
    """
    Searches through a CSV file for a specific header and returns the corresponding link.
    
    :param file_path: Path to the CSV file
    :param header_name: The header to search for (e.g., 1201, 1202)
    :param threshold: The minimum match score (0-100) for fuzzy matching
    :return: The corresponding link if the header is found, otherwise a message
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Ensure header_name is treated as a string for comparison
    header_name = str(header_name)

    # Perform fuzzy matching to find the best match for the header name
    matches = process.extract(header_name, df['Nested Section Name'], scorer=fuzz.partial_ratio, limit=1)
    
    if matches and matches[0][1] >= threshold:  # Check if match score exceeds the threshold
        best_match = matches[0][0]  # The best-matched string
        link = df.loc[df['Nested Section Name'] == best_match, 'Link'].iloc[0]
        return link
    else:
        return f"No link found for header: {header_name}"
