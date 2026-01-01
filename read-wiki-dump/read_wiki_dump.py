#!/usr/bin/env python3
"""
Wikipedia Dump File Reader
This script lists all available Wikipedia dump files from the Wikimedia dumps site.
"""

import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin
from datetime import datetime
import sys

class WikiDumpReader:
    def __init__(self, language='en'):
        """
        Initialize the WikiDumpReader
        
        Args:
            language (str): Wikipedia language code (default: 'en' for English)
        """
        self.language = language
        self.base_url = f"https://dumps.wikimedia.org/{language}wiki/"
        
    def get_available_dumps(self):
        """
        Fetch and return a list of available dump dates
        
        Returns:
            list: List of available dump dates
        """
        try:
            print(f"Fetching available dumps for {self.language}wiki...")
            response = requests.get(self.base_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all directory links (dump dates)
            dump_dates = []
            for link in soup.find_all('a', href=True):
                href = link.get('href')
                # Match date pattern (YYYYMMDD)
                if re.match(r'^\d{8}/$', href):
                    dump_date = href.rstrip('/')
                    dump_dates.append(dump_date)
            
            # Sort dates in descending order (newest first)
            dump_dates.sort(reverse=True)
            return dump_dates
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching dump list: {e}")
            return []
        except Exception as e:
            print(f"Error parsing dump list: {e}")
            return []
    
    def get_dump_files(self, dump_date):
        """
        Get list of files for a specific dump date
        
        Args:
            dump_date (str): Dump date in YYYYMMDD format
            
        Returns:
            list: List of available dump files
        """
        try:
            dump_url = urljoin(self.base_url, f"{dump_date}/")
            print(f"Checking dump date {dump_date}...")
            
            response = requests.get(dump_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            dump_files = []
            all_links = soup.find_all('a', href=True)
            
            for link in all_links:
                href = link.get('href')
                # Look for .gz, .bz2, .xml files
                if any(href.endswith(ext) for ext in ['.gz', '.bz2', '.xml', '.7z']):
                    file_info = {
                        'filename': href,
                        'url': urljoin(dump_url, href)
                    }
                    
                    # Try to get file size from the page
                    parent_td = link.find_parent('td')
                    if parent_td:
                        next_td = parent_td.find_next_sibling('td')
                        if next_td:
                            file_info['size'] = next_td.get_text(strip=True)
                    
                    dump_files.append(file_info)
            
            print(f"Found {len(dump_files)} dump files")
            return dump_files
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching dump files: {e}")
            return []
        except Exception as e:
            print(f"Error parsing dump files: {e}")
            return []
    
    def print_available_dumps(self):
        """
        Print all available Wikipedia dumps
        """
        print(f"\n{'='*60}")
        print(f"Available Wikipedia Dumps for {self.language.upper()}WIKI")
        print(f"{'='*60}")
        
        dump_dates = self.get_available_dumps()
        
        if not dump_dates:
            print("No dump dates found or error occurred.")
            return
        
        print(f"\nFound {len(dump_dates)} available dump dates:")
        print("-" * 40)
        
        for i, dump_date in enumerate(dump_dates[:10], 1):  # Show first 10
            # Format date for display
            try:
                date_obj = datetime.strptime(dump_date, '%Y%m%d')
                formatted_date = date_obj.strftime('%B %d, %Y')
            except ValueError:
                formatted_date = dump_date
            
            print(f"{i:2d}. {dump_date} ({formatted_date})")
        
        if len(dump_dates) > 10:
            print(f"    ... and {len(dump_dates) - 10} more")
        
        # Get files for the most recent dump that actually has files
        if dump_dates:
            print(f"\n{'='*60}")
            print(f"Finding a dump with available files...")
            print(f"{'='*60}")
            
            dump_files = []
            selected_dump = None
            
            # Try the first few dumps until we find one with files
            for dump_date in dump_dates[:3]:
                dump_files = self.get_dump_files(dump_date)
                if dump_files:
                    selected_dump = dump_date
                    break
            
            if dump_files and selected_dump:
                print(f"\n{'='*60}")
                print(f"Files available in dump ({selected_dump}):")
                print(f"{'='*60}")
                
                print(f"\nFound {len(dump_files)} files in dump {selected_dump}:")
                print("-" * 80)
                
                # Group files by type
                file_types = {}
                for file_info in dump_files:
                    filename = file_info['filename']
                    
                    # Categorize files
                    if 'pages-articles' in filename:
                        category = 'Articles'
                    elif 'pages-meta-history' in filename:
                        category = 'Full History'
                    elif 'abstract' in filename:
                        category = 'Abstracts'
                    elif 'categorylinks' in filename:
                        category = 'Category Links'
                    elif 'pagelinks' in filename:
                        category = 'Page Links'
                    elif 'redirect' in filename:
                        category = 'Redirects'
                    else:
                        category = 'Other'
                    
                    if category not in file_types:
                        file_types[category] = []
                    file_types[category].append(file_info)
                
                # Print categorized files
                for category, files in file_types.items():
                    print(f"\n{category}:")
                    for file_info in files:
                        size_info = f" ({file_info.get('size', 'Unknown size')})" if 'size' in file_info else ""
                        print(f"  • {file_info['filename']}{size_info}")

def main():
    """
    Main function to demonstrate the WikiDumpReader
    """
    # Default to English Wikipedia, but can be changed
    language = 'en'
    
    # Check if language code provided as argument
    if len(sys.argv) > 1:
        language = sys.argv[1].lower()
    
    print("Wikipedia Dump File Reader")
    print("-" * 30)
    
    reader = WikiDumpReader(language=language)
    reader.print_available_dumps()
    
    print(f"\nNote: To use a different language, run: python {sys.argv[0]} <language_code>")
    print("Example: python read_wiki_dump.py fr  (for French Wikipedia)")

if __name__ == "__main__":
    main()