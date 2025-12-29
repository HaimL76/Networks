import bz2
import xml.etree.ElementTree as ET
import re
import requests
import json
from urllib.parse import urljoin

def read_wikipedia_dump(file_path):
    """
    Read and parse a Wikipedia dump file in bz2 format.
    
    Args:
        file_path: Path to the .bz2 Wikipedia dump file
    
    Yields:
        dict: Dictionary containing page information (title, id, text, etc.)
    """
    # Open the bz2 compressed file
    with bz2.open(file_path, 'rt', encoding='utf-8') as file:
        # Parse XML incrementally to handle large files
        context = ET.iterparse(file, events=('start', 'end'))
        context = iter(context)
        event, root = next(context)
        
        current_page = {}
        current_element = None
        
        for event, elem in context:
            # Remove namespace from tag name for easier handling
            tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
            
            if event == 'start':
                current_element = tag
            elif event == 'end':
                if tag == 'page':
                    # Yield the completed page and reset for next page
                    if current_page:
                        yield current_page
                        current_page = {}
                elif tag == 'title':
                    current_page['title'] = elem.text
                elif tag == 'id' and current_element != 'revision':
                    # Page ID (not revision ID)
                    current_page['id'] = elem.text
                elif tag == 'text':
                    current_page['text'] = elem.text
                elif tag == 'ns':
                    current_page['namespace'] = elem.text
                
                # Clear the element to save memory
                elem.clear()
                root.clear()

def clean_wikitext(text):
    """
    Clean Wikipedia markup from text content.
    
    Args:
        text: Raw Wikipedia text with markup
    
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
    
    # Remove wiki markup
    text = re.sub(r'\{\{[^}]*\}\}', '', text)  # Remove templates
    text = re.sub(r'\[\[([^|\]]*\|)?([^\]]*)\]\]', r'\2', text)  # Remove links, keep text
    text = re.sub(r'\[http[^\s\]]* ([^\]]*)\]', r'\1', text)  # Remove external links
    text = re.sub(r'==+([^=]*)==+', r'\1', text)  # Remove section headers
    text = re.sub(r"'''([^']*)'''", r'\1', text)  # Remove bold
    text = re.sub(r"''([^']*)''", r'\1', text)  # Remove italic
    text = re.sub(r'<[^>]*>', '', text)  # Remove HTML tags
    
    # Clean up extra whitespace
    text = re.sub(r'\n+', '\n', text)
    text = text.strip()
    
    return text

def get_dump_files(status_url="https://dumps.wikimedia.org/enwiki/20251220/dumpstatus.json"):
    """
    Fetch and parse the Wikimedia dump status JSON to create a list of available dump files.
    
    Args:
        status_url: URL to the dumpstatus.json file
    
    Returns:
        list: List of dictionaries containing dump file information
    """
    try:
        # Fetch the JSON data
        response = requests.get(status_url)
        response.raise_for_status()
        
        dump_status = response.json()
        
        # Extract the base URL for files
        base_url = "https://dumps.wikimedia.org/enwiki/20251220/"
        
        dump_files = []
        
        # Parse the jobs section which contains file information
        jobs = dump_status.get('jobs', {})
        
        for job_name, job_data in jobs.items():
            if 'files' in job_data:
                for filename, file_info in job_data['files'].items():
                    file_entry = {
                        'filename': filename,
                        'job': job_name,
                        'url': urljoin(base_url, filename),
                        'size': file_info.get('size', 'Unknown'),
                        'md5': file_info.get('md5', ''),
                        'status': file_info.get('status', 'unknown')
                    }
                    dump_files.append(file_entry)
        
        return dump_files
    
    except requests.RequestException as e:
        print(f"Error fetching dump status: {e}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return []

def print_dump_files_summary(dump_files):
    """
    Print a summary of available dump files.
    
    Args:
        dump_files: List of dump file dictionaries
    """
    print(f"\nFound {len(dump_files)} dump files:")
    print("-" * 80)
    
    # Group files by job type
    jobs = {}
    for file_info in dump_files:
        job = file_info['job']
        if job not in jobs:
            jobs[job] = []
        jobs[job].append(file_info)
    
    for job_name, files in jobs.items():
        print(f"\n{job_name.upper()} ({len(files)} files):")
        for file_info in files[:3]:  # Show first 3 files of each type
            size_mb = int(file_info['size']) / (1024 * 1024) if file_info['size'] != 'Unknown' and file_info['size'] else 0
            print(f"  - {file_info['filename']} ({size_mb:.1f} MB)")
        if len(files) > 3:
            print(f"  ... and {len(files) - 3} more files")

def filter_dump_files(dump_files, job_type=None, file_extension=None):
    """
    Filter dump files by job type and/or file extension.
    
    Args:
        dump_files: List of dump file dictionaries
        job_type: Filter by job type (e.g., 'articlesmultistreamdump')
        file_extension: Filter by file extension (e.g., '.bz2')
    
    Returns:
        list: Filtered list of dump files
    """
    filtered = dump_files
    
    if job_type:
        filtered = [f for f in filtered if f['job'] == job_type]
    
    if file_extension:
        filtered = [f for f in filtered if f['filename'].endswith(file_extension)]
    
    return filtered

def main():
    print("Fetching available Wikipedia dump files...")
    
    # Get list of available dump files
    dump_files = get_dump_files()
    
    # Print all files found in the wiki JSON URL
    if dump_files:
        print(f"\n{'='*100}")
        print("ALL FILES FOUND IN WIKI JSON URL:")
        print(f"{'='*100}")
        for i, file_info in enumerate(dump_files, 1):
            filename: str = file_info['filename']

            if not filename.startswith("enwiki"):
                continue

            size_mb = int(file_info['size']) / (1024 * 1024) if file_info['size'] != 'Unknown' and file_info['size'] else 0
            print(f"{i:3d}. File: {file_info['filename']}")
            continue
            print(f"     Job: {file_info['job']}")
            print(f"     URL: {file_info['url']}")
            print(f"     Size: {size_mb:.2f} MB")
            print(f"     Status: {file_info['status']}")
            if file_info['md5']:
                print(f"     MD5: {file_info['md5']}")
            print("-" * 100)
        print(f"TOTAL: {len(dump_files)} files found\n")
    
    if dump_files:
        # Print summary of all files
        print_dump_files_summary(dump_files)
        
        # Show some specific examples
        print("\n" + "="*80)
        print("ARTICLE DUMP FILES (.bz2 format):")
        article_files = filter_dump_files(dump_files, 
                                        job_type='articlesmultistreamdump', 
                                        file_extension='.bz2')
        for i, file_info in enumerate(article_files[:5]):
            size_mb = int(file_info['size']) / (1024 * 1024) if file_info['size'] != 'Unknown' and file_info['size'] else 0
            print(f"{i+1}. {file_info['filename']} ({size_mb:.1f} MB)")
            print(f"   URL: {file_info['url']}")
            print(f"   Status: {file_info['status']}")
            if file_info['md5']:
                print(f"   MD5: {file_info['md5']}")
            print()
        
        if len(article_files) > 5:
            print(f"... and {len(article_files) - 5} more article dump files")
        
        print("\n" + "="*80)
        print("You can now download and process these files using the existing read_wikipedia_dump() function.")
    
    # Example of processing a local dump file (commented out)
    # Uncomment the code below to process a downloaded dump file:
    #
    # file_path = "C:\\Users\\HaimL1\\Downloads\\enwiki-20251220-pages-articles-multistream1.xml-p1p41242.bz2"
    # 
    # page_count = 0
    # for page in read_wikipedia_dump(file_path):
    #     page_count += 1
    #     
    #     # Only process articles (namespace 0)
    #     if page.get('namespace') == '0':
    #         title = page.get('title', 'No title')
    #         page_id = page.get('id', 'No ID')
    #         text = page.get('text', '')
    #         
    #         # Clean the wikitext
    #         clean_text = clean_wikitext(text)
    #         
    #         print(f"Page {page_count}: {title} (ID: {page_id})")
    #         print(f"Text preview: {clean_text[:200]}...")
    #         print("-" * 50)
    #         
    #         # Limit output for demonstration
    #         if page_count >= 10:
    #             break
    # 
    # print(f"Processed {page_count} pages total.")

if __name__ == "__main__":
    main()