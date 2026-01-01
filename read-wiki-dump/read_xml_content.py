#!/usr/bin/env python3
"""
Wikipedia XML Reader
Reads and displays content from Wikipedia XML dump files
"""

import bz2
import xml.etree.ElementTree as ET
import re
import sys
import os

def read_wikipedia_xml(file_path, max_articles=0, line_limit=100000):
    """
    Read and parse Wikipedia XML dump file
    
    Args:
        file_path (str): Path to the bz2 compressed XML file
        max_articles (int): Maximum number of articles to show
    """
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    print(f"Reading Wikipedia XML file: {file_path}")
    print("=" * 80)

    list_redirections: list[str] = []
    
    try:
        with bz2.open(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
            article_count = 0
            current_article = {}
            in_text = False
            text_content = ""
            
            for line_num, line in enumerate(f):
                if line_limit > 0 and line_num > line_limit:  # Limit to first 100k lines for demo
                    break
                    
                line = line.strip()
                
                # Look for page start
                if '<page>' in line:
                    current_article = {}
                    in_text = False
                    text_content = ""
                
                # Extract title
                elif '<title>' in line:
                    title_match = re.search(r'<title>(.*?)</title>', line)
                    if title_match:
                        current_article['title'] = title_match.group(1)
                
                # Extract page ID
                elif '<id>' in line and 'title' in current_article:
                    id_match = re.search(r'<id>(\d+)</id>', line)
                    if id_match and 'id' not in current_article:
                        current_article['id'] = id_match.group(1)
                
                # Extract namespace
                elif '<ns>' in line:
                    ns_match = re.search(r'<ns>(\d+)</ns>', line)
                    if ns_match:
                        current_article['namespace'] = ns_match.group(1)
                
                # Start of text content
                elif '<text' in line:
                    in_text = True
                    # Check if text content is on the same line
                    if '>' in line:
                        start_idx = line.find('>') + 1
                        text_content = line[start_idx:]
                        if '</text>' in text_content:
                            text_content = text_content[:text_content.find('</text>')]
                            in_text = False
                
                # Continue reading text content
                elif in_text:
                    if '</text>' in line:
                        text_content += line[:line.find('</text>')]
                        in_text = False
                    else:
                        text_content += line + '\n'
                
                # End of page
                elif '</page>' in line and current_article:
                    if 'title' in current_article and text_content:
                        current_article['text'] = text_content
                        
                        # Only show articles in main namespace (ns=0)
                        if current_article.get('namespace') == '0':
                            print_article(current_article, article_count + 1, list_redirections)
                            article_count += 1
                            
                            if max_articles > 0 and article_count >= max_articles:
                                break
                    
                    current_article = {}
                    text_content = ""
    
    except Exception as e:
        print(f"Error reading file: {e}")
    
    print(f"\nShowed {article_count} articles from the Wikipedia dump.")

def print_article(article, number, list_redirections: list[str]=None):
    """Print article information in a readable format"""
    title: str = article.get('title', 'Unknown')
    page_id: str = article.get('id', 'Unknown')
    
    print(f"ARTICLE #{number}, Title: {title}, Page ID: {page_id}")

    text = article.get('text', '')
    
    if text:
        skip_links: bool = False

        # Check for redirection
        if text.startswith('#REDIRECT') or text.startswith('#redirect'):
            skip_links = True

            print("This article is a redirection.")
            if list_redirections is not None:
                list_redirections.append(title)

        if not skip_links:
            # Extract links from the article
            wiki_links = extract_wiki_links(text)

            print(f"Extracted {len(wiki_links)} wiki links: {', '.join(wiki_links[:10])}" + (f", ..." if len(wiki_links) > 10 else ""))

    return
    print(f"Namespace: {article.get('namespace', 'Unknown')}")
    print(f"\nContent Preview (first 1000 characters):")
    print("-" * 80)
    
    text = article.get('text', '')
    # Clean up the text a bit
    text = re.sub(r'\{\{[^}]+\}\}', '[TEMPLATE]', text)  # Replace templates
    text = re.sub(r'\[\[([^|\]]+)(\|[^\]]+)?\]\]', r'\1', text)  # Simplify links
    text = re.sub(r'\n+', '\n', text)  # Reduce multiple newlines
    
    print(text[:1000] + ('...' if len(text) > 1000 else ''))
    
    # Show some statistics
    word_count = len(text.split())
    char_count = len(text)
    link_count = len(re.findall(r'\[\[.*?\]\]', article.get('text', '')))
    
    print("-" * 80)
    print(f"Statistics: {word_count} words, {char_count} characters, ~{link_count} links")

def extract_wiki_links(text):
    """Extract wiki links from wikitext"""
    # Pattern for wiki links: [[Target Page|Display Text]] or [[Target Page]]
    import re
    link_pattern = r'\[\[([^|\]]+)(?:\|[^\]]*)?\]\]'
    links = re.findall(link_pattern, text)
    
    # Clean up links (remove namespace prefixes for files, categories, etc.)
    clean_links = []
    for link in links:
        link = link.strip()
        # Skip files, categories, templates, etc. - focus on article links
        if not any(link.startswith(prefix) for prefix in ['File:', 'Category:', 'Template:', 'Help:', 'Wikipedia:', 'User:']):
            clean_links.append(link)
    
    return clean_links

def main():
    """Main function"""
    # List of possible XML files to try
    download_dir = r"c:\Users\HaimL1\Networks\downloads"

    download_dir = r"C:\Users\HaimL1\Downloads"
    
    # Look for XML files
    xml_files = []
    if os.path.exists(download_dir):
        for filename in os.listdir(download_dir):
            if filename.endswith('.xml.bz2') and 'articles' in filename:
                xml_files.append(os.path.join(download_dir, filename))
    
    if not xml_files:
        print("No Wikipedia article XML files found in downloads directory.")
        print("Looking for files like: enwiki-20251220-pages-articles*.xml.bz2")
        
        # Try the root directory
        root_dir = r"c:\Users\HaimL1\Networks"
        for filename in os.listdir(root_dir):
            if filename.endswith('.xml.bz2') and 'articles' in filename:
                xml_files.append(os.path.join(root_dir, filename))
    
    if not xml_files:
        print("No Wikipedia article XML files found.")
        print("You may need to download them first using the wiki dump reader.")
        return
    
    # Use the first found file
    xml_file = xml_files[0]
    print(f"Found XML file: {xml_file}")
    
    # Read and display content
    read_wikipedia_xml(xml_file, max_articles=0, line_limit=0)

if __name__ == "__main__":
    main()