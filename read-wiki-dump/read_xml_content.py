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

class WikiDumpReader:
    def __init__(self):
        self.last_title_to_id_count: int = 0
        self.title_to_id_dict: dict[str, int] = {}
        self.title_to_id_buffer: dict[str, int] = {}
        self.article_links_buffer: dict[int, set[int]] = {}
        self.article_links_found: dict[int, int] = {}
        self.list_redirections: list[str] = []
        self.title_to_id_directory: str = "title-id"
        self.article_links_directory: str = "article-links"
        self.list_title_to_id_files: list[str] = []
        self.list_article_links_files: list[str] = []
        self.subject_counter: int = 0
        self.last_updated_subject_counter = 0
        
    def read_wikipedia_xml(self, file_path, max_articles=0, line_limit=100000):
        """
        Read and parse Wikipedia XML dump file
        
        Args:
            file_path (str): Path to the bz2 compressed XML file
            max_articles (int): Maximum number of articles to show
        """
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return
        
        self.read_title_to_id_dictionary()

        self.read_articles_with_links()
        
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
                                self.print_article(current_article, article_count + 1)
                                article_count += 1
                                
                                if max_articles > 0 and article_count >= max_articles:
                                    break
                        
                        current_article = {}
                        text_content = ""
        
        except Exception as e:
            print(f"Error reading file: {e}")
        
        print(f"\nShowed {article_count} articles from the Wikipedia dump.")


    def read_articles_with_links(self):
        """Read title to page ID dictionary from text files"""
        if not os.path.exists(self.article_links_directory):
            print(f"Article links directory does not exist: {self.article_links_directory}")
            return

        list_files: list[str] = os.listdir(self.article_links_directory)

        if list_files:
            for filename in list_files:
                if filename.endswith('.txt'):
                    file_path = os.path.join(self.article_links_directory, filename)

                    if self.list_article_links_files is None:
                        self.list_article_links_files = []

                    self.list_article_links_files.append(file_path)

                    self.list_article_links_files = sorted(self.list_article_links_files, 
                                                         key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

        if len(self.list_article_links_files) > 0:
            for file_path in self.list_article_links_files:
                with open(file_path, 'r', encoding='utf-8') as fr:
                    for line in fr:
                        if line:
                            line = line.strip()
                        
                        if line:
                            arr: list[str] = line.split(':')
                                    
                            if isinstance(arr, list) and len(arr) > 1:
                                page_id_str: str = arr[0].strip()

                                if page_id_str and page_id_str.isnumeric():
                                    page_id: int = int(page_id_str)
                                
                                self.article_links_found[page_id] = len(arr[1])


    def read_title_to_id_dictionary(self):
        """Read title to page ID dictionary from text files"""
        if not os.path.exists(self.title_to_id_directory):
            print(f"Title to ID directory does not exist: {self.title_to_id_directory}")
            return

        self.title_to_id_dict = {}

        list_files: list[str] = os.listdir(self.title_to_id_directory)

        if list_files:
            for filename in list_files:
                if filename.endswith('.txt'):
                    file_path = os.path.join(self.title_to_id_directory, filename)

                    if self.list_title_to_id_files is None:
                        self.list_title_to_id_files = []

                    self.list_title_to_id_files.append(file_path)

                    self.list_title_to_id_files = sorted(self.list_title_to_id_files, 
                                                         key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

        if len(self.list_title_to_id_files) > 0:
            for file_path in self.list_title_to_id_files:
                with open(file_path, 'r', encoding='utf-8') as fr:
                    for line in fr:
                        if line:
                            line = line.strip()
                        
                        if line:
                            index: int = line.rfind(',')

                            if index > 0:
                                title: str = line[:index]

                                if title:
                                    title = title.strip()

                                page_id: str = line[index + 1:]

                                if page_id:
                                    page_id = page_id.strip()
                                
                                if page_id and page_id.isnumeric():
                                    self.title_to_id_dict[title] = int(page_id)

                                    if "," in title:
                                        print(f"{file_path}, Loaded title to ID: {title} -> {page_id}")

    def save_title_to_id_dictionary_to_file(self):
        """Save title to page ID dictionary to a text file"""
        if not os.path.exists(self.title_to_id_directory):
            os.makedirs(self.title_to_id_directory)

        if not self.list_title_to_id_files:
            file_path = os.path.join(self.title_to_id_directory, "1.txt")
            
            self.list_title_to_id_files = [file_path]

        file_path: str = self.list_title_to_id_files[-1]
                
        try:
            with open(file_path, 'a', encoding='utf-8') as f:
                for title, page_id in self.title_to_id_buffer.items():
                    f.write(f"{title},{page_id}\n")
            print(f"Saved dictionary to {file_path}")

            file_size: int = os.path.getsize(file_path)

            if file_size >= 10 * 1024 * 1024:  # 10 MB
                file_name = os.path.basename(file_path)
                file_index_str = os.path.splitext(file_name)[0] 
                new_file_index: int = int(file_index_str) + 1
                new_file_path: str = os.path.join(self.title_to_id_directory, f"{new_file_index}.txt")
                self.list_title_to_id_files.append(new_file_path)
        except Exception as e:
            print(f"Error saving dictionary to file: {e}")

    def save_article_links_dictionary_to_file(self):
        """Save title to page ID dictionary to a text file"""
        if not os.path.exists(self.article_links_directory):
            os.makedirs(self.article_links_directory)

        if not self.list_article_links_files:
            file_path = os.path.join(self.article_links_directory, "1.txt")
            
            self.list_article_links_files = [file_path]

        file_path: str = self.list_article_links_files[-1]
                
        try:
            with open(file_path, 'a', encoding='utf-8') as f:
                for article, links in self.article_links_buffer.items():
                    str_link_ids: str = ",".join(str(id) for id in links)
                    f.write(f"{article}:{str_link_ids}\n")
            print(f"Saved dictionary to {file_path}")

            file_size: int = os.path.getsize(file_path)

            if file_size >= 10 * 1024 * 1024:  # 10 MB
                file_name = os.path.basename(file_path)
                file_index_str = os.path.splitext(file_name)[0] 
                new_file_index: int = int(file_index_str) + 1
                new_file_path: str = os.path.join(self.article_links_directory, f"{new_file_index}.txt")
                self.list_article_links_files.append(new_file_path)
        except Exception as e:
            print(f"Error saving dictionary to file: {e}")

    def print_article(self, article, number):
        """Print article information in a readable format"""
        title: str = article.get('title', 'Unknown')

        if title:
            title = title.strip()

        int_page_id: int = None

        page_id: str = article.get('id', 'Unknown')

        if page_id:
            page_id = page_id.strip()

        if page_id and page_id.isnumeric():
            int_page_id = int(page_id)

        if int_page_id and title not in self.title_to_id_dict:
            self.title_to_id_buffer[title] = int_page_id

        current_count: int = len(self.title_to_id_dict)
        diff_count: int = current_count - self.last_title_to_id_count

        self.subject_counter += 1

        if self.subject_counter - self.last_updated_subject_counter >= 100:
            print(f"Processed {self.subject_counter} subjects so far...")
            self.last_updated_subject_counter = self.subject_counter

        if len(self.title_to_id_buffer) >= 1000:
            print(f"Title to ID dictionary size: {current_count} entries (added {diff_count})")
            self.save_title_to_id_dictionary_to_file()
            
            self.title_to_id_dict.update(self.title_to_id_buffer)
            self.title_to_id_buffer.clear()

        return
        
        #print(f"ARTICLE #{number}, Title: {title}, Page ID: {page_id}")

        text = article.get('text', '')
        
        if text and int_page_id is not None and int_page_id not in self.article_links_found:
            skip_links: bool = False

            # Check for redirection
            if text.startswith('#REDIRECT') or text.startswith('#redirect'):
                #skip_links = True

                #print("This article is a redirection.")
                if self.list_redirections is not None:
                    self.list_redirections.append(title)

            if not skip_links:
                # Extract links from the article
                wiki_links = self.extract_wiki_links(text)

                if wiki_links:
                    list_link_ids: set[int] = set()

                    for link in wiki_links:
                        link_id: int = None

                        if link:
                            link = link.strip()

                        if link:
                            #print(f"Link: {link}")

                            if link in self.title_to_id_dict:
                                link_id = self.title_to_id_dict[link]
                                
                                list_link_ids.add(link_id)

                                #print(f"Link found: {link} -> {link_id}")
                            else:
                                print(f"Link title not found in dictionary: {link}")

                    if isinstance(list_link_ids, set) and len(list_link_ids) > 0:
                        self.article_links_buffer[int_page_id] = sorted(list_link_ids)

        if len(self.article_links_buffer) >= 1000:
            self.save_article_links_dictionary_to_file()
            print(f"Article links buffer size: {len(self.article_links_buffer)} entries")
            # Here you would typically save the article links buffer to a file or database
            self.article_links_buffer.clear()

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

    def extract_wiki_links(self, text):
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

    reader: WikiDumpReader = WikiDumpReader()
    
    # Read and display content
    reader.read_wikipedia_xml(xml_file, max_articles=0, line_limit=0)

    reader.save_title_to_id_dictionary_to_file()

if __name__ == "__main__":
    main()