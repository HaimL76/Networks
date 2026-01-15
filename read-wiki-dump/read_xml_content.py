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
        self.id_to_title_dict: dict[int, str] = {}
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

    def read_sql_dumps(self):        
        self.read_title_to_id_dictionary()

        #self.read_articles_with_links()

        file_path = "C:\\Users\\HaimL1\\Downloads\\hewiki-20251220-category.sql.gz"

        self.read_sql_dump(file_path)

        file_path = "C:\\Users\\HaimL1\\Downloads\\hewiki-20251220-categorylinks.sql.gz"

        self.read_sql_dump(file_path)

    def read_sql_dump(self, file_path: str):
        import gzip
        
        if not os.path.exists(file_path):
            print(f"SQL dump file not found: {file_path}\n")
            return
            
        print(f"Reading SQL dump file: {file_path}")
        
        try:
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                line_count = 0
                insert_line_buffer = ""
                in_insert_statement = False

                for line in f:
                    line_count += 1
                    line = line.strip()

                    # Skip comments and empty lines
                    if not line or line.startswith('--') or line.startswith('/*'):
                        continue

                    # Check if this is the start of an INSERT statement
                    if line.upper().startswith('INSERT INTO'):
                        in_insert_statement = True
                        insert_line_buffer = line
                        
                        # If the entire INSERT is on one line
                        if ';' in line:
                            self._process_insert_line(insert_line_buffer, file_path)
                            in_insert_statement = False
                            insert_line_buffer = ""
                        continue

                    # Continue building the INSERT statement if we're in one
                    if in_insert_statement:
                        insert_line_buffer += " " + line
                        
                        # Check if INSERT statement is complete
                        if ';' in line:
                            self._process_insert_line(insert_line_buffer, file_path)
                            in_insert_statement = False
                            insert_line_buffer = ""
                        continue

                    # Print progress every 10000 lines
                    if line_count % 10000 == 0:
                        print(f"Processed {line_count} lines...")
                        
                print(f"Finished processing {line_count} lines from SQL dump")
                
        except Exception as e:
            print(f"Error reading SQL dump file: {e}")

    def _process_insert_line(self, insert_line: str, file_path: str):
        """Process a complete INSERT statement and extract tuples"""
        try:
            # Determine table type based on file name for different parsing strategies
            is_category_table = 'category.sql' in file_path and 'categorylinks' not in file_path
            is_categorylinks_table = 'categorylinks.sql' in file_path
            
            # Find the VALUES part
            values_start = insert_line.upper().find('VALUES')
            if values_start == -1:
                return

            values_part = insert_line[values_start + 6:].strip()
            
            # Remove trailing semicolon
            if values_part.endswith(';'):
                values_part = values_part[:-1]

            # Extract tuples from the VALUES part
            tuples = self._extract_tuples_from_values(values_part)
            
            for tuple_data in tuples:
                values = self._parse_tuple_values(tuple_data)
                
                if is_category_table:
                    self._process_category_tuple(values)
                elif is_categorylinks_table:
                    self._process_categorylinks_tuple(values)
                else:
                    print(f"Unknown table type, raw values: {values}")
                
        except Exception as e:
            print(f"Error processing INSERT line: {e}")

    def _process_category_tuple(self, values):
        """Process category table tuple (cat_id, cat_title, cat_pages, cat_subcats, cat_files)"""
        if len(values) >= 2:
            cat_id = values[0] if isinstance(values[0], int) else None
            cat_title = values[1] if len(values) > 1 else None
            cat_pages = values[2] if len(values) > 2 and isinstance(values[2], int) else 0
            cat_subcats = values[3] if len(values) > 3 and isinstance(values[3], int) else 0
            cat_files = values[4] if len(values) > 4 and isinstance(values[4], int) else 0
            
            if cat_title:
                cat_title = str(cat_title).strip().strip("'\"")
            
            print(f"Category - ID: {cat_id}, Title: '{cat_title}', Pages: {cat_pages}, Subcats: {cat_subcats}, Files: {cat_files}")

    def _process_categorylinks_tuple(self, values):
        """Process categorylinks table tuple (cl_from, cl_sortkey, cl_timestamp, cl_sortkey_prefix, cl_type, cl_collation_id, cl_target_id)"""
        if len(values) >= 2:
            cl_from = values[0] if isinstance(values[0], int) else None  # Page ID
            cl_sortkey = values[1] if len(values) > 1 else None  # Sort key
            cl_timestamp = values[2] if len(values) > 2 else None  # Timestamp
            cl_sortkey_prefix = values[3] if len(values) > 3 else None  # Sort key prefix
            cl_type = values[4] if len(values) > 4 else 'page'  # page, subcat, file
            cl_collation_id = values[5] if len(values) > 5 and isinstance(values[5], int) else 0  # Collation ID
            cl_target_id = values[6] if len(values) > 6 and isinstance(values[6], int) else None  # Category ID
            
            # Clean up string values
            if cl_sortkey:
                cl_sortkey = str(cl_sortkey).strip().strip("'\"")
            if cl_sortkey_prefix:
                cl_sortkey_prefix = str(cl_sortkey_prefix).strip().strip("'\"")
            if cl_type:
                cl_type = str(cl_type).strip().strip("'\"")

            page_title: str = None

            if cl_from in self.id_to_title_dict:
                page_title = self.id_to_title_dict[cl_from]
            
            print(f"CategoryLink - From Page: {cl_from} ({page_title}), Target Category ID: {cl_target_id}, Type: {cl_type}, Sort Key: '{cl_sortkey}'")

    def _extract_tuples_from_values(self, values_string: str):
        """Extract individual tuples from VALUES string"""
        tuples = []
        current_tuple = ""
        parentheses_level = 0
        in_quotes = False
        quote_char = None
        escape_next = False
        
        i = 0
        while i < len(values_string):
            char = values_string[i]
            
            if escape_next:
                current_tuple += char
                escape_next = False
            elif char == '\\':
                escape_next = True
                current_tuple += char
            elif char in ('"', "'") and not escape_next:
                if not in_quotes:
                    in_quotes = True
                    quote_char = char
                elif char == quote_char:
                    in_quotes = False
                    quote_char = None
                current_tuple += char
            elif char == '(' and not in_quotes:
                if parentheses_level == 0:
                    current_tuple = ""  # Start new tuple
                else:
                    current_tuple += char
                parentheses_level += 1
            elif char == ')' and not in_quotes:
                parentheses_level -= 1
                if parentheses_level == 0:
                    # Complete tuple found
                    if current_tuple.strip():
                        tuples.append(current_tuple.strip())
                    current_tuple = ""
                else:
                    current_tuple += char
            elif parentheses_level > 0:  # Inside a tuple
                current_tuple += char
            # Skip commas and whitespace between tuples
            
            i += 1
            
        return tuples

    def _parse_tuple_values(self, buffer: str):
        """
        Parse comma-separated values from a tuple buffer.
        Handles quoted strings, escaped characters, and numeric values.
        """
        values = []
        current_value = ""
        in_quotes = False
        quote_char = None
        escape_next = False
        
        i = 0
        while i < len(buffer):
            char = buffer[i]
            
            if escape_next:
                current_value += char
                escape_next = False
            elif char == '\\':
                escape_next = True
                current_value += char
            elif char in ('"', "'") and not escape_next:
                if not in_quotes:
                    in_quotes = True
                    quote_char = char
                elif char == quote_char:
                    in_quotes = False
                    quote_char = None
                else:
                    current_value += char
            elif char == ',' and not in_quotes:
                values.append(self._convert_value(current_value.strip()))
                current_value = ""
            else:
                current_value += char
            
            i += 1
        
        # Add the last value
        if current_value.strip():
            values.append(self._convert_value(current_value.strip()))
        
        return values

    def _convert_value(self, value_str: str):
        """Convert a string value to appropriate type (int or cleaned string)."""
        if not value_str:
            return ""
        
        # Remove surrounding quotes
        if ((value_str.startswith('"') and value_str.endswith('"')) or 
            (value_str.startswith("'") and value_str.endswith("'"))):
            value_str = value_str[1:-1]
        
        # Try to convert to integer
        if value_str.isdigit() or (value_str.startswith('-') and value_str[1:].isdigit()):
            return int(value_str)
        
        # Handle escaped characters in strings
        value_str = value_str.replace("\\'", "'").replace('\\"', '"').replace('\\\\', '\\')
        
        return value_str
        
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
        
        with open("edges.csv", 'w') as edges_file:
            edges_file.write("Source,Target\n")

            for article_id, link_ids in self.article_links_buffer.items():
                for link_id in link_ids:
                    edges_file.write(f"{article_id},{link_id}\n")

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

                            page_id_str: str = None
                            str_link_ids: str = None
                            pade_id: int = None

                            if isinstance(arr, list) and len(arr) > 1:
                                page_id_str = arr[0].strip()

                            if page_id_str and page_id_str.isnumeric():
                                page_id = int(page_id_str)

                            if page_id is not None:
                                str_link_ids = arr[1]

                            if str_link_ids:
                                str_link_ids = str_link_ids.strip()

                            if str_link_ids:
                                list_link_ids_str: list[str] = str_link_ids.split(',')

                                if isinstance(list_link_ids_str, list) and len(list_link_ids_str) > 0:
                                    set_link_ids: set[int] = set()

                                    for link_id_str in list_link_ids_str:
                                        link_id_str = link_id_str.strip()

                                        if link_id_str and link_id_str.isnumeric():
                                            link_id: int = int(link_id_str)

                                            set_link_ids.add(link_id)

                                    if len(set_link_ids) > 0:
                                        self.article_links_found[page_id] = set_link_ids


    def read_title_to_id_dictionary(self):
        """Read title to page ID dictionary from text files"""
        if not os.path.exists(self.title_to_id_directory):
            print(f"Title to ID directory does not exist: {self.title_to_id_directory}")
            return

        self.title_to_id_dict = {}
        self.id_to_title_dict = {}

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

                                str_page_id: str = line[index + 1:]

                                if str_page_id:
                                    str_page_id = str_page_id.strip()
                                
                                if str_page_id and str_page_id.isnumeric():
                                    page_id: int = int(str_page_id)
                                    self.title_to_id_dict[title] = page_id
                                    self.id_to_title_dict[page_id] = title
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
                            skip_link: bool = False

                            if link.startswith('קטגוריה: '):
                                skip_link = True

                            
                            if not skip_link:
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
    reader = WikiDumpReader()

    reader.read_sql_dumps()

    return

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

    reader.save_article_links_dictionary_to_file()

if __name__ == "__main__":
    main()