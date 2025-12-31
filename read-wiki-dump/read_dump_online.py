import bz2
import xml.etree.ElementTree as ET
import re
import requests
import json
import io
from urllib.parse import urljoin
from typing import Iterator, Dict, Any, Optional

class OnlineWikipediaDumpReader:
    """
    A class to read and process Wikipedia dumps directly from online sources
    without downloading the entire file first.
    """
    
    def __init__(self, base_url: str = "https://dumps.wikimedia.org/enwiki/20251220/"):
        self.base_url = base_url
        self.status_url = base_url.rstrip('/') + "/dumpstatus.json"
    
    def get_dump_files(self) -> list:
        """
        Fetch and parse the Wikimedia dump status JSON to get available dump files.
        
        Returns:
            list: List of dictionaries containing dump file information
        """
        try:
            response = requests.get(self.status_url)
            response.raise_for_status()
            
            dump_status = response.json()
            dump_files = []
            
            jobs = dump_status.get('jobs', {})
            
            for job_name, job_data in jobs.items():
                if 'files' in job_data:
                    for filename, file_info in job_data['files'].items():
                        file_entry = {
                            'filename': filename,
                            'job': job_name,
                            'url': urljoin(self.base_url, filename),
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
    
    def stream_dump_from_url(self, url: str, chunk_size: int = 65536) -> Iterator[Dict[str, Any]]:
        """
        Stream and parse a Wikipedia dump file directly from a URL.
        
        Args:
            url: URL to the .bz2 Wikipedia dump file
            chunk_size: Size of chunks to read at a time
        
        Yields:
            dict: Dictionary containing page information (title, id, text, etc.)
        """
        print(f"Streaming dump from: {url}")
        
        try:
            # Stream the file from URL
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            
            # Create decompressor and XML buffer
            decompressor = bz2.BZ2Decompressor()
            xml_buffer = ""
            pages_found = 0
            
            # Track if we're inside a page element
            inside_page = False
            current_page = {}
            current_tag = None
            current_text = ""
            
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    downloaded_size += len(chunk)
                    
                    # Show progress every 1MB
                    if downloaded_size % (1024 * 1024) == 0 or total_size > 0:
                        if total_size > 0:
                            progress = (downloaded_size / total_size) * 100
                            print(f"\rProgress: {progress:.1f}% - Pages found: {pages_found}", end='', flush=True)
                        else:
                            print(f"\rDownloaded: {downloaded_size:,} bytes - Pages found: {pages_found}", end='', flush=True)
                    
                    # Decompress chunk
                    try:
                        decompressed = decompressor.decompress(chunk)
                        if decompressed:
                            xml_buffer += decompressed.decode('utf-8', errors='replace')
                            
                            # Process the buffer line by line to extract complete pages
                            while True:
                                # Look for complete page elements
                                page_start = xml_buffer.find('<page>')
                                page_end = xml_buffer.find('</page>')
                                
                                if page_start != -1 and page_end != -1 and page_end > page_start:
                                    # Extract complete page XML
                                    page_xml = xml_buffer[page_start:page_end + 7]  # +7 for '</page>'
                                    
                                    # Remove this page from buffer
                                    xml_buffer = xml_buffer[page_end + 7:]
                                    
                                    # Parse this page
                                    page_data = self._parse_single_page(page_xml)
                                    if page_data:
                                        pages_found += 1
                                        yield page_data
                                    
                                else:
                                    # No complete page found, break and wait for more data
                                    break
                                
                                # Limit buffer size to prevent memory issues
                                if len(xml_buffer) > 10 * 1024 * 1024:  # 10MB limit
                                    # Keep only the last 1MB of buffer
                                    xml_buffer = xml_buffer[-1024*1024:]
                    
                    except EOFError:
                        # End of compressed data
                        break
                    except Exception as e:
                        print(f"\nDecompression error: {e}")
                        continue
            
            # Process any remaining complete pages in buffer
            while True:
                page_start = xml_buffer.find('<page>')
                page_end = xml_buffer.find('</page>')
                
                if page_start != -1 and page_end != -1 and page_end > page_start:
                    page_xml = xml_buffer[page_start:page_end + 7]
                    xml_buffer = xml_buffer[page_end + 7:]
                    
                    page_data = self._parse_single_page(page_xml)
                    if page_data:
                        pages_found += 1
                        yield page_data
                else:
                    break
            
            print(f"\nStream processing completed. Total pages found: {pages_found}")
            
        except requests.RequestException as e:
            print(f"\nError streaming file: {e}")
        except Exception as e:
            print(f"\nUnexpected error: {e}")
    
    def _parse_single_page(self, page_xml: str) -> Optional[Dict[str, Any]]:
        """
        Parse a single page XML string.
        
        Args:
            page_xml: XML string containing a single page
        
        Returns:
            dict: Dictionary containing page information, or None if parsing failed
        """
        try:
            # Wrap in a root element to make it valid XML
            wrapped_xml = f"<root>{page_xml}</root>"
            root = ET.fromstring(wrapped_xml)
            
            page_elem = root.find('page')
            if page_elem is None:
                return None
            
            # Extract page data
            page_data = {}
            
            # Get title
            title_elem = page_elem.find('title')
            if title_elem is not None and title_elem.text:
                page_data['title'] = title_elem.text
            
            # Get page ID (not revision ID)
            id_elem = page_elem.find('id')
            if id_elem is not None and id_elem.text:
                page_data['id'] = id_elem.text
            
            # Get namespace
            ns_elem = page_elem.find('ns')
            if ns_elem is not None and ns_elem.text:
                page_data['namespace'] = ns_elem.text
            
            # Get text from revision
            revision = page_elem.find('revision')
            if revision is not None:
                text_elem = revision.find('text')
                if text_elem is not None and text_elem.text:
                    page_data['text'] = text_elem.text
            
            return page_data
            
        except ET.ParseError as e:
            print(f"\nXML Parse error: {e}")
            return None
        except Exception as e:
            print(f"\nError parsing page: {e}")
            return None
    
    def clean_wikitext(self, text: str) -> str:
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
    
    def filter_dump_files(self, dump_files: list, job_type: Optional[str] = None, 
                         file_extension: Optional[str] = None, 
                         filename_prefix: Optional[str] = None,
                         max_size_mb: Optional[int] = None) -> list:
        """
        Filter dump files by various criteria.
        
        Args:
            dump_files: List of dump file dictionaries
            job_type: Filter by job type (e.g., 'articlesmultistreamdump')
            file_extension: Filter by file extension (e.g., '.bz2')
            filename_prefix: Filter by filename prefix
            max_size_mb: Maximum file size in MB
        
        Returns:
            list: Filtered list of dump files
        """
        filtered = dump_files
        
        if job_type:
            filtered = [f for f in filtered if f['job'] == job_type]
        
        if file_extension:
            filtered = [f for f in filtered if f['filename'].endswith(file_extension)]
        
        if filename_prefix:
            filtered = [f for f in filtered if f['filename'].startswith(filename_prefix)]
        
        if max_size_mb:
            max_size_bytes = max_size_mb * 1024 * 1024
            filtered = [f for f in filtered if f['size'] != 'Unknown' and int(f['size']) <= max_size_bytes]
        
        return filtered
    
    def process_pages_online(self, url: str, max_pages: int = 10, namespace: str = '0') -> None:
        """
        Process pages from an online Wikipedia dump.
        
        Args:
            url: URL to the dump file
            max_pages: Maximum number of pages to process
            namespace: Namespace to filter (0 for articles)
        """
        page_count = 0
        
        for page in self.stream_dump_from_url(url):
            page_count += 1
            
            # Only process specified namespace (0 = articles)
            if page.get('namespace') == namespace:
                title = page.get('title', 'No title')
                page_id = page.get('id', 'No ID')
                text = page.get('text', '')
                
                # Clean the wikitext
                clean_text = self.clean_wikitext(text)
                
                print(f"\nPage {page_count}: {title} (ID: {page_id})")
                print(f"Text preview: {clean_text[:200]}...")
                print("-" * 80)
                
                # Limit output for demonstration
                if page_count >= max_pages:
                    break
        
        print(f"\nProcessed {page_count} pages total.")


def main():
    """
    Main function to demonstrate online Wikipedia dump reading.
    """
    print("=== Online Wikipedia Dump Reader ===\n")
    
    # Create reader instance
    reader = OnlineWikipediaDumpReader()
    
    # Get available dump files
    print("Fetching available dump files...")
    dump_files = reader.get_dump_files()
    
    if not dump_files:
        print("No dump files found.")
        return
    
    # Filter for small article dump files
    article_files = reader.filter_dump_files(
        dump_files,
        job_type='articlesmultistreamdump',
        file_extension='.bz2',
        max_size_mb=100  # Only files smaller than 100MB
    )
    
    print(f"Found {len(article_files)} small article dump files:")
    
    if article_files:
        for article_file in article_files:
            reader.process_pages_online(article_file['url'])#, max_pages=2)

        return

        # Show available files
        for i, file_info in enumerate(article_files[:5], 1):  # Show first 5 files
            size_mb = int(file_info['size']) / (1024 * 1024) if file_info['size'] != 'Unknown' else 0
            print(f"{i}. {file_info['filename']} ({size_mb:.1f} MB)")
        
        # Process the smallest file as example
        smallest_file = min(article_files, 
                           key=lambda x: int(x['size']) if x['size'] != 'Unknown' else float('inf'))
        
        size_mb = int(smallest_file['size']) / (1024 * 1024) if smallest_file['size'] != 'Unknown' else 0
        print(f"\nProcessing smallest file: {smallest_file['filename']} ({size_mb:.1f} MB)")
        print(f"URL: {smallest_file['url']}")
        
        # Process pages from the online dump
        reader.process_pages_online(smallest_file['url'], max_pages=5)
    
    else:
        print("No small article files found for processing.")


if __name__ == "__main__":
    main()
