import xml.etree.ElementTree as ET
import re
import bz2
import urllib.request
from collections import defaultdict

class HebrewWikiParser:
    def __init__(self, dump_file_path):
        """
        Initialize parser with path to Hebrew Wikipedia dump file.
        Download from: https://dumps.wikimedia.org/hewiki/latest/hewiki-latest-pages-articles.xml.bz2
        """
        self.dump_file = dump_file_path
        self.ns = {'mw': 'http://www.mediawiki.org/xml/export-0.10/'}
        
    def parse_categories(self, text):
        """Extract categories from wiki text."""
        if not text:
            return []
        
        # Pattern for Hebrew Wikipedia categories: [[קטגוריה:...]]
        pattern = r'\[\[קטגוריה:([^\]]+)\]\]'
        categories = re.findall(pattern, text)
        
        # Clean up category names (remove sort keys)
        clean_cats = []
        for cat in categories:
            # Remove sort key if present (after |)
            cat_name = cat.split('|')[0].strip()
            clean_cats.append(cat_name)
        
        return clean_cats
    
    def open_dump(self):
        """Open dump file (handles both .xml and .xml.bz2)"""
        if self.dump_file.endswith('.bz2'):
            return bz2.open(self.dump_file, 'rt', encoding='utf-8')
        else:
            return open(self.dump_file, 'r', encoding='utf-8')
    
    def parse_dump(self, max_pages=None):
        """
        Parse the dump file and extract pages with their categories.
        Returns a dictionary: {page_title: [categories]}
        """
        pages_data = {}
        page_count = 0
        
        print("Starting to parse Hebrew Wikipedia dump...")
        
        with self.open_dump() as f:
            # Use iterparse for memory efficiency with large files
            for event, elem in ET.iterparse(f, events=('end',)):
                if elem.tag == '{http://www.mediawiki.org/xml/export-0.10/}page':
                    # Extract title
                    title_elem = elem.find('mw:title', self.ns)
                    if title_elem is None:
                        elem.clear()
                        continue
                    
                    title = title_elem.text
                    
                    # Skip special namespaces (keep only main namespace articles)
                    ns_elem = elem.find('mw:ns', self.ns)
                    if ns_elem is not None and ns_elem.text != '0':
                        elem.clear()
                        continue
                    
                    # Extract page text
                    revision = elem.find('mw:revision', self.ns)
                    if revision is not None:
                        text_elem = revision.find('mw:text', self.ns)
                        if text_elem is not None and text_elem.text:
                            categories = self.parse_categories(text_elem.text)
                            
                            if categories:  # Only store pages with categories
                                pages_data[title] = categories
                                page_count += 1
                                
                                if page_count % 1000 == 0:
                                    print(f"Processed {page_count} pages with categories...")
                    
                    # Clear element to free memory
                    elem.clear()
                    
                    # Stop if max_pages reached
                    if max_pages and page_count >= max_pages:
                        break
        
        print(f"Parsing complete! Found {page_count} pages with categories.")
        return pages_data
    
    def build_category_hierarchy(self, pages_data, category_pages):
        """
        Build hierarchical structure of categories.
        category_pages: dict mapping category names to their parent categories
        """
        hierarchy = {}
        
        for page, cats in pages_data.items():
            hierarchy[page] = {
                'direct_categories': cats,
                'full_hierarchy': self._get_full_hierarchy(cats, category_pages)
            }
        
        return hierarchy
    
    def _get_full_hierarchy(self, categories, category_pages, visited=None):
        """Recursively build full category hierarchy."""
        if visited is None:
            visited = set()
        
        hierarchy = {}
        
        for cat in categories:
            if cat in visited:  # Avoid cycles
                continue
            
            visited.add(cat)
            hierarchy[cat] = []
            
            if cat in category_pages:
                parent_cats = category_pages[cat]
                if parent_cats:
                    hierarchy[cat] = self._get_full_hierarchy(
                        parent_cats, category_pages, visited.copy()
                    )
        
        return hierarchy
    
    def parse_category_pages(self):
        """Extract category pages and their parent categories."""
        category_pages = {}
        
        print("Parsing category pages...")
        
        with self.open_dump() as f:
            for event, elem in ET.iterparse(f, events=('end',)):
                if elem.tag == '{http://www.mediawiki.org/xml/export-0.10/}page':
                    # Check if this is a category page
                    ns_elem = elem.find('mw:ns', self.ns)
                    if ns_elem is not None and ns_elem.text == '14':  # Category namespace
                        title_elem = elem.find('mw:title', self.ns)
                        if title_elem is not None:
                            # Remove "קטגוריה:" prefix
                            cat_name = title_elem.text.replace('קטגוריה:', '')
                            
                            # Extract parent categories
                            revision = elem.find('mw:revision', self.ns)
                            if revision is not None:
                                text_elem = revision.find('mw:text', self.ns)
                                if text_elem is not None and text_elem.text:
                                    parent_cats = self.parse_categories(text_elem.text)
                                    category_pages[cat_name] = parent_cats
                    
                    elem.clear()
        
        print(f"Found {len(category_pages)} category pages.")
        return category_pages


# Example usage
if __name__ == "__main__":
    # Path to your Hebrew Wikipedia dump file
    dump_path = r"c:\users\haiml1\downloads\hewiki-latest-pages-articles.xml.bz2"
    
    # Initialize parser
    parser = HebrewWikiParser(dump_path)
    
    # Parse article pages (limit to 100 for testing, remove for full parse)
    pages = parser.parse_dump(max_pages=100)
    
    # Parse category pages to build hierarchy
    category_pages = parser.parse_category_pages()
    
    # Build full hierarchy
    hierarchy = parser.build_category_hierarchy(pages, category_pages)
    
    # Display results
    print("\n" + "="*80)
    print("SAMPLE RESULTS (first 5 pages):")
    print("="*80)
    
    for i, (title, data) in enumerate(list(hierarchy.items())[:5]):
        print(f"\nPage: {title}")
        print(f"Direct Categories: {data['direct_categories']}")
        print(f"Full Hierarchy: {data['full_hierarchy']}")
        
        if i >= 4:
            break
    
    # Save to file
    print("\n" + "="*80)
    print("Saving results to file...")
    
    with open('hebrew_wiki_categories.txt', 'w', encoding='utf-8') as f:
        for title, data in hierarchy.items():
            f.write(f"Page: {title}\n")
            f.write(f"Categories: {', '.join(data['direct_categories'])}\n")
            f.write(f"Hierarchy: {data['full_hierarchy']}\n")
            f.write("-" * 80 + "\n")
    
    print(f"Results saved! Total pages processed: {len(hierarchy)}")