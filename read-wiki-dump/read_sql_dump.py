import gzip
import re

# Parse categorylinks dump
def parse_categorylinks(dump_file):
    category_map = {}  # page_id -> [category_names]
    
    try:
        # Try UTF-8 first
        with gzip.open(dump_file, 'rt', encoding='utf-8') as f:
            return parse_file_content(f, category_map)
    except UnicodeDecodeError:
        print("UTF-8 failed, trying UTF-8 with error handling...")
        try:
            # Try UTF-8 with error handling
            with gzip.open(dump_file, 'rt', encoding='utf-8', errors='ignore') as f:
                return parse_file_content(f, category_map)
        except Exception as e:
            print(f"UTF-8 with ignore failed: {e}")
            try:
                # Try latin-1 encoding which accepts any byte values
                with gzip.open(dump_file, 'rt', encoding='latin-1') as f:
                    return parse_file_content(f, category_map)
            except Exception as e:
                print(f"Latin-1 failed: {e}")
                # Read as binary and decode line by line
                with gzip.open(dump_file, 'rb') as f:
                    return parse_binary_content(f, category_map)

def parse_file_content(f, category_map):
    for line in f:
        #print(line)
        if line.startswith('INSERT INTO'):
            # Extract values from SQL INSERT statements
            # Format: (cl_from, cl_to, ...)
            values = re.findall(r'\((\d+),\'([^\']+)\'', line)
            for page_id, category in values:
                if len(category_map) % 10000 == 0:
                    print(f"{page_id} -> {category}")
                #print(page_id, category)
                if page_id not in category_map:
                    category_map[page_id] = []
                category_map[page_id].append(category)
    return category_map

def parse_binary_content(f, category_map):
    for line in f:
        try:
            # Try to decode each line individually
            line_str = line.decode('utf-8', errors='ignore')
        except:
            # If that fails, try latin-1
            try:
                line_str = line.decode('latin-1')
            except:
                # Skip lines that can't be decoded
                continue
                
        if line_str.startswith('INSERT INTO'):
            # Extract values from SQL INSERT statements
            # Format: (cl_from, cl_to, ...)
            values = re.findall(r'\((\d+),\'([^\']+)\'', line_str)
            for page_id, category in values:
                if page_id not in category_map:
                    category_map[page_id] = []
                category_map[page_id].append(category)
    return category_map

# Build reverse map: category_name -> page_id
def get_category_pages(page_dump):
    category_ids = {}
    # Parse page.sql to find which page_ids are categories (namespace 14)
    # Then map category_name -> page_id
    return category_ids

# Recursive hierarchy traversal
def get_full_hierarchy(page_id, category_map, category_ids, visited=None):
    if visited is None:
        visited = set()
    
    if page_id in visited:
        return []  # Avoid cycles
    
    visited.add(page_id)
    hierarchy = []
    
    if page_id in category_map:
        for category in category_map[page_id]:
            hierarchy.append(category)
            # Find this category's page_id and recurse
            if category in category_ids:
                cat_page_id = category_ids[category]
                parent_cats = get_full_hierarchy(cat_page_id, category_map, 
                                                 category_ids, visited)
                hierarchy.extend(parent_cats)
    
    return hierarchy

def main():
    parse_categorylinks(r"c:\Users\HaimL1\Downloads\hewiki-20251220-categorylinks.sql.gz")
    
    for i in range(999999):
        hierarchy = get_full_hierarchy(i, {}, {})  # Example usage

        print(f"{i} {hierarchy}")
main()