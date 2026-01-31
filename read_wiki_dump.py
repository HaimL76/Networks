import bz2
import re
import webbrowser

def get_wikipedia_url_by_id(page_id: int, lang: str = "he") -> str:
    """Generate Wikipedia URL from page ID"""
    return f"https://{lang}.wikipedia.org/w/index.php?curid={page_id}"

def open_wikipedia_page(page_id: int, lang: str = "he"):
    """Open Wikipedia page in browser"""
    url = get_wikipedia_url_by_id(page_id, lang)
    webbrowser.open(url)

def read_dump_file(file_path: str, open_in_browser: bool = False, max_pages: int = None):
    with bz2.open(file_path, 'rt', encoding='utf-8', errors='ignore') as f, open(r"c:\gpp\kuku.txt", "w", encoding="utf-8") as fw:   
        counter: int = 0

        in_page: bool = False

        page_id: int = None
        page_title: str = None
        
        for line_num, line in enumerate(f):
            if line:
                line = line.strip()

            #fw.write(line + "\n")

            if line == "<page>":
                in_page = True

            if line == "</page>":
                if in_page:
                    print(f"counter: {counter}, line_num: {line_num}, page_id: {page_id}, page_title: {page_title}")
                    
                    page_title = None
                    page_id = None
                    counter += 1
                in_page = False

            if False:# line_num > 10000:
                break

            if not isinstance(page_id, int) and '<id>' in line:
                id_match = re.search(r'<id>(.*?)</id>', line)

                if id_match:
                    str_page_id: str = id_match.group(1)

                    if str_page_id:
                        str_page_id = str_page_id.strip()

                    if str_page_id and str_page_id.isnumeric():
                        page_id: int = int(str_page_id)


            if not isinstance(page_title, str) and '<title>' in line:
                title_match = re.search(r'<title>(.*?)</title>', line)
                
                if title_match:
                    title: str = title_match.group(1)

                    if title:
                        title = title.strip()
                        
                    page_title = title

                    continue

                    # Generate Wikipedia URL
                    wiki_url = get_wikipedia_url_by_id(id, "he")
                    
                    print(f"[{line_num, counter}]: ID={id}, Title={title}")
                    print(f"  URL: {wiki_url}")
                    
                    # Optionally open in browser
                    if open_in_browser and counter <= 5:  # Limit to first 5 to avoid spam
                        open_wikipedia_page(tup[0], "he")
                    
                    # Break if max_pages reached
                    if max_pages and counter >= max_pages:
                        print(f"Reached maximum of {max_pages} pages")
                        break
                    
                    tup = None  # Reset for next page

def main():
    # Example usage:
    # read_dump_file(r"C:\Users\HaimL1\Downloads\hewiki-latest-pages-articles.xml.bz2", open_in_browser=True, max_pages=10)
    read_dump_file(r"C:\Users\HaimL1\Downloads\hewiki-latest-pages-articles.xml.bz2", max_pages=20)

main()