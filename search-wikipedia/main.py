import os
import re
import wikipedia

# Example usage of wikipedia module
# print(wikipedia.summary("Wikipedia"))
# print(wikipedia.search("Barack"))

def read_names(folder: str):
    names: set[str] = set()

    files: list[str] = os.listdir(folder)

    for file in files:
        full_path: str = os.path.join(folder, file)
        if os.path.isfile(full_path):
            with open(full_path, "r", encoding="utf-8") as fr:
                for line in fr:
                    if "<li>" in line:
                        arr: list[str] = re.split(r'<li>|</li>', line)
                        for name in arr:
                            name = name.strip()
                            if name and not name.startswith("<"):
                                name_parts: list[str] = name.split(" ")

                                if len(name_parts) > 1:
                                    for part in name_parts:
                                        part = part.strip()
                                        if part:
                                            names.add(part)
    return names

def search(word: str):
    files = os.listdir(r"c:\\gpp\\names\\pages")
    if files:
        for file in files:
            os.remove(os.path.join(r"c:\\gpp\\names\\pages", file))
    
    names: set[str] = read_names(r"c:\\gpp\\names")

    names = sorted(names)

    for name in names:
        name = name.strip()
        print(f"Name: {name}")
        try:
            search = wikipedia.search(name)

            for index in range(len(search)):
                search_term = search[index]
                print(f"[{index}]: {search_term}")

                try_page(search_term, level=0, names=names)
        except Exception as e:
            print(f"Error searching for '{name}': {e}")

def try_page(search_term, level=0, names: set[str] = None):
    output_path: str = f"c:\\gpp\\names\\pages\\{search_term}.txt"
    
    if os.path.exists(output_path):
        print(f"[{level}]: Page '{search_term}' already processed.")
        return
    
    try:
        page = wikipedia.page(search_term)

        page_links = page.links

        print(f"[{level}]: Processing page {search_term} with {len(page_links)} links.")

        list_links: list[tuple[str, list[str]]] = []

        print(f"[{level}]: Writing links to {output_path}")
        
        for link_index in range(len(page_links)):
            page_link = page_links[link_index]

            link_parts: list[str] = page_link.split(" ")

            part_index: int = 0
            name_parts: list[str] = []
            
            while len(name_parts) < 2 and part_index < len(link_parts):
                part: str = link_parts[part_index]
                part_index += 1
                part = part.strip()
                if part in names:
                    name_parts.append(part)

                if len(name_parts) > 1:
                    tup: tuple[str, list[str]] = (page_link, name_parts)
                    list_links.append(tup)
        
        if isinstance(list_links, list) and len(list_links) > 0:
            with open(output_path, "w", encoding="utf-8") as fw:
                for tup in list_links:
                    page_link: str = tup[0]
                    fw.write(f"{page_link}\n")
                    print(f"[{level}][]: {page_link}, name parts: {','.join(tup[1])}")

            for tup in list_links:
                page_link: str = tup[0]
                try_page(page_link, level=level+1, names=names)

    except Exception as e:  # wikipedia.DisambiguationError as e:
        _ = 0  # print(f"[{index}]: DisambiguationError: {e.options}")
search("Haim")