# Source - https://stackoverflow.com/a
# Posted by furas, modified by community. See post 'Timeline' for change history
# Retrieved 2025-12-24, License - CC BY-SA 4.0

import requests

params = {
    'action': 'parse',
#    'page': 'Pet_door',
    'page': 'USER:Catrope',
#    'prop': 'text',
    'prop': 'wikitext',
    'formatversion': 2,
    'format': 'json'
}

headers: dict = {'User-Agent': 'search-wikipedia/0.0 (https://example.org/wikipedia/; haiml76@yahoo.com)'}

response = requests.get("https://en.wikipedia.org/w/api.php", 
                        params=params, headers=headers)
data = response.json()

#print(data.keys())
#print(data)
#print('---')

#print(data['parse'].keys())
#print(data['parse'])
#print('---')

#print(data['parse']['text'])    # if you use param `'prop': 'text'
#print('---')

print(data['parse']['wikitext']) # if you use param `'prop': 'wikitext'
print('---')

# print all not empty lines
for line in data['parse']['wikitext'].split('\n'):
    line = line.strip()  # remove spaces
    if line: # skip empty lines
        print('--- line ---')
        print(line)

print('---')

# get first line of text (with "I'm not usually active on English Wikipedia. Please refer...")
print(data['parse']['wikitext'].split('\n')[0])
