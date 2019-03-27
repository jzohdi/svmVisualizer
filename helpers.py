from bs4 import BeautifulSoup
from requests import get
from random import shuffle
import threading, time, signal
import json
import numpy as np
from re import sub

from datetime import timedelta
class Parser:
    def __init__(self):
        self.url = 'https://en.wikiquote.org/wiki/Special:Random'
        self.title = None
        self.status = None
        self.all_quotes = []
        
    def make_request(self):
        response = get(self.url)
        if response.status_code == 200:
            self.build_quotes(response)
        else:
            return False
        
    def build_quotes(self, response):
        self.status = 200
        content = response.content
        soup = BeautifulSoup(content, 'html.parser')
        self.title = ' '.join(soup.find('title').get_text().split('-')[:-1])
        self.get_quotes_and_author(soup)

    def get_quotes_and_author(self, soup):
        if self.status != 200:
            return None
        quotes = soup.find_all('ul')
        shuffle(quotes)
        for quote in quotes:
            self.add_quote_if_valid(quote)
            
    def add_quote_if_valid(self, quote):
        this_quote = quote.find('li')
        this_author = quote.find('ul')
        if this_quote and this_author:
            ( this_quote, this_author ) = ( this_quote.get_text(), this_author.get_text() )
#            ( this_quote, this_author ) = self.scrub_title_and_author(this_quote, this_author)
            if len(this_quote) > 10 and len(this_author) > 0:
                
                quote_dict = {'source': self.title.strip(), 
                              'author' : this_author.strip(), 
                              'quote' : this_quote.strip().replace('  ', ' ')}
                
                self.all_quotes.append(quote_dict)
                
        if len(self.all_quotes) == 0:
            self.make_request()
#        else:
#            print(self.all_quotes)
            
    def scrub_title_and_author(self, text, author):
        title_to_list = self.title.split(" ")
        author_to_list = author.split(" ")
        for word in author_to_list:
            word = self.remove_end_quotes(word)
            text = sub(r"\b%s\b" % word, '', text)
        for word in title_to_list:
            text = sub(r"\b%s\b" % word, '', text)
        return ( text, author )
    
    def remove_end_quotes(self, word):
        if word.startswith("'") or word.startswith('"'):
            word = word[1:]
        if word.endswith('"') or word.endswith("'"):
            word = word[:-1]
        return word
    def get_recent_quotes(self):
        quotes_to_return = self.all_quotes[:]
        self.all_quotes = []
        return quotes_to_return
    
def parse_request(data_Set):
    
    raw_data = json.loads(data_Set)
    dimensions = len(raw_data[0]) - 1
#    print(dimensions)
    
    min_x = float("inf")
    max_x = float("-inf")
    min_y = float("inf")
    max_y = float("-inf")
    min_z = float("inf")
    max_z = float("-inf")
    
    if dimensions == 2:
        for data in raw_data:
            min_x = min(float(min_x), float(data[0]))
            max_x = max(float(max_x), float(data[0]))
            min_y = min(float(min_y), float(data[1]))
            max_y = max(float(max_y), float(data[1]))
            
        x_y = [[float(raw_data[i][0]), float(raw_data[i][1])] for i in range( len(raw_data) )]
        
        labels = [raw_data[i][2] for i in range(len(raw_data))]
        
        unique_labels = set(labels)
        low_count = min([labels.count(label) for label in unique_labels])
        low_cv = min(5, low_count)
        
        range_vals = {'max_x' : max_x, 'min_x': min_x, 'max_y' : max_y, 'min_y' : min_y}
        return ( np.array(x_y), np.array(labels), low_cv, range_vals)
    if dimensions == 3:
        for data in raw_data:
            min_x = min(float(min_x), float(data[0]))
            max_x = max(float(max_x), float(data[0]))
            min_y = min(float(min_y), float(data[1]))
            max_y = max(float(max_y), float(data[1]))
            min_z = min(float(min_z), float(data[2]))
            max_z = max(float(max_z), float(data[2]))
            
        x_y_z = [[float(raw_data[i][0]), float(raw_data[i][1]), float(raw_data[i][2]) ] for i in range( len(raw_data) ) ]
        
        labels = [raw_data[i][3] for i in range( len(raw_data) ) ]
        
        unique_labels = set(labels)
        low_count = min([labels.count(label) for label in unique_labels])
        low_cv = min(5, low_count)
        
        range_vals = {'max_x' : max_x, 'min_x': min_x, 'max_y' : max_y, 'min_y' : min_y, 'max_z' : max_z, 'min_z' : min_z}
        return (np.array(x_y_z), np.array(labels), low_cv, range_vals)
        
class ProgramKilled(Exception):
    pass
    
def signal_handler(signum, frame):
    raise ProgramKilled
    
class Job(threading.Thread):
    def __init__(self, interval, execute, *args, **kwargs):
        threading.Thread.__init__(self)
        self.daemon = False
        self.stopped = threading.Event()
        self.interval = interval
        self.execute = execute
        self.args = args
        self.kwargs = kwargs
        
    def stop(self):
                self.stopped.set()
                self.join()
    def run(self):
            while not self.stopped.wait(self.interval.total_seconds()):
                self.execute(*self.args, **self.kwargs)

"""
def show_map(A, B, colors, filename):
    plt.figure()
    plt.scatter(A, B, c = colors)
    #    plt.savefig(str(parent_folder) + filename, bbox_inches='tight')
    plt.show()
    plt.clf()
    
parent_folder = os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir))

while True:
      try:
          time.sleep(1)
          print('here')
      except (ProgramKilled,KeyboardInterrupt, SystemExit):
          print("Program killed: running cleanup code")
          job.stop()
          break
"""