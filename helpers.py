class Parser:
    def __init__(self, get, np, BeautifulSoup, json, shuffle, sub):
        self.url = 'https://en.wikiquote.org/wiki/Special:Random'
        self.title = None
        self.status = None
        self.all_quotes = []

        self.get = get
        self.np = np
        self.BeautifulSoup = BeautifulSoup
        self.json = json
        self.shuffle = shuffle
        self.sub = sub

    def make_request(self):
        response = self.get(self.url)
        if response.status_code == 200:
            self.build_quotes(response)
        else:
            return False

    def build_quotes(self, response):
        self.status = 200
        content = response.content
        soup = self.BeautifulSoup(content, 'html.parser')
        self.title = ' '.join(soup.find('title').get_text().split('-')[:-1])
        self.get_quotes_and_author(soup)

    def get_quotes_and_author(self, soup):
        if self.status != 200:
            return None
        quotes = soup.find_all('ul')
        self.shuffle(quotes)
        for quote in quotes:
            self.add_quote_if_valid(quote)

    def add_quote_if_valid(self, quote):
        this_quote = quote.find('li')
        this_author = quote.find('ul')
        if this_quote and this_author:
            (this_quote, this_author) = (this_quote.get_text(),
                                         this_author.get_text())
            (this_quote, this_author) = self.scrub_title_and_author(
                this_quote, this_author)
            if len(this_quote) > 10 and len(this_author) > 0:

                quote_dict = {
                    'source': self.title.strip(),
                    'author': this_author.strip(),
                    'quote': this_quote.strip().replace('  ', ' ')
                }

                self.all_quotes.append(quote_dict)

        if len(self.all_quotes) == 0:
            self.make_request()

    def scrub_title_and_author(self, text, author):
        title_to_list = self.title.split(" ")
        text = text.replace(author, '')
        for word in title_to_list:
            text = self.sub(r"\b%s\b" % word, '', text)
        return (text, author)

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