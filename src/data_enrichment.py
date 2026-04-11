import requests
from bs4 import BeautifulSoup
import pandas as pd

class DataScraper:
    def __init__(self):
        # We store the URL and headers as attributes of the class
        self.url = 'https://en.wikipedia.org/wiki/List_of_U.S._states_and_territories_by_population'
        self.headers = {'User-Agent': 'Mozilla/5.0'}

    def _fetch_wikipedia_data(self):
        """Internal helper method to do the actual web scraping."""
        print("Scraper: Fetching live data from Wikipedia...")
        response = requests.get(self.url, headers=self.headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'class': 'wikitable'})

        data = []
        for row in table.find_all('tr'):
            cols = row.find_all(['td', 'th'])
            cols = [ele.text.strip() for ele in cols]
            if cols:
                data.append(cols)

        max_len = max(len(r) for r in data)
        norm_data = [r + ['']*(max_len - len(r)) for r in data]

        scraped_df = pd.DataFrame(norm_data[2:], columns=norm_data[0])
        scraped_df = scraped_df.iloc[:, :3]
        scraped_df.columns = ['State', 'Population_2024', 'Population_2020']
        
        return scraped_df[['State', 'Population_2024']]

    def enrich_data(self, df):
        """Public method to merge the scraped data into the main dataframe."""
        scraped_df = self._fetch_wikipedia_data()
        print("Scraper: Merging live population data into the main dataset...")

        df['State'] = 'California'
        df = pd.merge(df, scraped_df, on='State', how='left')

        df['Population_2024'] = df['Population_2024'].str.replace(',', '')
        df['Population_2024'] = pd.to_numeric(df['Population_2024'], errors='coerce')
        df['Population_2024'] = df['Population_2024'].fillna(df['Population_2024'].median())

        return df