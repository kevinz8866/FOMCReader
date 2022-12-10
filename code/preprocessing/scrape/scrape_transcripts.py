import requests
import pandas as pd

if __name__ == "__main__":
    FOMC_meta = pd.read_csv('data/meta/FOMC_meta_trail.csv')
    FOMC_dates = FOMC_meta['date'].str.replace('-', '')
    for FOMC_date in FOMC_dates:
        url = f'https://www.federalreserve.gov/mediacenter/files/FOMCpresconf{FOMC_date}.pdf'
        response = requests.get(url)
        with open(f'data/transcript/pdf/{FOMC_date}.pdf', 'wb') as f:
            f.write(response.content)