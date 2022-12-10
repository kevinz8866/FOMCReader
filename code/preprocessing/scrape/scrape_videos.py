import pytube
import pandas as pd
from datetime import datetime

def scrape(x, FOMC_past):
    """
    This function scrapes the FOMC videos that are not already scraped. 
    x can either be a channel or a playlist object from pytube. 
    """
    FOMC_dates = pd.DataFrame(columns=['chair', 'date', 'res', 'filename'])
    print(f'Downloading videos:')
    for video in x.videos:
        # only download full videos
        title = video.title
        if 'Introductory Statement' not in title and title.startswith('FOMC Press Conference '):
            FOMC_date = ' '.join(title.split(' ')[-3:])
            FOMC_date = datetime.strptime(FOMC_date, '%B %d, %Y').date()
            FOMC_date_str = FOMC_date.strftime('%Y-%m-%d')
            if (FOMC_past['date'].ne(FOMC_date_str)).all():
                print(f'Downloading FOMC press conference on {FOMC_date_str}')
                # only download videos that are not already downloaded
                # Check for chairman
                if FOMC_date.year > 2017:
                    chair = 'Powell'
                elif FOMC_date.year > 2013:
                    chair = 'Yellen'
                else:
                    chair = 'Bernanke'
                # Different period has different resolutions
                res_break = datetime.strptime('2012-01-25', '%Y-%m-%d').date()
                if FOMC_date == res_break:
                    res = '240p'
                    #video.streams.filter(file_extension='mp4', res=res).first().download('data/video')
                elif FOMC_date < res_break:
                    res = '144p'
                else:
                    res = '720p'
                video.streams.filter(file_extension='mp4', res=res).first().download(output_path='data/video/', 
                                                                                     filename=FOMC_date.strftime('%Y%m%d') + '.mp4')
                
                FOMC_dates = FOMC_dates.append({'chair':chair, 'date':FOMC_date, 
                                                'res':res}, ignore_index=True)
    return FOMC_dates

if __name__ == "__main__":
    # Scraping by FOMC announcement playlist
    # a lot of past videos are missing in this list
    c = pytube.Channel('https://www.youtube.com/c/federalreserve/videos')
    FOMC_old = pd.read_csv('data/meta/FOMC_meta.csv')[['chair', 'date', 'res']]
    FOMC_dates = scrape(c, FOMC_old)
    FOMC_dates.to_csv('data/meta/FOMC_meta_new.csv', index=False)   
    FOMC_dates = pd.concat([FOMC_old, FOMC_dates], ignore_index=True)
    FOMC_dates['date'] = pd.to_datetime(FOMC_dates['date'])
    FOMC_dates.sort_values(by='date', inplace=True)
    FOMC_dates.to_csv('data/meta/FOMC_meta.csv', index=False)