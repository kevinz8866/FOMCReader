import datetime
import numpy as np
import pandas as pd
from utils import reshape_into_train, reshape_into_train2, string_to_date, string_to_date2, string_to_date3
import warnings
warnings.filterwarnings("ignore")

def preprocess_transcript():
    
    data = pd.read_csv('../input/processed/word_per_sec_with_trades.csv')
    raw = data[['Trade_Or_Not','Word1','Word2','Word3','Word4']]
    word_per_sec = 4
    target_length=15
    
    X_raw = raw[['Word1','Word2','Word3','Word4']].values.reshape(-1,)
    unique_words = sorted(set(X_raw))
    vocabulary = {w:i for i, w in enumerate(unique_words)}
    X_raw = list(map(lambda x: vocabulary[x], X_raw))
    X_raw = np.array(X_raw).reshape(-1,word_per_sec)
    
    y_raw = raw['Trade_Or_Not'].rolling(15).max().values
    
    return reshape_into_train(X_raw,y_raw), vocabulary

def preprocess_transcript_statement(add_tone=False, return_statement_dict=False):
    
    data = pd.read_csv('../input/processed/word_per_sec_with_trades.csv')
    statement = pd.read_excel('../input/FOMC_statements_text.xlsx')
    
    statement_raw = statement[np.isin(statement['FOMC Statement Release Date'].apply(string_to_date).values,(data['Date'].apply(string_to_date2).unique()))]
    statement_raw['Date'] = statement_raw['FOMC Statement Release Date'].apply(string_to_date)
    statement_raw['Statement_list'] = statement_raw['FOMC Statement'].apply(lambda x: x.lower().split())
    statement_raw['Statement_list'] = statement_raw['Statement_list'].apply(lambda x: np.append(np.array(x),np.full((800-len(x), ), '[PAD]')))
    
    X_raw = data[['Word1','Word2','Word3','Word4']].values.reshape(-1,)
    unique_words =set(np.concatenate(statement_raw['Statement_list'].values))
    unique_words.update(set(X_raw))
    vocabulary = {w:i for i, w in enumerate(unique_words)}
    
    def list_to_index(x):
        res = []
        for i in x:
            res.append(vocabulary[i])
        return np.array(res).astype('float64')

    statement_raw['Statement_list'] = statement_raw['Statement_list'].apply(list_to_index)
    statement_dict = statement_raw[['Date','Statement_list']].set_index(['Date']).to_dict()['Statement_list']
    
    def append_statement(x): 
        return statement_dict[datetime.datetime.strptime(x[:10],'%Y-%m-%d')]
    
    data['statement']=data['Date'].apply(append_statement)
    
    word_per_sec = 4
    target_length=15
    raw = data[['Trade_Or_Not','statement']]
    
    X_raw = list(map(lambda x: vocabulary[x], X_raw))
    X_raw = np.array(X_raw).reshape(-1,word_per_sec)
    S_raw = np.stack(raw['statement'].values)
    y_raw = raw['Trade_Or_Not'].rolling(15).max().values
    
    X,S,y = reshape_into_train2(X_raw,S_raw,y_raw)
    X = X.astype('float64')
    
    if add_tone:
        tone =pd.read_csv('../input/processed/tones.csv')
        tone_raw = tone[np.isin(tone['date'].apply(string_to_date2).values,data['Date'].apply(string_to_date3).unique())]
        a = tone_raw[['sad', 'angry', 'neutral', 'happy', 'disgust','fearful']].values.reshape((-1,18))[:-10]
        T = np.transpose(np.tile(a,(15,1,1)),[1,2,0])
        
        D = np.concatenate([X,S,T],axis=1)
        if return_statement_dict:
            return D, y, vocabulary, statement_dict
        return D, y, vocabulary 
    
    D = np.append(X,S,axis=1)
    return D, y, vocabulary