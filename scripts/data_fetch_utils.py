import scripts.config
import pandas as pd
import requests

def fetch_sentiment_data(time_from, time_to):
    """
    Function to fetch news sentiment data from Alpha Vantage
    """
    base_url = "https://www.alphavantage.co/query"

    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": scripts.config.tickers,  # Tickers to filter news for
        "time_from": time_from,
        "time_to": time_to,
        "apikey": scripts.config.API_KEY,
    }

    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        data = response.json()
        
        if "feed" in data:
            # Convert the feed to a DataFrame
            news_df = pd.DataFrame(data["feed"])
            
            news_df = news_df.explode("ticker_sentiment")  # Expand each list in `ticker_sentiment` into separate rows
            news_df["ticker"] = news_df["ticker_sentiment"].apply(lambda x: x.get("ticker") if isinstance(x, dict) else None)
            news_df["ticker_sentiment_score"] = news_df["ticker_sentiment"].apply(lambda x: x.get("ticker_sentiment_score") if isinstance(x, dict) else None)
            news_df["ticker_relevance_score"] = news_df["ticker_sentiment"].apply(lambda x: x.get("relevance_score") if isinstance(x, dict) else None)
            
            news_df = news_df.drop(columns=["ticker_sentiment"])
            
            news_df.to_csv("sentiment_data_with_tickers.csv", index=False)
        else:
            print("No news sentiment data available for the specified period.")
    else:
        print(f"Error: {response.status_code}, Message: {response.text}")

    #let's build the features tha we'll actually need
    news_df['time_published'] = pd.to_datetime(news_df['time_published'])
    news_df['date'] = news_df['time_published'].dt.date

    #convert to float
    news_df[['overall_sentiment_score', 'ticker_sentiment_score', 'ticker_relevance_score']] = news_df[['overall_sentiment_score', 'ticker_sentiment_score', 'ticker_relevance_score']].astype(float)

    #aggregate data to daily level (currently it's hourly)
    daily_sentiment = news_df.groupby(['ticker', 'date']).agg({
        'ticker_sentiment_score': ['mean', 'sum'],
        'ticker_relevance_score': ['mean', 'sum'],
        'overall_sentiment_score': ['mean', 'sum'],

    }).reset_index() #convert back to normal df

    daily_sentiment.columns = ['ticker', 'date', 'avg_sentiment', 'total_sentiment', 'avg_relevance_score', 'total_relevance_score', 'avg_overall_sentiment', 'total_overall_sentiment']

    daily_sentiment_filtered = daily_sentiment.loc[daily_sentiment['ticker'].isin(['AAPL', 'GOOG'])]

    return daily_sentiment_filtered