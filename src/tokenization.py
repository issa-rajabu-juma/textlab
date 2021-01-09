from datasets import load_dataset


swahili_news = load_dataset('swahili news')

news = swahili_news['train'][0]
print(news)