# The task is to extract the word "politics" from the given URL.
# We will use Python's `split` function to isolate the word "politics" at the end of the URL.

url = "https://www.bloomberg.com/news/articles/2024-10-04/kosovo-s-premier-hits-back-at-western-criticism-over-treatment-of-serbs?srnd=phx-politics"

category = url.split('=')[-1]
category = category.split('-')[-1]
print(category)
