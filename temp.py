with open('urls.txt', 'r') as file:
    urls = file.readlines()

# Filter out the inserted URLs
remaining_urls = [url for url in urls if url.strip(
) != 'https://www.bloomberg.com/news/articles/2024-10-03/softbank-s-son-envisions-ai-running-households-in-next-few-years?srnd=phx-technology']

print(remaining_urls)
