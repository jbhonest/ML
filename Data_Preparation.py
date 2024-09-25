import requests
from bs4 import BeautifulSoup
import psycopg2


def scrape_content(url):
    try:
        # Send an HTTP GET request to fetch the page content
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Parse the page content with BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        title = soup.title.string if soup.title else 'No title found'

        # Find all divs with data-component="text-block"
        divs = soup.find_all('div', {'data-component': 'text-block'})

        # Initialize an empty list to store paragraph text
        all_paragraphs = []

        # Extract the content of all <p> tags and combine them
        for div in divs:
            paragraphs = div.find_all('p')
            for p in paragraphs:
                all_paragraphs.append(p.get_text())

        # Join all paragraphs into a multi-paragraph text with line breaks
        combined_text = '\n'.join(all_paragraphs)

        return title, combined_text

    except requests.RequestException as e:
        print(f"Failed to retrieve URL {url}: {e}")
        return None


def save_to_database(url, title, content):
    """Saves the scraped content into the PostgreSQL database."""
    if content:
        try:
            # PostgreSQL connection details
            host = "localhost"
            dbname = "ml"
            user = "ml_user"
            password = "ml_10925"

            # Connect to PostgreSQL
            connection = psycopg2.connect(
                host=host,
                dbname=dbname,
                user=user,
                password=password
            )
            cursor = connection.cursor()

            # Insert title and content into the table
            cursor.execute('''
                INSERT INTO articles (url, title, content) 
                VALUES (%s, %s, %s)
            ''', (url, title, content))

            # Commit the transaction
            connection.commit()
            print("Data inserted successfully into database")

        except Exception as error:
            print("Error while connecting to database:", error)

        finally:
            # Close the connection
            if connection:
                cursor.close()
                connection.close()
    else:
        print("No content to save to the database")


def read_urls_from_file(file_path):
    """Reads URLs from a given file."""
    with open(file_path, 'r') as file:
        urls = [line.strip() for line in file.readlines()]
    return urls


# File containing the URLs
url_file = 'urls.txt'

# Read URLs from the file
urls = read_urls_from_file(url_file)

# Scrape and save content for each URL
for url in urls:
    title, content = scrape_content(url)
    save_to_database(url, title, content)
