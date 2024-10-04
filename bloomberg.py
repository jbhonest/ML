from selenium import webdriver
from bs4 import BeautifulSoup
from datetime import datetime
import psycopg2
import time


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
            current_time = datetime.now()
            print(f"Data inserted successfully into database at {
                  current_time.strftime("%H:%M")}")
            return True

        except Exception as error:
            print("Error while connecting to database:", error)

        finally:
            # Close the connection
            if connection:
                cursor.close()
                connection.close()
    else:
        print("No content to save to the database")
        return False


def scrape_content(url):
    try:
        driver = webdriver.Chrome()  # Or use webdriver.Firefox(), etc.

        driver.get(url)
        # Get the page source
        html = driver.page_source

        # Parse the page with BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')

        title = soup.title.string if soup.title else 'No title found'

        paragraphs = soup.find_all('p', {"data-component": "paragraph"})

        # Extract the text from each <p> tag
        paragraph_texts = [p.get_text() for p in paragraphs]

        # Initialize an empty list to store paragraph text
        all_paragraphs = []
        # Print the results
        for text in paragraph_texts:
            all_paragraphs.append(text)

        # Join all paragraphs into a multi-paragraph text with line breaks
        combined_text = '\n'.join(all_paragraphs)

        return title, combined_text

    except Exception as e:
        print(f"Failed to retrieve URL {url}: {e}")
        return None


def read_urls_from_file(file_path):
    """Reads URLs from a given file."""
    with open(file_path, 'r') as file:
        urls = [line.strip() for line in file.readlines()]
    return urls


def remove_url_from_file(file_path, url_to_remove):
    with open(file_path, 'r') as file:
        urls = file.readlines()

    # Filter out the inserted URLs
    remaining_urls = [url for url in urls if url.strip() != url_to_remove]

    # Write the remaining URLs back to the file
    with open(file_path, 'w') as file:
        file.writelines(remaining_urls)


# File containing the URLs
url_file = 'urls.txt'

# Read URLs from the file
urls = read_urls_from_file(url_file)

# Scrape and save content for each URL
for url in urls:
    title, content = scrape_content(url)
    success = save_to_database(url, title, content)
    if success:
        remove_url_from_file(url_file, url)

    # Slow down requests with sleep
    time.sleep(300)  # Sleep for 2 minutes
