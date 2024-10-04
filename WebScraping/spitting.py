import os
import requests
from bs4 import BeautifulSoup
import base64
import urllib.parse

def download_images(search_term, num_images=10):
    # Create a directory to save images
    save_dir = f"{search_term}_images"
    os.makedirs(save_dir, exist_ok=True)

    # Google Images search URL
    search_url = f"https://www.google.com/search?hl=en&tbm=isch&q={urllib.parse.quote(search_term)}"
    
    # Fetch the search results page
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    response = requests.get(search_url, headers=headers)
    
    if response.status_code != 200:
        print(f"Failed to retrieve images. Status code: {response.status_code}")
        return

    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')
    images = soup.find_all('img')

    downloaded_count = 0
    for i, img in enumerate(images):
        if downloaded_count >= num_images:
            break
        
        # Attempt to get the 'src' attribute
        img_url = img.get('src')
        
        # Debug: print the image URL
        print(f"Image URL: {img_url}")

        if img_url:
            try:
                if img_url.startswith('data:image/'):
                    # Handle base64 data
                    head, base64_data = img_url.split(',', 1)
                    img_data = base64.b64decode(base64_data)
                    # Define the image file name
                    img_filename = os.path.join(save_dir, f"{search_term}_{downloaded_count + 1}.jpg")
                else:
                    # Download the image from the URL
                    response = requests.get(img_url)
                    response.raise_for_status()  # Raise an error for bad responses
                    img_data = response.content
                    # Define the image file name
                    img_filename = os.path.join(save_dir, f"{search_term}_{downloaded_count + 1}.jpg")

                # Ensure the filename is unique
                if os.path.exists(img_filename):
                    base, ext = os.path.splitext(img_filename)
                    count = 1
                    while os.path.exists(img_filename):
                        img_filename = f"{base}_{count}{ext}"
                        count += 1

                # Save the image
                with open(img_filename, 'wb') as f:
                    if img_url.startswith('data:image/'):
                        f.write(img_data)
                        print(f"Downloaded image {downloaded_count + 1} from base64 data")
                    else:
                        f.write(img_data)
                        print(f"Downloaded image {downloaded_count + 1} from {img_url}")

                downloaded_count += 1
            except Exception as e:
                print(f"Could not download image {downloaded_count + 1}: {e}")
        else:
            print(f"Could not retrieve image URL for image {downloaded_count + 1}")

    print(f"Downloaded {downloaded_count} images.")

if _name_ == "_main_":
    spit = ["Human spitting behavior",
    "Humans spitting on the ground",
    "Human spit fight",
    "Humans spitting in public",
    "Human tobacco spitting",
    "Human spitting during confrontation",
    "Humans spitting in sports",
    "Human spitting match",
    "Human spitting incident",
    "Humans spitting at each other",
    "Humans spitting during arguments",
    "Human spitting scandal",
    "Human spitting on the street",
    "Human spitting insult",
    "Humans spitting for distance",
    "mouth water fight spit",
    "football spit",
    "human paan spit",
    "human spit",
    "mouth water fight spit",
    "spit football",
    "spit sneeze",
    "spitting on road",
    "spitting",
    "sspitting water",
    "water spit",
    "spit on roadside india",
    "spit at monument india",
    "spit in trains"]
    for i in spit:
        download_images(i, num_images=20)  # Change num_images as needed
