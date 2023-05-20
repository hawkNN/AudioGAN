# Collect audio files from American Rhetoric Collection,
# 'https://www.americanrhetoric.com/' 

# url_target = 'https://www.americanrhetoric.com/newtop100speeches.htm'
# get_speeches(url_target, source_dir)

import os
import requests
from bs4 import BeautifulSoup
import argparse
import urllib.parse

def mp3_strip(url,headers,dest_dir):
  # Pulls mp3 files from url & copies to dest_dir
  # get html data, parse with soup
  try:
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')

    # source field contains audio link for speech bank, "https://www.americanrhetoric.com/"
    if soup.source:
      src = soup.source.get('src')
      if src and src.endswith('.mp3'):
        print (f'  source is: {src}')
        link_base = soup.a.get('href') # "https://www.americanrhetoric.com/"
        full_audio_link = urllib.parse.urljoin(link_base,src)
        # Download the audio files
        # append audio_link with url base 
        response = requests.get(full_audio_link, headers=headers)
        if response.status_code == 200:
          # define write to file
        #   audio_file = urllib.parse.urlsplit(src). #src.split('/') # mp3 name
          audio_filename = src.split('/')[-1] # mp3 name
          dest_path = os.path.join(dest_dir,audio_filename) # write destination
          print(f'  destination path is {dest_path}')
          # write from response.content to dest_path
          with open(dest_path, 'wb') as f:
              f.write(response.content)
              print(f'  Downloaded {src} to {dest_path}')
          return 1
        else:
          print(f'Error downloading audio file: {response.status_code}')
          return 0
      return 0

    else:
      print(f'  no mp3 files in {url}')
      return 0
  except:
    print(f'request failed for {url}')

def mp3_probe(url, 
              headers, 
              write_dir,
              max_files = None):
  # Looks for mp3 files one layer down from url, where they are for am rhet site
  mp3_count = 0 # used to limit total files downloaded to less than max_files
  response = requests.get(url, headers=headers)

  if response.status_code == 200:
    print(f'Finding links in {url}')
    soup = BeautifulSoup(response.content, 'html.parser')
    links = soup.find_all('a')
  else: 
    print(f'failed to open {url}')
    links = []
  # Only probing 1 layer deeper, because this could get infinite!!
  for link in links:
    url_end = link.get('href')
    full_url = urllib.parse.urljoin(url,url_end)
    print(f'stripping: {full_url}')
    status = mp3_strip(full_url, headers, write_dir) # pull out mp3 files from link
    if not status:
      print(f'  no mp3 files found')
    if max_files: # Using limiter on downloaded file count, mp3_count
        if status: 
          mp3_count+=1 # increment
          print(f'  {mp3_count} files acquired')
        if mp3_count >= max_files:
          return mp3_count  # break
  return mp3_count

def get_speeches(url_target, 
                 dest_dir,
                 headers = {'User-Agent': 'soundBot/0.1 (hawk.nervenet@gmail.com)'},
                 max_files = None):
  # pulls links in page to look for separate mp3 pages, 
  url_parts = urllib.parse.urlsplit(url_target)
  response = requests.get(url_target, headers=headers)
  soup = BeautifulSoup(response.content, 'html.parser')
  links = soup.find_all('a')
  # links contains all the links as href =*.htm
  # open these to look for mp3... 
  for link in links:
    url_end = link.get('href')
    full_url = urllib.parse.urljoin(url_target,url_end)
    print(f'probing {full_url}')
    mp3_count = mp3_probe(full_url, headers, dest_dir, max_files) 

def main():
  
  parser = argparse.ArgumentParser()
  parser.add_argument('--target_url', default='https://www.americanrhetoric.com/newtop100speeches.htm')
  parser.add_argument('--max_files', default=10)
  parser.add_argument('--output_dir', default='speech_audio')

  a = parser.parse_args()

  print(f'Downloading mp3 files from {a.target_url}')
  print(f'Download limit {a.max_files} files')
  print(f'Saving in {a.output_dir}')
  
  if not os.path.exists(a.output_dir):
    os.mkdir(a.output_dir)

  get_speeches(a.target_url, a.output_dir, max_files = a.max_files)

if __name__ == "__main__":
  main()