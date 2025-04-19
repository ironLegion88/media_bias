import csv
from docx import Document

def read_links_from_docx(docx_path):
    document = Document(docx_path)
    links = []
    for para in document.paragraphs:
        text = para.text.strip()
        if text:
            links.append(text)
    return links

def append_links_to_csv(csv_path, links, outlet, category):
    with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for link in links:
            writer.writerow([outlet, category, link])

# Paths to the files
docx_path = './src/article-links.docx'
csv_path = './src/article-links.csv'

# Read links from the .docx file
links = read_links_from_docx(docx_path)

# Append links to the .csv file
append_links_to_csv(csv_path, links, 'IndianExpress', 'column')

print("Links have been successfully added to the CSV file.")