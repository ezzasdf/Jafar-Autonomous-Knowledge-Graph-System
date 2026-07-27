import PyPDF2

reader = PyPDF2.PdfReader(r'E:\projects\Ai\books\Automate the Boring Stuff with Python.pdf')
start_page = 196
end_page = 219

text_lines = []
for i in range(start_page, end_page):
    page = reader.pages[i]
    raw = page.extract_text()
    text_lines.append(f'--- Page {i} ---')
    text_lines.append(raw)

with open(r'E:\projects\Ai\books\ch8_raw.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(text_lines))

print(f'Extracted pages {start_page} to {end_page - 1} ({end_page - start_page} pages)')
