# -*- coding: utf-8 -*-
import xlrd, json, string, random, os
from goose import Goose
from ftfy import fix_text

# File location
file_location = raw_input("File Path - ")
file_name = os.path.basename(file_location)
print file_name
json_file = ".".join(file_location.split(".")[:-1]) + \
	''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5)) + '.json'
print file_location
# Reading File
wb = xlrd.open_workbook(file_location)
sheet = wb.sheet_by_name("Final sheet")

fact_matrix = ['for', 'against']
quote_matrix = ['media', 'actor', 'celebrity']
all_matrix = fact_matrix + quote_matrix
subtopic_matrix = {}
links_matrix = []
link_req_data = {}

print "Finding Columns ....."
# Finding Columns
col_name = {'url': -1, 'tags': -1, 'quote': -1, 'text': -1}
for i in range(sheet.ncols):
	for j in col_name.keys():
		if j in sheet.cell_value(0, i).lower():
			col_name[j] = i

not_found_cols = filter(lambda x: col_name[x] == -1, col_name.keys())

if len(not_found_cols):
	print "Columns Not Found : ", not_found_cols
	exit()

for i in range(1, sheet.nrows):
	tags = sheet.cell_value(i, col_name['tags'])
	if 'delete' in tags:
		pass
	url = sheet.cell_value(i, col_name['url'])
	if url in link_req_data:
		 data = link_req_data[url]
	else:
		goose_client = Goose()
	        g_content = goose_client.extract(url = url)
		link_req_data[url] = {'title': g_content.title,
			'meta_description': g_content.meta_description,
			'image': g_content.top_image.src if g_content.top_image else '-',
			'video': g_content.movies[0].src if len(g_content.movies) else '-',
			'favicon': g_content.meta_favicon,
			'domain': g_content.domain}
		data = link_req_data[url]
	article_data = {'link': url,
		'story': data['title'],
                'meta_description': data['meta_description'],
		'tags': sheet.cell_value(i, col_name['tags']),
		'title': data['title'],
		'favicon': data['favicon'],
		'domain': data['domain']
        	}
	tags = map(lambda x: x.strip() ,article_data['tags'].split(","))
	link_data = {
		'link_title': data['title'],
		'meta_description': data['meta_description'],
		'tags': tags,
		'link_url': url,
		'link_type': 'na',
		'link_image': data['image'],
		'favicon': data['favicon'],
                'domain': data['domain']
		}
	print link_data
	# Link Type
	if data['video'] != '-':
		link_data['link_type'] = 'video'
		link_data['video_url'] = data['video']
	links_matrix.append(link_data)
	# Filter Subtopic
	subtopic = filter(lambda x: not x.lower() in fact_matrix and \
		not 'quote' in x.lower() and not 'fact' in x.lower(), tags)
	if not len(subtopic):
		print "No Subtopic found !"
		continue
	print subtopic
	if not subtopic[0] in subtopic_matrix.keys():
		print "New Subtopic"
		subtopic_matrix[subtopic[0]] = {"story" : subtopic[0],
			"against_links" : [], "for_links": []}
	left_tags = list(set(tags) - set(subtopic))
	print left_tags
	# Find Quote tag
	is_quote_tag = filter(lambda x: 'quote' in x.lower(), left_tags)
	if len(is_quote_tag):
		is_quote_tag = is_quote_tag[0]
		left_tags = list(set(left_tags) - set([is_quote_tag]))
		print "Quote"
		article_data['quote'] = fix_text(sheet.cell_value(i, col_name['quote']))
		# Finding Actor
		if 'actor' in is_quote_tag.lower() or 'celebrity' in is_quote_tag.lower():
			actor_text = fix_text(sheet.cell_value(i, col_name['text']))
			actor_text = actor_text.split("-")
			if len(actor_text):
				article_data['author'] = actor_text[-1]
			print "Author", article_data['author']
	else:
		# Fact
		print "Fact, No quotes !"
		article_data['fact'] = fix_text(sheet.cell_value(i, col_name['quote']))
	print left_tags
	for_tag = filter(lambda x: x.lower().strip() == 'for', left_tags)
	print for_tag
	if len(for_tag):
		print "For Argument"
		print subtopic
		print subtopic[0]
		subtopic_matrix[subtopic[0]]['for_links'].append(article_data)
	else:
		subtopic_matrix[subtopic[0]]['against_links'].append(article_data)
	

story_title = file_name.replace("_", " ").replace(".xlsx", "")

final_matrix = {
  "links" : [{
    "cat" : story_title,
    "links" : links_matrix}],
    "stories": [{'story': story_title,
    "sub-stories": subtopic_matrix.values()}]}

matrix_txt = json.dumps(final_matrix ,sort_keys=True, indent=4)

with open(json_file, 'w') as j_file:
    j_file.write(matrix_txt)

