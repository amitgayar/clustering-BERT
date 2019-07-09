import urlparse, logging, string, xlsxwriter
import os
import pickle
import random
import time
from torrequest import TorRequest
import sys, json, requests
from goose import Goose

torpassword = 'truenews05101991'
#torpassword = '16:E29C6BCF1B6C8AB860E9B45DE1794ECFB940862DB7B924185B5A5ECA85'

from bs4 import BeautifulSoup


NUMBER_OF_CALLS_TO_GOOGLE_NEWS_ENDPOINT = 0

GOOGLE_NEWS_URL = 'https://www.google.co.jp/search?q={}&hl=eng&source=lnt&tbs=cdr%3A1%2Ccd_min%3A{}%2Ccd_max%3A{}&tbm=nws&start={}'

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


from stem import Signal
from stem.control import Controller

# signal TOR for a new connection 
def renew_connection():
    with Controller.from_port(port = 9051) as controller:
        controller.authenticate(password=torpassword)
        controller.signal(Signal.NEWNYM)

def get_tor_session():
    renew_connection()
    session = requests.session()
    # Tor uses the 9050 port as the default socks port
    session.proxies = {'http':  'socks5://127.0.0.1:9050',
                       'https': 'socks5://127.0.0.1:9050'}
    return session

# Goose Timeout
def g_timeout_handler(num, stack):
    print("Received SIGALRM")
    raise Exception("goose_timeout")

def run(keyword):
	headers = {'User-agent': "HotJava/1.1.2 FCS"}
        logging.debug('KEYWORD = {}'.format(keyword))
        #generate_articles(keyword)
	safe_keyword = "+".join(keyword.split(" "))
	link = "https://www.quora.com/search?q=%s"%safe_keyword
	session = get_tor_session()
        print(session.get("http://httpbin.org/ip").text)
	response = session.get(link, headers=headers, timeout=20)
	print response.status_code
	if response.status_code != 200:
		pass
	html = response.content
	soup = BeautifulSoup(html, 'html.parser')
	# q_data = {ques_link: [ques_text, [{
	#	external_link, link_text, answer_id, answer_text, 
	#	meta_keyword, meta_description, meta_title, image, video, favicon, domain}]]}
	q_data = {}
	print list(soup.find_all("a", {"class": "question_link"}))
	for i in soup.find_all("a", {"class": "question_link"}):
		print i
		print i.text, i['href']
		ques_link = "https://www.quora.com/%s"%i['href']
		ques_response = session.get(ques_link, headers=headers, timeout=20)
		if ques_response.status_code != 200:
			session = get_tor_session()
	                ques_response = session.get(ques_link, headers=headers, timeout=20)
		ques_html = ques_response.content
		ques_soup = BeautifulSoup(ques_html, 'html.parser')
		answers = ques_soup.find_all("div", {"class": "AnswerBase"})
		ans_links = ques_soup.find_all("span", {"class": "qlink_container"})
		print list(answers)
		if len(list(answers)) and len(list(ans_links)):
			q_data[i['href']] = {'text': i.text, 'links' : []}
		for ans in answers:
			print ans.text
			external_links = ans.find_all("span", {"class": "qlink_container"})
			for e_link in external_links:
				if len(list(e_link.children)):
					a_link = list(e_link.children)[0]
					if bool(urlparse.urlparse(a_link['href']).netloc):
						print a_link['href']
						link_url = a_link['href']
						if 'https://www.quora.com/_/redirect' in a_link['href']:
							try:
								link_url = filter(lambda y: y[0] == 'url', map(lambda x: x.split("="), urlparse.urlparse(link_url).query.split("&")))[0][1]
							except:
								print sys.exc_info()
						if len(urlparse.urlparse(link_url).path) < 3:
							pass
						signal.signal(signal.SIGALRM, g_timeout_handler)
						signal.alarm(20)
						try:
							# Got external link
							goose_client = Goose()
					                g_content = goose_client.extract(url = link_url)
				        	        q_data[i['href']]['links'].append({
							'title': g_content.title,
				                        'meta_description': g_content.meta_description,
                        				'image': g_content.top_image.src \
								if g_content.top_image else '-',
			                	        'video': g_content.movies[0].src \
								if len(g_content.movies) else '-',
	                        			'favicon': g_content.meta_favicon,
				                        'domain': g_content.domain,
							'a_link': a_link['href'],
							'a_link_text': a_link.text,
							'a_link_answer_id': ans.get('id'),
							'a_link_answer_text': ans.text,
							})
						except Exception as ex:
							if "goose_timeout" in ex:
								print "Goose Timeout!"
							else:
								print "New Error", ex
							q_data[i['href']]['links'].append({
							'a_link': a_link['href'],
                                                        'a_link_text': a_link.text,
                                                        'a_link_answer_id': ans.get('id'),
                                                        'a_link_answer_text': ans.text,
                                                        })
						finally:
							signal.alarm(0)
	json_file_n = safe_keyword + ''.join(random.choice(string.ascii_uppercase + string.digits) \
		for _ in range(5)) + '.json'
	with open(json_file_n, 'w') as json_file:
    		json.dump(q_data, json_file)
	return


def json_to_excel(file_name):
	with open('json_dump/%s'%file_name, 'r') as f:
	        file_content = json.loads(f.read())
	workbook = xlsxwriter.Workbook('%s.xlsx'%file_name.replace(".json", ""))
	worksheet = workbook.add_worksheet()
	headers = ['Question URL', 'Question Text', 'Answer ID', 'Answer Text', 'Link Text in Answer',
		'Link Domain', 'Link', 'Link Meta Title', 'Link Meta Description', 'Link Image',
		'Link Video', 'Favicon']
	row = 0
	col = 0
	for i in headers:
		worksheet.write(row, col, i)
		col += 1
	row = 1
	for i in file_content.keys():
		v = file_content[i]
		print v
		worksheet.write(row, 0, i)
		worksheet.write(row, 1, v['text'])
		for j in v['links']:
	                worksheet.write(row, 2, j['a_link_answer_id'])
        	        worksheet.write(row, 3, j['a_link_answer_text'])
                	worksheet.write(row, 4, j['a_link_text'])
	                worksheet.write(row, 5, j['domain'])
        	        worksheet.write(row, 6, j['a_link'])
                	worksheet.write(row, 7, j['title'])
	                worksheet.write(row, 8, j['meta_description'])
        	        worksheet.write(row, 9, j['image'])
	                worksheet.write(row, 10, j['video'])
        	        worksheet.write(row, 11, j['favicon'])
                	row += 1
	workbook.close()
	return


if __name__ == '__main__':
	option_sel = raw_input('Enter -> 1. Search, 2. Data to excel')
	if option_sel == 1:
	        keyword = raw_input('keyword : ')
		try:
	                run(keyword)
        	except:
                	print sys.exc_info()
	else :
		file_name = raw_input('File : ')
		if 1:#try:
			json_to_excel(file_name)
		#except:
		#	print sys.exc_info()
		
			


