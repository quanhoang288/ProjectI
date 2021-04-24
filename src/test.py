from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import sys
import time
import os
import unicodedata
import re
from tqdm import tqdm 
# if len(sys.argv) > 1:
#     src_file = sys.argv[1]
# else:
#     src_file = 'km_to_txt.txt'
# src_files = './data_khm_samples.txt'
# trans_files = './data_vi_samples_trans.txt'

# src_files = './km_100_sens.txt'
# trans_files = './vi_100_sens.txt'

def create_driver():
    chrome_options = Options()  
    chrome_options.add_argument("--headless") 
    driver = webdriver.Chrome(os.path.abspath('./chromedriver'), chrome_options=chrome_options)
    return driver 

# now = time.time()
prev_text = ''
def translate_doc(src_file, trans_file):
	driver = create_driver()
	# driver = webdriver.Chrome('./chromedriver.exe')
	driver.get('https://translate.google.com/#view=home&op=translate&sl=km&tl=vi')
	time.sleep(2)


	def translate(text):
		global prev_text
		def find_trans_box(driver):
			print('finding trans element...')
			element = driver.find_element_by_xpath('//div[@class="J0lOec"]')
			if element and element.text == prev_text:
				return False
			if element and element.text != prev_text:
				return element
			else:
				return False

		def find_src_box(driver):
			print('finding source element...')
			element = driver.find_element_by_tag_name('textarea')
			if element:
				return element
			else:
				return False

		src_text = WebDriverWait(driver, 50).until(find_src_box)
		src_text.clear()
		src_text.send_keys(text)
		time.sleep(2)
		# WebDriverWait(driver, 50).until(EC.presence_of_element_located((By.XPATH, '//div[@class="text-wrap tlid-copy-target"]')))
		dst_text = WebDriverWait(driver, 50).until(find_trans_box)
		prev_text = dst_text.text
		time.sleep(1)
		# dst_text = driver.find_element_by_xpath('//div[@class="text-wrap tlid-copy-target"]')
		
		return prev_text
	# with open(trans_file,'a+', encoding='utf-8') as f:
	# 	# f.seek(0)
	# 	# n_processed = len(f.readlines())//2
	# 	# print(n_processed,"sens have been processed")
	# 	# f.seek(0, os.SEEK_END)
	# 	for i,line in enumerate(open(src_file, encoding='utf-8')):
	# 		# if i < n_processed:
	# 		# 	continue
	# 		if i > 745:
	# 			line = line.strip()
	# 			### Remove emoji icon that's not accepted by ChromeDriver
	# 			line = ''.join(c for c in unicodedata.normalize('NFC', line) if c <= '\uFFFF')
	# 			#f.write(line+'\n'+translate(line)+'\n')
	# 			print('current line: ' + line)
	# 			trans_line = translate(line)
	# 			trans_line = re.sub('\n', '', trans_line)
	# 			f.write(trans_line + '\n')

	with open(trans_file,'w', encoding='utf-8') as f:
		# f.seek(0)
		# n_processed = len(f.readlines())//2
		# print(n_processed,"sens have been processed")
		# f.seek(0, os.SEEK_END)
		with open(src_file, encoding='utf-8') as src:
			lines = src.readlines() 
			for i in tqdm(range(0, len(lines), 30)):
				src_lines = '' 
				chunk = lines[i: min(i + 30, len(lines))]
				for line in chunk:
					line = line.strip()
					line = ''.join(c for c in unicodedata.normalize('NFC', line) if c <= '\uFFFF')
					src_lines += line + '\n'
				f.write(translate(src_lines) + '\n')
		driver.close()
		# for i,line in enumerate(open(src_file, encoding='utf-8')):
		# 	# if i < n_processed:
		# 	# 	continue
		# 	if i > 745:
		# 		line = line.strip()
		# 		### Remove emoji icon that's not accepted by ChromeDriver
		# 		line = ''.join(c for c in unicodedata.normalize('NFC', line) if c <= '\uFFFF')
		# 		#f.write(line+'\n'+translate(line)+'\n')
		# 		print('current line: ' + line)
		# 		trans_line = translate(line)
		# 		trans_line = re.sub('\n', '', trans_line)
		# 		f.write(trans_line + '\n')

# translate_doc(src_files, trans_files)
# print('Done in %.1f secs'%(time.time() - now))


