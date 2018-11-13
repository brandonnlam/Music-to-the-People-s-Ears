from bs4 import BeautifulSoup as bs4
import codecs
from urllib.request import Request, urlopen


class Soup:
	def __init__(self,html):
		self.html = html

def main():
	lupe = bs4(codecs.open("a.html",'r'),'html.parser')

	urlpage = 'https://genius.com/Jay-z-renegade-lyrics'
	req = Request(urlpage, headers={'User-Agent': 'Mozilla/5.0'}) 
	souppage = bs4(urlopen(req).read(),'html.parser')
	print(souppage.prettify())
	
main()