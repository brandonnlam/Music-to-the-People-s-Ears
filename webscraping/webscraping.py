from bs4 import BeautifulSoup as bs4
import codecs
from urllib.request import Request, urlopen


class Soup: #why did i make this
	def __init__(self,html):
		self.html = html
#To Use, 
#1. comment out lupe variable
#2. put your url into urlpage variable and uncomment
#3. uncomment req variable and souppage variable
#4. make soupstring = str(souppage) instead of lupe
#5. voila bitch
def main():
	# lupe = bs4(codecs.open("b.html",'r'),'html.parser') #a.html from the same folder
	urlpage = 'https://www.lyricsfreak.com/e/eminem/renegade_20308049.html'
	req = Request(urlpage, headers={'User-Agent': 'Mozilla/5.0'})  #the block all humans
	souppage = bs4(urlopen(req).read(),'html.parser') #so many different variations
	soupstring=str(souppage)
	wordlist = []
	count=0
	flag=False
	superflag=False
	word = ""
	tagflag=False
	bracketflag=False
	for char in soupstring:
		if superflag==True:#if it should be recording
			if tagflag==True or bracketflag==True:
				if word == '<!--':
					break
				else:
					if char =='>':
						tagflag=False
						word=""
						pass
					if char==']':
						breakflag=False
						word=""
						pass
			elif char == '<':
				if len(word)>0:
					wordlist.append(word)
					word=""
				tagflag=True
			elif char=='[':
				if len(word)>0:
					wordlist.append(word)
					word=""
				breakflag=True
			elif char==' ':
				if len(word)>0:
					wordlist.append(word)
					word=""
			if char!='>':
				word=word+char
		
##########################################################################
		elif flag==True:#if it should be recording for song comment
			word=word+char
			if char=='>':
				if word == "<!-- SONG LYRICS -->":
					superflag=True# start the list recording
					word=""
				else:#start over to look for <
					word=""
					flag=False
			else:
				pass
		elif char=='<': #if this starts to become     <!-- SONG LYRICS -->
			flag=True
			word=word+char
		else:
			pass
	print(wordlist)
	# print(codecs.open("b.html",'r'))
	# print(souppage.div[''])
	# for line in souppage.find_all('div'):
	# 	if souppage.div['id'] == 'content_h':
	# 		print(line)
	# for page in souppage.find_all('a'):
	# 	for tag in page:#for every tag in page
	# 		# print(tag)
	# 		# print("hello")
	# 		for char in tag:
	# 			# print(char)
	# 			if char=='n':
	# 				pass
	# 			if char==' ' or char== '':
	# 				if len(word)!=0:
	# 					wordlist.append(word)
	# 					word=""
	# 			elif len(char)==1:
	# 				word= word + char
	# 				pass
	# 			else:
	# 				wordlist.append(char)
	# 				# print(char)
	# 				# print("hello")
	# 				pass
	# for elem in wordlist:
	# 	if elem=='\n':
	# 		wordlist.remove(elem)
	# print(wordlist)
	# print(souppage.prettify())
	# print(souppage.find_all('a'))

main()