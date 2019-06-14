# paragraph :
def art_to_para(article):
	para_list = article.split('\n\n')
	l = len(para_list)
	def para_thresholding(i=0, clustered_para=[]):
		temp = para_list[i]
		# print (i,' first')
		while len(temp.split(" ")) < threshold and i < l - 1:
			temp = temp + " " + para_list[i+1]
			i+=1
			# print("while ",i)
		clustered_para.append(temp) 
		# print(clustered_para)
		if i == l-1:
			return clustered_para
		else:
			return para_thresholding(i+1,clustered_para)

	return para_thresholding()
	