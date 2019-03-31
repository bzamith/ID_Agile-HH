from tkinter import *
import argparse #Import Argsparse
import ai

class Interface:
	def __init__(self, master=None):
		self.data = ai.read_dataset(file_name)
		self.count_vect, self.tf_transformer, self.clf = ai.nlp(self.data)
		self.saved_phrase = "."
		
		self.fontePadrao = ("Arial", "10")
		self.first_container = Frame(master)
		self.first_container["pady"] = 10
		self.first_container.pack()
  
		self.second_container = Frame(master)
		self.second_container["padx"] = 20
		self.second_container.pack()
  
		self.third_container = Frame(master)
		self.third_container["padx"] = 20
		self.third_container.pack()
  
		self.fourth_container = Frame(master)
		self.fourth_container["pady"] = 20
		self.fourth_container.pack()
  
		self.title = Label(self.first_container, text="What are you feeling?")
		self.title["font"] = ("Arial", "10", "bold")
		self.title.pack()
  
		self.phrase = Entry(self.second_container)
		self.phrase["width"] = 50
		self.phrase["font"] = self.fontePadrao
		self.phrase.pack(side=LEFT)
    
		self.send = Button(self.fourth_container)
		self.send["text"] = "Send"
		self.send["font"] = ("Calibri", "8")
		self.send["width"] = 12
		self.send["command"] = self.send_phrase
		self.send.pack()
  
		self.message = Label(self.fourth_container, text="", font=self.fontePadrao)
		self.message.pack()
  
	#Método verificar senha
	def send_phrase(self):
		self.saved_phrase = self.phrase.get()
		especialist = ai.predict_instance(self.saved_phrase,self.count_vect,self.tf_transformer,self.clf)
		self.message["text"] = especialist[0]

#Main Function
if __name__ == '__main__':
	#Args parse to get file name
	parser = argparse.ArgumentParser(description='Please provide the dataset file name', add_help=True)
	parser.add_argument('-i','--input', dest='inputFile', metavar='inputFile', type=str, help='Dataset', required=True)
	args = parser.parse_args()
	file_name = args.inputFile

	#Execution
	print()
	root = Tk()
	Interface(root)
	root.mainloop()
  
  
