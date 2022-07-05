
import model
from tkinter import *

# import filedialog module 
from tkinter import filedialog
import os

# Function for opening the
# file explorer window
from PIL import ImageTk, Image
import threading

filename = ''


def browseFiles():
    global filename
    filename = filedialog.askopenfilename(initialdir="/",
                                          title="Select a File",
                                          filetypes=(("JPEG file",
                                                      "*.jpg*"),
                                                     ("all files",
                                                      "*.*")))
    img = ImageTk.PhotoImage(Image.open(filename).resize((500, 500)), Image.ANTIALIAS)
    lbl.configure(image=img)
    lbl.image = img


def checkBreed():  # handler for check breed, calls function predict from "model" file
    if filename == '':
        breedLbl.configure(text='please choose a file first')
    else:
        print(filename)
        breedLbl.configure(text='Scanning...')
        breed = model.predict(filename)

        breed = breed.split("-")
        breedLbl.configure(text=breed[1].split("_"))


def check_breed_pressed():  # function run a thread with handler for check breed
    threading.Thread(target=checkBreed).start()


def init_model():  # function changes label and call initiate() function from "model" file
    breedLbl.configure(text='Loading Scanner...')
    model.initiate()
    breedLbl.configure(text='Ready')


# Create the root window
window = Tk()

# Set window title 
window.title('Dog Scanner')

# Set window size 
window.geometry("500x620")

# Set window background color
window.config(background="white")

# Create buttons
button_explore = Button(window,
                        text="Browse Files",
                        command=browseFiles)

button_exit = Button(window,
                     text="Exit",
                     command=exit)

button_breed = Button(window,
                      text="Check Breed",
                      command=check_breed_pressed)

button_explore.grid(column=0, row=2)
button_breed.grid(column=0, row=3)
button_exit.grid(column=0, row=4)

# Create label
lbl = Label(window, width=500, height=500)
breedLbl = Label(window, text="", font=("Courier", 20), bg='#ffffff')


# Add initial picture
img = ImageTk.PhotoImage(Image.open('dog_scanner.jpg'))
lbl.configure(image=img)
lbl.image = img
breedLbl.grid(column=0, row=0)
lbl.grid(column=0, row=1)

# load the MobileNet model
threading.Thread(target=init_model).start()

# window main loop
window.mainloop()

