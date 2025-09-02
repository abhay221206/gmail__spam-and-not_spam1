import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import tkinter as tk



df = pd.read_csv(r"mail_data.csv")
df.loc[df['Category'] == 'ham', 'Category'] = 0
df.loc[df['Category'] == 'spam', 'Category'] = 1

y = df['Category']
x = df['Message']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


trans = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
x_train = trans.fit_transform(x_train)
x_test = trans.transform(x_test)

y_train = y_train.astype('int')
y_test = y_test.astype('int')


model = LogisticRegression()
model.fit(x_train, y_train)

predict = model.predict(x_train)
accuracy_data = accuracy_score(y_train, predict)
print("Training Accuracy:", accuracy_data * 100)


root = tk.Tk()
root.title('Email Spam Detector')
root.geometry('800x400')

label = tk.Label(root, text='Enter Email Text', font=("Arial", 14))
label.pack()

data_input = tk.Entry(root, width=60, font=("Arial", 14), fg="black", bg="white")
data_input.pack(padx=20, pady=20)

def show():
    data = data_input.get()
    if data.strip() == "":
        result_label.config(text="Please enter some text!", fg="orange")
        return
    
    vectorized = trans.transform([data])  
    p = model.predict(vectorized)
    
    if p[0] == 1:
        result = "Spam Email hai"
    else:
        result = "Not Spam hai"
    
    print(result)
    result_label.config(text=result, fg="red" if p[0] == 1 else "green")
    
 
    data_input.delete(0, tk.END)

button = tk.Button(root, text='Check', font=("Arial", 12), fg="black", bg="white", width=15, command=show)
button.pack(padx=20, pady=5)

result_label = tk.Label(root, text="", font=("Arial", 16))
result_label.pack(pady=20)

root.mainloop()
