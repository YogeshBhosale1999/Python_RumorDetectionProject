import pandas as pd, numpy as np, re

from sklearn.metrics import classification_report, accuracy_score , confusion_matrix
from sklearn.model_selection import train_test_split
import tkinter as tk
from sklearn import svm
from PIL import Image, ImageTk
from tkinter import ttk
from joblib import dump , load
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import pickle
import nltk

# Text may contain stop words like ‘the’, ‘is’, ‘are’
nltk.download('stopwords')
stop = stopwords.words('english')

# Punkt Sentence Tokenizer
# This tokenizer divides a text into a list of sentences
nltk.download('punkt')

# punkt is used for tokenising sentences and
# averaged_perceptron_tagger is used for tagging words with their parts of speech (POS)
nltk.download('averaged_perceptron_tagger')

# Create a root pane window
root = tk.Tk()
root.title("Rumor Detection System")

w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
image2 =Image.open(r'E:\rumour Detection\BG.jpg')
image2 =image2.resize((w,h), Image.ANTIALIAS)

background_image=ImageTk.PhotoImage(image2)

background_label = tk.Label(root, image=background_image)

background_label.image = background_image

background_label.place(x=0, y=0)


label_l1 = tk.Label(root, text="Rumor Detection System",font=("Times New Roman", 35, 'bold'),
                    background="#152238", fg="white", width=25)
label_l1.place(x=300, y=10)




frame = tk.LabelFrame(root,text="Rumor Detector",width=300,height=400,bd=3,background="cyan2",font=("Tempus Sanc ITC",15,"bold"))
frame.place(x=30,y=100)
frame['borderwidth'] = 10

#--------------------------------------------------------------
def Data_Display():
    columns = ['TID', 'Tweets', 'Label']
    print(columns)

    data1 = pd.read_csv(r"E:\rumour Detection\Tweet_Dataset.csv", encoding='unicode_escape')

    data1.shape

    data1.shape

    data1.head()

    data1

    data1

    TID = data1.iloc[:, 0]
    Tweets = data1.iloc[:, 1]
    Label = data1.iloc[:, 2]


    display = tk.LabelFrame(root, width=300, height=400, )
    display.place(x=700, y=100)
    display['borderwidth'] = 15

    tree = ttk.Treeview(display, columns=(
    'TID', 'Tweets', 'Label'))

    style = ttk.Style()
    style.configure('Treeview', rowheight=50)
    style.configure("Treeview.Heading", font=("Tempus Sans ITC", 15, "bold italic"))
    style.configure(".", font=('Helvetica', 15), background="blue")
    style.configure("Treeview", foreground='white', background="black")

    tree["columns"] = ("1", "2", "3")
    tree.column("1", width=100)
    tree.column("2", width=400)
    tree.column("3", width=100)

    tree.heading("1", text="TID")
    tree.heading("2", text="Tweets")
    tree.heading("3", text="Label")

    #treeview = tree

    tree.grid(row=0, column=0, sticky=tk.NSEW)

    print("Data Displayed")

    for i in range(0, 2981):
        tree.insert("", 'end', values=(
        TID[i], Tweets[i], Label[i]))
        i = i + 1
        print(i)




def Train():

    result = pd.read_csv(r"E:\rumour Detection\Tweet_Dataset.csv",encoding = 'unicode_escape')

    result.head()

    result['headline_without_stopwords'] = result['Tweets'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))


    def pos(headline_without_stopwords):
        return TextBlob(headline_without_stopwords).tags


    os = result.headline_without_stopwords.apply(pos)
    os1 = pd.DataFrame(os)
    #
    os1.head()

    os1['pos'] = os1['headline_without_stopwords'].map(lambda x: " ".join(["/".join(x) for x in x]))

    result = result = pd.merge(result, os1, right_index=True, left_index=True)
    result.head()
    result['pos']
    result_train, result_test, label_train, label_test = train_test_split(result['pos'], result['Label'],
                                                                              test_size=0.2, random_state=1)

    # TF-IDF : “Term Frequency – Inverse Document”
    # tf_vect = TfidfVectorizer(lowercase=True, use_idf=True, smooth_idf=True, sublinear_tf=False)
    tf_vect = TfidfVectorizer(lowercase=True, use_idf=True)

    X_train_tf = tf_vect.fit_transform(result_train)
    X_test_tf = tf_vect.transform(result_test)


    def svc_param_selection(X, y, nfolds):
        Cs = [0.001, 0.01, 0.1, 1, 10]
        gammas = [0.001, 0.01, 0.1, 1]
        param_grid = {'C': Cs, 'gamma': gammas}
        grid_search = GridSearchCV(svm.SVC(kernel='linear'), param_grid, cv=nfolds)
        grid_search.fit(X, y)
        return grid_search.best_params_


    svc_param_selection(X_train_tf, label_train, 5)
    #

    clf = svm.SVC(C=10, gamma=0.001, kernel='linear')
    clf.fit(X_train_tf, label_train)
    pred = clf.predict(X_test_tf)

    with open('vectorizer.pickle', 'wb') as fin:
        pickle.dump(tf_vect, fin)
    with open('mlmodel.pickle', 'wb') as f:
        pickle.dump(clf, f)

    pkl = open('mlmodel.pickle', 'rb')
    clf = pickle.load(pkl)
    vec = open('vectorizer.pickle', 'rb')
    tf_vect = pickle.load(vec)

    X_test_tf = tf_vect.transform(result_test)
    pred = clf.predict(X_test_tf)

    print(metrics.accuracy_score(label_test, pred))

    print(confusion_matrix(label_test, pred))

    print(classification_report(label_test, pred))


    print("=" * 40)
    print("==========")
    print("Classification Report : ",(classification_report(label_test, pred)))
    print("Accuracy : ",accuracy_score(label_test, pred)*100)
    accuracy = accuracy_score(label_test, pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    ACC = (accuracy_score(label_test, pred) * 100)
    repo = (classification_report(label_test, pred))

    label4 = tk.Label(root,text =str(repo),width=35,height=10,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label4.place(x=300,y=100)

    label5 = tk.Label(root,text ="Accuracy : "+str(ACC)+"%\nModel saved as SVM_MODEL.joblib",width=35,height=3,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label5.place(x=300,y=320)

    dump (clf,"SVM_MODEL.joblib")
    print("Model saved as SVM_MODEL.joblib")



entry = tk.Entry(frame,width=20,font=("Tempus Sanc ITC",14))
entry.insert(0,"Enter text here...")
entry.place(x=10,y=150)

def Test():
    predictor = load("SVM_MODEL.joblib")
    Given_text = entry.get()
    #Given_text = "the 'roseanne' revival catches up to our thorny po..."
    vec = open('vectorizer.pickle', 'rb')
    tf_vect = pickle.load(vec)
    X_test_tf = tf_vect.transform([Given_text])
    y_predict = predictor.predict(X_test_tf)
    print(y_predict[0])
    if y_predict[0]==0:
        label4 = tk.Label(root,text ="True Rumor Tweet",width=35,height=2,bg='Green',fg='black',font=("Tempus Sanc ITC",14))
        label4.place(x=50,y=550)
        label4['borderwidth'] = 10
    elif y_predict[0]==1:
        label4 = tk.Label(root,text ="False Rumor Tweet ",width=35,height=2,bg='Red',fg='black',font=("Tempus Sanc ITC",14))
        label4.place(x=50,y=550)
        label4['borderwidth'] = 10
    elif y_predict[0]==2:
        label4 = tk.Label(root,text ="Unverified Tweet Detected",width=35,height=2,bg='Red',fg='black',font=("Tempus Sanc ITC",14))
        label4.place(x=50,y=550)
        label4['borderwidth'] = 10
    elif y_predict[0]==3:
        label4 = tk.Label(root,text ="Non-Rumor Tweet Detected",width=35,height=2,bg='Red',fg='black',font=("Tempus Sanc ITC",14))
        label4.place(x=50,y=550)
        label4['borderwidth'] = 10

def window():
  root.destroy()


button1 = tk.Button(frame,command=Data_Display,text="Data_Display",bg="gold",fg="black",width=15,font=("Times New Roman",15,"italic"))
button1.place(x=25,y=50)

button2 = tk.Button(frame,command=Train,text="Train",bg="red",fg="black",width=15,font=("Times New Roman",15,"italic"))
button2.place(x=25,y=100)

button3 = tk.Button(frame,command=Test,text="Test",bg="red",fg="black",width=15,font=("Times New Roman",15,"italic"))
button3.place(x=25,y=200)

button4 = tk.Button(frame,command=window,text="Exit",bg="red",fg="black",width=15,font=("Times New Roman",15,"italic"))
button4.place(x=25,y=250)
root.mainloop()