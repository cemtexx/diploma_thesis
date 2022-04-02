import os

# current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

print(current_dir)

current_dir = 'data/obj'

# percentage of images to be used for the test set
percentage_test = 10

# create and/or truncate train.txt and test.txt
file_train = open('data/train.txt', 'w')
file_test = open('data/test.txt', 'w')

# populate train.txt and test.txt
counter = 1
index_test = round(100 / percentage_test)
for filename in os.listdir("data"):
    if filename[-4:] == '.jpg':
        title = filename[:-4]
        if counter == index_test:
            counter = 1
            file_test.write("data" + "/" + title + '.jpg' + "\n")
        else:
            file_train.write("data" + "/" + title + '.jpg' + "\n")
            counter = counter + 1

file_test.close()
file_train.close()