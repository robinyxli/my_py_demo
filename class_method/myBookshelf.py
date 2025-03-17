import logging
logging.basicConfig(level=logging.INFO, filename='log.txt')
import datetime
import csv

def logger_add(func):
    def wrapper(*args, **kwargs):
        logging.info("Book Name: '{0}', Book Added to Shelf: '{1}'".
                     format(args[1], datetime.datetime.now()))
        return func(*args, **kwargs)
    return wrapper

def logger_remove(func):
    def wrapper(*args, **kwargs):
        logging.info("Book Name: '{0}', Book Removed to Shelf: '{1}'".
                     format(args[1], datetime.datetime.now()))
        return func(*args, **kwargs)
    return wrapper

class bookShelf:

    def __init__(self):
        self.stack = {}

    def empty_stack(self):
        if len(self.stack) == 0:
            return True
        else:
            return False

    @classmethod
    def load_from_csv(cls, file_name):
        try:
            with open(file_name, mode='r') as f:
                reader = csv.reader(f)
                return dict((x[0], x[1]) for x in reader)
        except FileNotFoundError:
            return dict()

    @logger_add
    def push(self, title, status):
        self.stack[title] = status

    @logger_remove
    def pop(self, title):
        if self.empty_stack():
            print('Empty BookShelf!')
        else:
            self.stack.pop(title)

    def access_status(self, title):
        try:
            return self.stack[title]
        except KeyError:
            ans = input(str(title) + ' Not Found! Would You Like to Add to the BookShelf? y/n: ')
            while ans not in ['y', 'Y', 'n', 'N']:
                ans = input('Your Answer is Invalid! Please answer Y/y/N/n: ')

            if ans == 'y' or ans == 'Y':
                status = input('Have you Read this Book? Read/Unread: ')
                while status not in ['Read', 'Unread']:
                    status = input('Your Answer is Invalid! Please answer Read/Unread: ')
                self.push(title, status)
                return 'Book and Status Added!'
            elif ans == 'n' or ans == 'N':
                return 'GoodBye!'

    def change_status(self, title):
        try:
            status = self.stack[title]
            if status == 'Unread':
                self.stack[title] = 'Read'
                print(str(title) + ' Has Changed to Read!')
            elif status == 'Read':
                self.stack[title] = 'Unread'
                print(str(title) + ' Has Changed to Unread!')
        except KeyError:
            print(str(title) + ' Not Found!')

def update_dict(file_name):
    book = bookShelf()
    prev_ls = book.load_from_csv(file_name)
    book.stack = prev_ls
    book.push('Gone with the Wind', 'Read')
    book.push('Wuthering Heights', 'Unread')
    book.pop('Gone with the Wind')
    print(book.stack)
    book.access_status('Gone with the Wind')
    book.change_status('Emma')
    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        for k, v in book.stack.items():
            writer.writerow([k,v])

if __name__ == "__main__":
    file_name = 'books.csv'
    update_dict(file_name)
