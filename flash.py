from tkinter import *
import pandas
import random

def quiz(words_2_learn):
    BACKGROUND_COLOR = "#B1DDC6"
    WORDS_PATH = "data/words_to_learn.csv"
    ORIGINAL_PATH = "data/french_words.csv"
    current_card = {}
    df = pandas.DataFrame(columns=['english_words', 'foreign_words', 'tries_remaining'])

    def import_words(words_2_learn=words_2_learn):
        global words_to_learn
        words_to_learn = words_2_learn

        vals = list(words_to_learn.values())
        words_to_learn = list(words_to_learn.keys())
        print(words_2_learn)
        return words_2_learn, vals


    def word_known():
        global words_to_learn, current_card
        words_to_learn = list(words_to_learn)
        words_to_learn.remove(current_card)
        data = pandas.DataFrame(words_to_learn)
        data.to_csv("data/words_to_learn.csv", index=False)
        print('known:' + current_card)
        if current_card in df['english_words']:
            df[df['english_words'] == current_card][2] - 1
        else:
            saved_word = (current_card, words_2_learn[current_card], 3)
        next_card()


    def word_unknown():
        global words_to_learn, current_card, words_2_learn
        words_to_learn = list(words_to_learn)
        words_to_learn.remove(current_card)
        print('unknown:' + current_card)
        saved_word = (current_card, words_2_learn[current_card], 3)
        df.loc[len(df)] = saved_word
        next_card()


    def next_card(vals):
        global flip_timer
        global words_to_learn
        global current_card
      #  window.after_cancel(flip_timer)
        try:
            words_to_learn = list(words_to_learn)
            current_card = random.choice(words_to_learn)
        except IndexError:
            flashcard.itemconfig(card_background, image=front_image)
            flashcard.itemconfig(language, text="Well done!", fill="black")
            flashcard.itemconfig(word, text="No more words to learn", fill="black", font=("Ariel", 40, "bold"))
            window.after(5000)
            window.quit()
        else:
            words_to_learn = dict(zip(words_to_learn, vals))
            flashcard.itemconfig(card_background, image=front_image)
            flashcard.itemconfig(word, text=words_to_learn[current_card], fill="black")
            flashcard.itemconfig(language, text="Arabic", fill="black")
            flip_timer = window.after(3000, func=flip_card)


    def flip_card():
        flashcard.itemconfig(card_background, image=back_image)
        flashcard.itemconfig(word, text=current_card, fill="black")
        flashcard.itemconfig(language, text="English", fill="black")


    window = Tk()
    window.minsize(height=526, width=800)
    window.title("Flashcards")
    window.config(padx=50, pady=50, bg=BACKGROUND_COLOR)

    flip_timer = window.after(3000, flip_card)

    front_image = PhotoImage(file="images/card_front.gif")
    back_image = PhotoImage(file="images/card_back.gif")

    right_image = PhotoImage(file="images/right.gif")
    wrong_image = PhotoImage(file="images/wrong.gif")

    flashcard = Canvas(width=800, height=526, bg=BACKGROUND_COLOR, highlightthickness=0)
    card_background = flashcard.create_image(400, 263, image=front_image)
    language = flashcard.create_text(400, 150, text="", fill="black", font=("Ariel", 40, "italic"))
    word = flashcard.create_text(400, 263, text="", fill="black", font=("Ariel", 60, "bold"))
    flashcard.grid(row=0, column=0, columnspan=2)

    unknown_button = Button(image=wrong_image, highlightthickness=0, command=word_unknown, width=350, height=350)
    unknown_button.grid(row=1, column=0)

    known_button = Button(image=right_image, highlightthickness=0, command=word_known, width=350, height=350)
    known_button.grid(row=1, column=1)

    words_2_learn, vals = import_words()

    next_card(vals)

    window.mainloop()

    return df