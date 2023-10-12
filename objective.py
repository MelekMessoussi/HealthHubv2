import re
import nltk
import numpy as np
from nltk.corpus import wordnet as wn
import random

class ObjectiveTest:

    def __init__(self, data, noOfQues):
        
        self.summary = data
        self.noOfQues = noOfQues

    def get_trivial_sentences(self):
        sentences = nltk.sent_tokenize(self.summary)
        trivial_sentences = list()
        for sent in sentences:
            trivial = self.identify_trivial_sentences(sent)
            if trivial:
                trivial_sentences.append(trivial)
            else:
                continue
        return trivial_sentences

    def identify_trivial_sentences(self, sentence):
    # Tokenize the sentence into words
        words = nltk.word_tokenize(sentence)

        tags = nltk.pos_tag(words)

        noun_phrases = list()
        grammar = r"""
            CHUNK: {<NN>+<IN|DT>*<NN>+}
                {<NN>+<IN|DT>*<NNP>+}
                {<NNP>+<NNS>*}
        """
        chunker = nltk.RegexpParser(grammar)
        pos_tokens = nltk.tag.pos_tag(words)
        tree = chunker.parse(pos_tokens)

        for subtree in tree.subtrees():
            if subtree.label() == "CHUNK":
                temp = ""
                for sub in subtree:
                    temp += sub[0]
                    temp += " "
                temp = temp.strip()
                noun_phrases.append(temp)

        replace_nouns = []
        for word, _ in tags:
            for phrase in noun_phrases:
                if phrase[0] == '\'':
                    break
                if word in phrase:
                    [replace_nouns.append(phrase_word) for phrase_word in phrase.split()[-2:]]
                    break
            if len(replace_nouns) == 0:
                replace_nouns.append(word)
            break

        if len(replace_nouns) == 0:
            return None

        val = 99
        for i in replace_nouns:
            if len(i) < val:
                val = len(i)
            else:
                continue

        trivial = {
            "Answer": " ".join(replace_nouns),
            "Key": val
        }

        if len(replace_nouns) == 1:
            trivial["Similar"] = self.answer_options(replace_nouns[0])
        else:
            trivial["Similar"] = []

        replace_phrase = " ".join(replace_nouns)
        blanks_phrase = ("__________" * len(replace_nouns)).strip()
        expression = re.compile(re.escape(replace_phrase), re.IGNORECASE)
        sentence = expression.sub(blanks_phrase, str(sentence), count=1)
        trivial["Question"] = sentence

        return trivial



    @staticmethod
    def answer_options(word, num_options=8):
        synsets = wn.synsets(word, pos="n")

        if len(synsets) == 0:
            return []

        synset = synsets[0]
        hypernym = synset.hypernyms()[0]
        hyponyms = hypernym.hyponyms()

        # Filter out unwanted words (less than 5 characters)
        similar_words = [hyponym.lemmas()[0].name().replace("_", " ") for hyponym in hyponyms
                        if len(hyponym.lemmas()[0].name()) >= 5 and hyponym.lemmas()[0].name() != word]

        return similar_words[:num_options]


    def generate_test(self):
        trivial_pair = self.get_trivial_sentences()
        num_questions = min(int(self.noOfQues), len(trivial_pair))

        # Shuffle the questions to present them in a random order
        random.shuffle(trivial_pair)
        questions = []
        correct_answers = []
        user_answers = []
        score = 0

        for i, que_ans_dict in enumerate(trivial_pair[:num_questions]):
            question = que_ans_dict["Question"]
            answer = que_ans_dict["Answer"]

            # Store the question and correct answer
            questions.append(question)
            correct_answers.append(answer)

        return questions, correct_answers, user_answers, score