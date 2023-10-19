import numpy as np
import nltk
import numpy as np
from nltk.corpus import wordnet as wn
import random

class SubjectiveTest:
    def __init__(self, data, noOfQues):
        self.question_pattern = [
            "Explain in detail ",
            "Define ",
            "Write a short note on ",
            "What do you mean by "
        ]

        self.grammar = r"""
            CHUNK: {<NN>+<IN|DT>*<NN>+}
            {<NN>+<IN|DT>*<NNP>+}
            {<NNP>+<NNS>*}
        """
        self.summary = data
        self.noOfQues = int(noOfQues)
        self.generated_questions = set()  # Store generated questions to ensure uniqueness

    @staticmethod
    def word_tokenizer(sequence):
        word_tokens = list()
        for sent in nltk.sent_tokenize(sequence):
            for w in nltk.word_tokenize(sent):
                word_tokens.append(w)
        return word_tokens

    def generate_test(self):
        sentences = nltk.sent_tokenize(self.summary)
        cp = nltk.RegexpParser(self.grammar)
        questions = []
        correct_answers = []

        while len(questions) < self.noOfQues:
            rand_num = np.random.randint(0, len(sentences))
            sentence = sentences[rand_num]
            tagged_words = nltk.pos_tag(nltk.word_tokenize(sentence))
            tree = cp.parse(tagged_words)
            selected_key = None

            for subtree in tree.subtrees():
                if subtree.label() == "CHUNK":
                    temp = " ".join([sub[0] for sub in subtree])
                    temp = temp.strip()
                    temp = temp.upper()
                    if temp not in self.generated_questions:
                        self.generated_questions.add(temp)
                        selected_key = temp
                        break

            if selected_key is not None:
                answer = sentence
                rand_num %= 4
                question = self.question_pattern[rand_num] + selected_key + "?"

                # Generate options
                options = self.answer_options(answer, num_options=4)
                random.shuffle(options)  # Shuffle the options

                # Store the question and correct answer
                questions.append({"Question": question, "Options": options})
                correct_answers.append(answer)

        return questions, correct_answers

    @staticmethod
    def answer_options(provided_answer, num_options=4):
        # Initialize options with the provided answer
        options = [provided_answer]

        # Get WordNet synsets for the provided answer
        synsets = wn.synsets(provided_answer)

        # Add synonyms and related words to options
        for synset in synsets:
            for lemma in synset.lemmas():
                if lemma.name() != provided_answer:
                    options.append(lemma.name())

        # Generate false answers based on simple modifications of the provided answer
        words = provided_answer.split()
        for _ in range(num_options - len(options)):
            random.shuffle(words)
            false_answer = ' '.join(words)
            options.append(false_answer)

        # Shuffle the options and select only num_options options
        random.shuffle(options)
        options = options[:num_options]

        return options
