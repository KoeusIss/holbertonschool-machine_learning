#!/usr/bin/env python3
"""QA Bot module"""


EXIT_KEYWORD = ('bye', 'goodbye', 'quit', 'exit')
if __name__ == "__main__":
    while True:
        print('Q:', end=' ')
        question = input()
        if question.lower() in EXIT_KEYWORD:
            answer = 'Goodbye'
            print('A: {}'.format(answer))
            break
        answer = ''
        print('A: {}'.format(answer))
