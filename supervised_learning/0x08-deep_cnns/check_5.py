#!/usr/bin/env python3

con = 0
linesB = {hash(line) for line in open('intranet_5.txt')}
for line in open('output_5.txt'):
    if hash(line) not in linesB:
        print('line: {}'.format(con))
        print('diferente')
    con += 1
