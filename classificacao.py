from sklearn.naive_bayes import MultinomialNB

#is_fat, has_short_legs, barks
pig1 = [1, 1, 0]
pig2 = [1, 0, 0]
pig3 = [0, 1, 0]
dog1 = [1, 1, 1]
dog2 = [0, 1, 1]
dog3 = [1, 0, 1]

data = [pig1, pig2, pig3, dog1, dog2, dog3]

#dog = -1
#pig = 1
marks = [1, 1, 1, -1, -1, -1]

classifier = MultinomialNB()

classifier.fit(data, marks)

element1 = [0, 0, 1]
element2 = [0, 0, 0]
element3 = [0, 1, 1]

test = [element1, element2, element3]

test_marks = [-1, 1, -1]

result = classifier.predict(test)

diffs = result - test_marks

matches = [match for match in diffs if match == 0]

total_matches = len(matches)
total_elements = len(test)

success_rate = 100.0 * total_matches / total_elements
print "Acertos", total_matches
print "Total de elementos", total_elements
print "Percentual de sucess:", round(success_rate, 2), "%"
