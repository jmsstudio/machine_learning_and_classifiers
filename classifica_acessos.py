from dados import load_access
from sklearn.naive_bayes import MultinomialNB

x, y = load_access()

train_data = x[:90]
train_result = y[:90]

test_data = x[-9:]
test_result = y[-9:]

model = MultinomialNB()

model.fit(x, y)

result = model.predict(test_data)

differences = result - test_result

matches = [d for d in differences if d == 0]

total_matches = len(matches)
total_elements = len(test_data)

print "Taxa de acertos:", round(100.0 * total_matches / total_elements, 2), "%"
print "Total de elementos:", total_elements
