test = {"apple":5, "banana":3, "cherry":9}

print(dict(sorted(test.items(),key=lambda x: x[1])))
