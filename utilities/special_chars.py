

def safeprint(*s):
    try:
        print(*s)
    except UnicodeEncodeError:
        print("Unicode Error!!! Unable to print")

if __name__ == "__main__":
    s1 = "\u72d0"
    s2 = "\u8349"
    s3 = "\u7c7b"
    safeprint(s1)
    d = {1: s1}
    safeprint(d, s2)
       