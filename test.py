
from threading import Thread, Condition


res = []
condition = Condition()
current = "A"


def send_a(a):
    global current
    global res
    for _ in range(a):
        with condition:
            while current != "A":
                condition.wait()
            res.append("A")
            current = "B"
            condition.notify_all()

def send_b(a):
    global current
    global res
    for _ in range(a):
        with condition:
            while current != "B":
                condition.wait()
            res.append("B")
            current = "C"
            condition.notify_all()

def send_c(a):
    global current
    global res
    for _ in range(a):
        with condition:
            while current != "C":
                condition.wait()
            res.append("C")
            current = "D"
            condition.notify_all()

def send_d(a):
    global current
    global res
    for _ in range(a):
        with condition:
            while current != "D":
                condition.wait()
            res.append("D")
            current = "A"
            condition.notify_all()


if __name__ == "__main__":
    a =355
    thread_list = [Thread(target=send_a, args=(a,)), Thread(target=send_b, args=(a,)), Thread(target=send_c, args=(a,)),
                   Thread(target=send_d, args=(a,))]
    for thread in thread_list:
        thread.start()

    for thread in thread_list:
        thread.join()
    print("".join(res))
    print(len(res)/4)

