from threading import *
import time
import random
import socket
from concurrent.futures import ThreadPoolExecutor

even_lock = Lock()
odd_lock = Lock()
num_lock = Lock()
even_counter_condition = Condition()
odd_counter_condition = Condition()
number_counter = Condition()
number = 0


class EvenOddMonitor(Condition):
    ODD_TURN = True
    EVEN_TURN = False

    def __init__(self):
        super().__init__()
        self.turn = self.ODD_TURN

    def wait_turn(self, old_turn):
        with self:
            while self.turn != old_turn:
                self.wait()

    def toggle_turn(self):
        with self:
            self.turn ^= True
            self.notify()


class OddThread(Thread):
    def __init__(self, monitor):
        super().__init__()
        self.monitor = monitor

    def run(self):
        for i in range(1, 101, 2):
            self.monitor.wait_turn(EvenOddMonitor.ODD_TURN)
            print(i)
            self.monitor.toggle_turn()


class EvenThread(Thread):
    def __init__(self, monitor):
        super().__init__()
        self.monitor = monitor

    def run(self):
        for i in range(2, 101, 2):
            self.monitor.wait_turn(EvenOddMonitor.EVEN_TURN)
            print(i)
            self.monitor.toggle_turn()


class CustomCondition(Condition):
    EVEN_TURN = True
    ODD_TURN = False

    def __init__(self):
        super().__init__()
        self.state = CustomCondition.EVEN_TURN

    def wait_for_turn(self, turn):
        with self:
            while self.state != turn:
                self.wait()

    def toggle_turn(self):
        with self:
            self.state ^= True
            self.notify()


even_odd_result = []


def print_even(n, monitor):
    time.sleep(3)
    global number, even_odd_result
    for i in range(n):
        monitor.wait_for_turn(CustomCondition.EVEN_TURN)
        number += 1
        print(f"Even thread value is ==> {number}")
        even_odd_result.append(number)
        monitor.toggle_turn()


def print_odd(n, monitor):
    global number, even_odd_result
    for i in range(n):
        monitor.wait_for_turn(CustomCondition.ODD_TURN)
        number += 1
        print(f"Odd thread value is ==> {number}")
        even_odd_result.append(number)
        monitor.toggle_turn()


# 19.3 Implement synchronization for Even-Odd interleaved thread
def print_even_odd(n):
    global even_odd_result
    monitor = CustomCondition()
    even_thread = Thread(target=print_even, args=(n, monitor,))
    odd_thread = Thread(target=print_odd, args=(n, monitor,))
    even_thread.start()
    odd_thread.start()
    even_thread.join()
    odd_thread.join()
    return even_odd_result


# 19.51 Understand re-entrant lock
class Box():
    box_locker = RLock()

    def __init__(self):
        self.total_box_count = 0

    def perform_operation(self, n):
        Box.box_locker.acquire()
        self.total_box_count += n
        print(f" Total box count ==> {self.total_box_count}")
        Box.box_locker.release()

    def add_box(self, n):
        Box.box_locker.acquire()
        self.perform_operation(1)
        Box.box_locker.release()

    def remove_box(self, n):
        Box.box_locker.acquire()
        self.perform_operation(-1)
        Box.box_locker.release()


def box_adder(box, n):
    while n > 0:
        n -= 1
        box.add_box(1)


def box_remover(box, n):
    while n > 0:
        n -= 1
        box.remove_box(1)


def demo_re_entrant_lock():
    box = Box()
    adder = Thread(target=box_adder, args=(box, 5,))
    remover = Thread(target=box_remover, args=(box, 5,))
    adder.start()
    remover.start()
    adder.join()
    remover.join()


# 19.52 Demo producer consumer
items = []
latch = Condition()


class Producer(Thread):
    def __init__(self):
        super().__init__()

    def run(self):
        n = 100
        global items
        global latch
        while n > 0:
            n -= 1
            latch.acquire()
            if len(items) > 0 and len(items) % 5 == 0:
                latch.notify()
                latch.release()
                latch.wait()
            items.append(n)
            print(f"Adding item to queue ==> {n}")


class Consumer(Thread):
    def __init__(self):
        super().__init__()

    def run(self):
        n = 100
        global items
        global latch
        while n > 0:
            n -= 1
            latch.acquire()
            if len(items) == 0:
                latch.notify()
                latch.release()
                latch.wait()
            print(f"Removing item to queue {items.pop(0)}")


test_condition = Condition()


def acquirer():
    print(" acquirer start")
    global test_condition
    test_condition.acquire()
    print(" acquirer got access")
    test_condition.wait()
    print(" Acquirer came out")


def notifier():
    print(" notifier start")
    global test_condition
    test_condition.acquire()
    print(" notifier got access")
    test_condition.notify()
    test_condition.release()
    print(" notifier came out")


semaphore = Semaphore(0)
item = []


# 19.53 Demo producer consumer with Semaphore
def consumer_semaphore():
    global item
    for i in range(5):
        print("consumer is waiting.")
        semaphore.acquire()
        print("Consumer notify : consumed item number %s " % item.pop(0))


def producer_semaphore():
    global item
    for i in range(5):
        temp = random.randint(0, 1000)
        item.append(temp)
        print("producer notify : produced item number %s" % temp)
        semaphore.release()


# 19.54 Demo synchronization using Events
event = Event()
items = []


class Consumer_events(Thread):
    def __init__(self, items, event):
        Thread.__init__(self)
        self.items = items
        self.event = event

    def run(self):
        time.sleep(10)
        while True:
            time.sleep(2)
            self.event.wait()
            if len(self.items) > 0:
                item = self.items.pop()
                print('Consumer notify : %d popped from list by %s' % (item, self.name))
            else:
                print('********* No Item found ********* ')


class Producer_events(Thread):
    def __init__(self, integers, event):
        Thread.__init__(self)
        self.items = items
        self.event = event

    def run(self):
        global item
        for i in range(100):
            time.sleep(2)
            item = random.randint(0, 256)
            item = i
            self.items.append(item)
            print('Producer notify : item N %d appended to list by %s' % (item, self.name))
            print('Producer notify : event set by %s' % self.name)
            self.event.set()
            print('Produce notify : event cleared by %s ' % self.name)
            self.event.clear()


# 19.4 Start socket Server
def handle_client_socket(client_sock):
    pass

def start_server():
    port = 8080
    tp = ThreadPoolExecutor(max_workers=10)
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('', port))
    while True:
        sock, addr = server_socket.accept()
        tp.submit(handle_client_socket, sock)


if __name__ == "__main__":
    t1 = Consumer_events(items, event)
    t2 = Producer_events(items, event)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # acqi = Thread(target=acquirer)
    # noti = Thread(target=notifier)
    # acqi.start()
    # noti.start()
    # acqi.join()
    # noti.join()

    # producer = Producer()
    # consumer = Consumer()
    # producer.start()
    # consumer.start()
    # producer.join()
    # consumer.join()

# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
