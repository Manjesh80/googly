class Employee():
    def __init__(self, name):
        self._name = name

    def __hash__(self):
        print("Calling HASH")
        return hash(self._name)

    # def __eq__(self, other):
    #     print("Calling EQUAL")
    #     return self._name == other._name


class Person():
    def __init__(self, name, ssn):
        self.name = name
        self.ssn = ssn

    def __eq__(self, other):
        return isinstance(other, Person) and self.ssn == other.ssn

    def __hash__(self):
        # use the hashcode of self.ssn since that is used
        # for equality checks as well
        return hash(self.ssn)


if __name__ == "__main__":
    e1 = Employee("Ganesh")
    e2 = Employee("Ganesh")
    s = {e1, e2}
    print(len(s))
    # print(s)
    #
    # d = {e1: "emp1", e2: "emp2"}
    # print(len(d))
    # print(d)
    # p = Person('Foo Bar', 123456789)
    # q = Person('Fake Name', 123456789)
    # print(len({p, q}))
