from Person import Person
from PersonDB import PersonDB


pessoa = Person(1)
banco = PersonDB()

banco.add(pessoa)
a = banco.show()