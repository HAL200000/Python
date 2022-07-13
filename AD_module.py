class User:
    def __init__(self, first_name, last_name, sex):
        self.first_name = first_name
        self.last_name = last_name
        self.sex = sex
        self.login_attempts = 0

    def describe_user(self):
        print(f"the user's full name is {self.first_name} {self.last_name}, who is a {self.sex}")

    def greet_user(self):
        print(f"Hello, {self.first_name} {self.last_name}")

    def increment_login_attempts(self):
        self.login_attempts += 1

    def reset_login_attempts(self):
        self.login_attempts = 0

class Admin(User):
    def __init__(self, first_name, last_name, sex):
        super(Admin, self).__init__(first_name, last_name, sex)
        self.privilege = Privileges()



class Privileges:
    def __init__(self):
        self.privilege = ['can add post', 'can delete post', 'can be users']

    def show_privileges(self):
        for privilege in self.privilege:
            print(privilege)