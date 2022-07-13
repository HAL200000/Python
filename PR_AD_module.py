from USER_module import User

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