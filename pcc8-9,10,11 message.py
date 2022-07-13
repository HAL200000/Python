messages = ['hello', 'you are welcome', 'goodbye', 'thanks']
sent_messages= []
def show_messages(messages):
    for message in messages:
        print(message)

def send_messages(old, new):
   while old:
       current_message = old.pop()
       print(current_message)
       new.append(current_message)

   print(messages)
   print(sent_messages)

show_messages(messages)
print('\n')
send_messages(messages[:],sent_messages) # 创建列表副本