css = '''
<style>
.chat-message {
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
}
.chat-message.user {
    background-color: #2b313e;
}
.chat-message.bot {
    background-color: #475063;
}
.chat-message .avatar {
    width: 78px;
    height: 78px;
    flex-shrink: 0;
}
.chat-message .avatar img {
    width: 78px;
    height: 78px;
    border-radius: 50%;
    object-fit: cover;
}
.chat-message .message {
    flex: 1;
    padding: 0 1.5rem;
    color: #fff;
    font-size: 1rem;
    line-height: 1.5;
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.ibb.co/cN0nmSj/Screenshot-2023-05-28-at-02-37-21.png" alt="Bot Avatar">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://cdn-icons-png.flaticon.com/512/149/149071.png" alt="User Avatar">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''
