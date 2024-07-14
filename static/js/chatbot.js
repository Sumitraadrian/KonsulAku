$(document).ready(function () {
    var socket = io();
    var chatHistory = []; // Array untuk menyimpan riwayat percakapan

    // Fungsi untuk menampilkan pesan pengantar sistem saat halaman dimuat
    function showIntroduction() {
        var introductionMessage = "Hi there! I'm your Skin Disease Prediction Chatbot. You can ask me about various skin diseases or upload an image for diagnosis. I am Konsula, a chatbot that will make it easier for you to detect and know the type of skin disease you are experiencing. Konsula will provide responses related to skin disease problems such as treatment recommendations needed as an initial step (first aid).";
        appendMessage('bot', introductionMessage);
        // Simpan pesan pengantar ke dalam riwayat percakapan
        chatHistory.push({ sender: 'bot', message: introductionMessage });
    }

    // Memanggil fungsi showIntroduction saat halaman pertama kali dimuat
    showIntroduction()

    $('#send-button').on('click', function (event) {
        event.preventDefault();
        var message = $('#message-input').val();
        if (message.trim() !== '') {
            appendMessage('user', message);
            socket.emit('message', message);
            $('#message-input').val('');
            checkMessage(message);
        }
    });

    socket.on('response', function (data) {
        appendMessage('bot', data.response);
        if (data.response.toLowerCase().includes('upload')) {
            $('#upload-predict-buttons').show();
        } else {
            $('#upload-predict-buttons').hide();
        }
        // Simpan pesan dari bot ke riwayat percakapan
        chatHistory.push({ sender: 'bot', message: data.response });

        // Check if additional help message is needed
        if (data.message) {
            appendMessage('bot', data.message);
            chatHistory.push({ sender: 'bot', message: data.message });
        }
    });

    $('#image-form').on('submit', function (event) {
        event.preventDefault();
        var formData = new FormData(this);
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: formData,
            contentType: false,
            processData: false,
            success: function (response) {
                appendMessage('bot', 'Predicted Label: ' + response.label);
                appendMessage('bot', response.description);
                appendMessage('bot', '<img src="/static/uploads/' + response.filename + '" width="300">');
                // Simpan pesan dari bot ke riwayat percakapan
                chatHistory.push({ sender: 'bot', message: 'Predicted Label: ' + response.label });
                chatHistory.push({ sender: 'bot', message: response.description });

                // Check if additional help message is needed
                if (response.message) {
                    appendMessage('bot', response.message);
                    chatHistory.push({ sender: 'bot', message: response.message });
                }
            },
            error: function (error) {
                console.log(error);
            }
        });
    });

    // Fungsi untuk menampilkan tombol-tombol sesi
            // Event listener for 'New Chat' button
    // Event listener for 'New Chat' button
    $('#new-chat-button').on('click', function () {
        $.get('/new_chat', function () {
            $('#chat-box').empty(); // Clear chat box
            chatHistory = []; // Clear chat history
            displayPreviousChats(); // Display previous chats as buttons
        });
    });


    function appendMessage(sender, text) {
        var messageClass = sender === 'user' ? 'user' : 'bot';
        var messageHtml = '<div class="message ' + messageClass + '"><div class="text">' + text + '</div></div>';
        $('#chat-box').append(messageHtml);
        $('#chat-box').scrollTop($('#chat-box')[0].scrollHeight);

        // Check if user has thanked
        if (text.toLowerCase().includes('thank')) {
            var warningMessage = "Please remember to consult with a doctor for proper diagnosis and treatment of skin diseases.";
            var warningHtml = '<div class="message bot"><div class="text">' + warningMessage + '</div></div>';
            $('#chat-box').append(warningHtml);
            $('#chat-box').scrollTop($('#chat-box')[0].scrollHeight);
        }
    }

    function checkMessage(message) {
        if (message.toLowerCase().includes('help me detect skin disease')) {
            $('#upload-predict-buttons').show();
        } else {
            $('#upload-predict-buttons').hide();
        }
    }
});      