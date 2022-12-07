from tkinter import *
from tkinter.scrolledtext import *
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import random
import torch.nn.functional as F
from tqdm import trange
import os


tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
model = GPT2LMHeadModel.from_pretrained('distilgpt2')
SPECIAL_TOKENS_DICT = {
    'pad_token': '<pad>',
    'additional_special_tokens': ['<context>', '<response>'],
}

tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
model.resize_token_embeddings(len(tokenizer))



def load_model(model_name):

    model.load_state_dict(torch.load(model_name,map_location=torch.device('cpu')))
    print(model_name + " LOADED")

    model_set.set(True)
    
    header.set(header_base + model_name.replace("models/","") + "!")
    
    character.set(model_name.replace("models/",""))


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits

def sample_sequence(model, length, context, segments_tokens=None, num_samples=1, temperature=1, top_k=0, top_p=0.0, repetition_penalty=1.0,
                    device='cpu'):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context

    with torch.no_grad():
        for _ in trange(length):

            inputs = {'input_ids': generated}
            if segments_tokens != None:
              inputs['token_type_ids'] = torch.tensor(segments_tokens[:generated.shape[1]]).unsqueeze(0).repeat(num_samples, 1)


            outputs = model(**inputs)  
            next_token_logits = outputs[0][:, -1, :] / (temperature if temperature > 0 else 1.)

            
            for i in range(num_samples):
                for _ in set(generated[i].tolist()):
                    next_token_logits[i, _] /= repetition_penalty
                
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            if temperature == 0: 
                next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
            else:
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
    return generated

def send(event=None):
    # Enable editing/inserting into the chatbox.
    chatbox.config(state=NORMAL)
    if(model_set.get()):
        send = "You: " + e.get()
        chatbox.insert(END, "\n" + send)

        context_tkn = tokenizer.additional_special_tokens_ids[0]
        response_tkn = tokenizer.additional_special_tokens_ids[1]

        input_ids = [context_tkn] + tokenizer.encode(e.get())

        segments = [response_tkn] * 64
        segments[:len(input_ids)] = [context_tkn] * len(input_ids)

        input_ids += [response_tkn]

        generated = sample_sequence(
        model,
        length=20,
        context=input_ids,
        segments_tokens=segments,
        num_samples=1,
        temperature=0.3,
        top_k=2,
        top_p=0.8
        )  

        e.delete(0, END)
        
        
        response = tokenizer.decode(generated.squeeze().tolist())
        response = response.split('<|endoftext|>')[0]
        response = response.split('<context>')[1]
        response = response.split('<response>')

        chatbox.insert(END,"\n" + character.get() + ": " + str(response[1]))
        print(response[0],response[1])
       


        
        chatbox.yview(END)
    else:
        chatbox.insert(END,"\nUse the \"Models\" menu to select who you want to talk to.")
    # Disable editing/inserting into the chatbox.
    chatbox.config(state=DISABLED)

root = Tk()
root.geometry("800x720")
root.title("Chatbot")

root.bind('<Return>', send)


character = StringVar()
header_base = "You're talking to "

header = StringVar()
header.set("Welcome! Please select a character to speak with.")

model_set = BooleanVar()
model_set.set(False)


# Add Model Selection menu to switch between characters.
menubar = Menu(root)
model_menu = Menu(menubar, tearoff=0)

model_menu.add_radiobutton(label ="Finn the Human", command = lambda: load_model("models/Finn the Human"))
model_menu.add_radiobutton(label ="Ice King", command = lambda: load_model("models/Ice King"))

menubar.add_cascade(label="File", menu=model_menu)
root.config(menu=menubar)


BG_GRAY = "#ffc500"
BG_COLOR = "#203180"
TEXT_COLOR = "#ffc500"

FONT = "Helvetica 14"
FONT_BOLD = "Helvetica 13 bold"

label1 = Label(root, bg="#78cfff", fg="#dc3116", textvariable=header, font=FONT_BOLD, pady=10, width=50, height=1).grid(row=0)


chatbox = ScrolledText(root, bg=BG_COLOR, fg=TEXT_COLOR,bd=5, font=FONT, width=70, spacing3 = 4, wrap =WORD, state=DISABLED)
chatbox.grid(row=1, column=0, columnspan=2)
 

e = Entry(root, bg="#2C3E50", fg="#dadce6", font=FONT, width=67)
e.grid(row=2, column=0)
 

send = Button(root, text="Send", font=FONT_BOLD, bg=BG_GRAY,
              command=send).grid(row=2, column=1)


root.mainloop()
