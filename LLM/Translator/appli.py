# Libraries
import gradio as gr
import translator
import torch
import time

# Instanciate a translator 
translator_ = translator.translator()

# Function to translate an input sentence
def translate_input(input_sentence: str):
    return translator_.translate(input_sentence)

demo = gr.Interface(fn = translate_input,
                    inputs = gr.Dropdown(
                        ["Depuis 2011, une douzaine d'États ont adopté des lois exigeant des électeurs de prouver qu'ils sont citoyens américains.", 
                         "Ils cherchent comment, au cours de la prochaine décennie, ils pourront passer à un système permettant aux conducteurs de payer en fonction du nombre de miles parcourus.", 
                         "Cela est vraiment indispensable pour notre nation.",
                         "Toutefois, si aucune nouvelle commande n'est annoncée au cours des prochains mois, nous nous attendons à ce que le marché devienne plus sceptique au sujet du programme.",
                         "Le premier avion d'essai a été dévoilé en mars et s'est envolé pour la première fois en septembre après des mois de retard."], 
                        label = "Choose a sentence to translate", 
                        info = "These sentences are taken from the wmt14 dataset ('fr-en' test split)."
                    ),
                    outputs = ["text"],
)

if __name__ == "__main__":
    demo.launch()