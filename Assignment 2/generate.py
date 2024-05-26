# import some packages you need here
import dataset
from model import CharLSTM
import torch
import torch.nn.functional as F

def generate(model, seed_characters, temperature, char_idx, idx_char, max_length):
    """ Generate characters

    Args:
        model: trained model
        seed_characters: seed characters
        temperature: T
        args: other arguments if needed

    Returns:
        samples: generated characters
    """

    # write your codes here
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    samples = []

    for seed in seed_characters:
        hidden = model.init_hidden(1)

        if isinstance(hidden, tuple):
            hidden = tuple(h.to(device) for h in hidden)
        else:
            hidden = hidden.to(device)

        generated = seed.copy()
        input_idx = torch.tensor([[char_idx[c] for c in seed[0]]]).to(device)

        with torch.no_grad():
            for _ in range(max_length - len(seed[0])):
                output, _ = model(input_idx, hidden)
                output = output[:, -1, :] / temperature
                probabilities = F.softmax(output, dim=-1)
                next_char_idx = torch.multinomial(probabilities, 1).item()
                generated_char = idx_char[next_char_idx]

                generated.append(generated_char)
                input_idx = torch.tensor([[next_char_idx]]).to(device)
    
        samples.append(generated)

    return samples

if __name__ == '__main__':
    total_dataset = dataset.Shakespeare(input_file='./shakespeare_train.txt')
    char_idx = total_dataset.char_idx
    idx_char = total_dataset.idx_char

    model_path = './Best model/lstm_best_model.pkl'
    
    model = CharLSTM()
    
    model.load_state_dict(torch.load(model_path))

    seed_characters = [['The'], ['I'], ['Me'], ['You'], ['That']]

    temps = [0.2, 0.4, 0.6, 0.8, 1.0] 

    for temperature in temps:
        print('**********************************************')
        print("Temperature:", temperature)
        generated_text = generate(model, seed_characters, temperature, char_idx, idx_char, 100)
        
        for sample in generated_text:
            full_txt = ''
            for s in sample:
                full_txt+=s
            print(full_txt)
            print('-------------------------------------------')