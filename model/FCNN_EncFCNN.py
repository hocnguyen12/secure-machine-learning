import torch
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import time
from torch.utils.data import DataLoader
import tenseal as ts
import os
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt


torch.manual_seed(73)

# Paramètres
batch_size = 128
pixels = 28
input_size = pixels * pixels
mean = [0.4584]
std = [0.2492]
n_epochs = 50

# Chemins vers l'état du modèle sauvegardé 
model_full_path_en_clair = ""

# Chemins vers les dossiers de données
train_dir = 'your_path_to_training_set'
test_dir = 'your_path_to_test_set'

# Transformation des données 
transform = transforms.Compose([
    transforms.Grayscale(),  # Conversion en niveaux de gris
    transforms.Resize((pixels, pixels)),  
    transforms.ToTensor(),  # Conversion en torch.Tensor et normalisation [0, 1]
    transforms.Normalize(mean, std),
])





# Chargement des données
train_data = datasets.ImageFolder(train_dir, transform=transform)
test_data = datasets.ImageFolder(test_dir, transform=transform)

# DataLoader
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

print('{0: "Chat", 1: "Chien"}')

class FullyConnectedNet(torch.nn.Module):
    def __init__(self, input_size=pixels*pixels, hidden=64, output=2):
        super(FullyConnectedNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 1024)
        self.fc2 = torch.nn.Linear(1024, hidden)
        self.fc3 = torch.nn.Linear(hidden, output)

    def forward(self, x):
        x = x.view(-1, input_size)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def test(model, test_loader, criterion):
    test_loss = 0.0
    class_correct = list(0. for i in range(2))
    class_total = list(0. for i in range(2))

    # model in evaluation mode
    model.eval()

    for data, target in test_loader:
        output = model(data)
        loss = criterion(output, target)
        test_loss += loss.item()
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct = pred.eq(target.data.view_as(pred))
        # calculate test accuracy for each object class
        for i in range(len(target)):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    # calculate and print avg test loss
    test_loss = test_loss / len(test_loader)
    print(f'Test Loss: {test_loss:.6f}\n')

    # Calculer l'accuracy globale
    total_correct = sum(class_correct)
    total = sum(class_total)
    accuracy = 100 * total_correct / total

    # Afficher l'accuracy pour chaque classe
    for label in range(2):  # Assurez-vous que cette plage est correcte pour votre nombre de classes
        if class_total[label] > 0:
            print(
                f'Test Accuracy of {label}: {100 * class_correct[label] / class_total[label]:.2f}% '
                f'({int(class_correct[label])}/{int(class_total[label])})'
            )
        else:
            print(f'Test Accuracy of {label}: N/A (no samples)')
            
    print(f'\nOverall Test Accuracy: {accuracy:.2f}%\n')

    return test_loss, accuracy

def train(model, train_loader, criterion, optimizer,scheduler, n_epochs):
    # Initialiser les listes pour enregistrer les métriques à chaque époque
    epoch_train_losses = []
    epoch_val_losses = []
    epoch_accuracies = []

    model.train()
    for epoch in range(1, n_epochs+1):

        train_loss = 0.0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # calculate average losses
        train_loss = train_loss / len(train_loader)
        epoch_train_losses.append(train_loss)

        # Validation phase
        val_loss, val_accuracy = test(model, test_loader, criterion)
        epoch_val_losses.append(val_loss)
        epoch_accuracies.append(val_accuracy)

        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
        scheduler.step()
    # model in evaluation mode
    model.eval()
    return model, epoch_train_losses, epoch_val_losses, epoch_accuracies


model = FullyConnectedNet()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)


# Vérifie si un état de modèle pré-entraîné existe et le charge
if os.path.exists(model_full_path_en_clair):
    print("Chargement du modèle en clair pré-entraîné...")
    model = torch.load(model_full_path_en_clair)
else:
    print("Entraînement d'un nouveau modèle...")

# Que le modèle soit chargé ou nouvellement initialisé, nous pouvons continuer l'entraînement ici
print("Entraînement du modèle en clair...")
model.train()  # Assurez-vous que le modèle est en mode entraînement
start = time.time()
model, epoch_train_losses, epoch_val_losses, epoch_accuracies = train(model, train_loader, criterion, optimizer,scheduler, n_epochs)  
end = time.time()
print("L'entraînement du modèle en clair a pris", end - start, "secondes")


torch.save(model, model_full_path_en_clair)               



# Test du modèle en clair
start = time.time()
test(model, test_loader, criterion)
end = time.time()

print("model testing takes",end - start,"s\n")



class EncFullyConnectedNet: 
    def __init__(self, torch_nn):
        # Initialisation avec les poids et les biais du modèle en clair convertis en listes
        self.fc1_weight = torch_nn.fc1.weight.T.data.tolist()
        self.fc1_bias = torch_nn.fc1.bias.data.tolist()
        
        self.fc2_weight = torch_nn.fc2.weight.T.data.tolist()
        self.fc2_bias = torch_nn.fc2.bias.data.tolist()
        
        self.fc3_weight = torch_nn.fc3.weight.T.data.tolist()
        self.fc3_bias = torch_nn.fc3.bias.data.tolist()
        
        
    def forward(self, enc_x):
        # Évaluation du modèle sur des données chiffrées
        #start = time.time()
        enc_x = enc_x.mm(self.fc1_weight) + self.fc1_bias
        #end = time.time()
        #print("fc1 takes", end - start)
        
        # square activation
        #start = time.time()
        enc_x.square_()
        #end = time.time()
        #print("square activation takes", end - start)
        
        # fc2 layer
        #start = time.time()
        enc_x = enc_x.mm(self.fc2_weight) + self.fc2_bias
        #end = time.time()
        #print("fc2 takes", end - start)
        
        #start = time.time()
        enc_x.square_()
        #end = time.time()
        #print("square activation takes", end - start)
        
        # fc3 layer
        #start = time.time()
        enc_x = enc_x.mm(self.fc3_weight) + self.fc3_bias
        #end = time.time()
        #print("fc3 takes", end - start)
        return enc_x
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


# Définir la taille du sous-ensemble de test
n_test_samples = 100  

# Créer des indices aléatoires pour le sous-ensemble de test
test_indices = np.random.choice(len(test_data), n_test_samples, replace=False)

# Créer un DataLoader pour le sous-ensemble de test
test_sampler = SubsetRandomSampler(test_indices)
test_subset_loader = DataLoader(test_data, batch_size=1, sampler=test_sampler)

def enc_test(context, model, test_loader, criterion):
    times = []  # Liste pour stocker les temps de calcul pour chaque échantillon

    test_loss = 0.0
    class_correct = list(0. for i in range(2))
    class_total = list(0. for i in range(2))
    class_counts = {0: 0, 1: 0}  # Pour le débogage : compter le nombre d'images par classe
    sample_counter = 0


    for data, target in test_loader:
        sample_counter += 1  # Incrémenter le compteur pour chaque échantillon traité
        # Aplatir correctement les données pour chaque image
        data_flat = data.view(-1).tolist()  # Supprime toutes les dimensions sauf la dimension des features
        # Mesurer le temps de chiffrement
        start_time = time.time()
        # Chiffrer les données
        x_enc = ts.ckks_vector(context, data_flat)
        encryption_time = time.time() - start_time
        # Effectuer la prédiction
        # Mesurer le temps d'inférence
        start_time = time.time()
        enc_output = model(x_enc)
        inference_time = time.time() - start_time
        # Mesurer le temps de déchiffrement
        start_time = time.time()
        output = enc_output.decrypt()
        output = torch.tensor(output).view(1, -1)
        decryption_time = time.time() - start_time

        # Ajouter le temps total pour cet échantillon
        times.append(encryption_time + inference_time + decryption_time)

        # Ajuster la dimension de target pour la fonction de perte
        target = target.view(-1)  # Supprime la dimension de lot si présente
        # Calculer la perte
        loss = criterion(output, target)
        test_loss += loss.item()
        # Déterminer la classe prédite
        _, pred = torch.max(output, 1)
        correct = pred.eq(target.view_as(pred))
        label = target.item()
        class_correct[label] += correct.item()
        class_total[label] += 1
        class_counts[label] += 1

        # Affichage après chaque prédiction
        print(f"Échantillon #{sample_counter}: target={label}, prediction={pred.item()}")

    # Calculer et afficher la perte de test moyenne et l'exactitude pour chaque classe
    test_loss /= len(test_loader)
    print(f'Test Loss: {test_loss:.6f}\n')

    # Afficher l'exactitude pour chaque classe
    labels_map = {0: "Chat", 1: "Chien"}
    for label in range(2):
        if class_total[label] > 0:
            print(f'Test Accuracy of {labels_map[label]} (classe {label}): {100 * class_correct[label] / class_total[label]:.2f}% ({class_correct[label]}/{class_total[label]})')
        else:
            print(f'Test Accuracy of {labels_map[label]} (classe {label}): N/A (no samples)')

    # Afficher la distribution des classes dans les images testées
    print("Distribution des classes dans les images testées :", class_counts)
    # Calculer le temps moyen de calcul
    average_time = sum(times) / len(times)
    print(f"Moyenne du temps de calcul par échantillon: {average_time:.4f} secondes")
    
    return times  




## Encryption Parameters

# controls precision of the fractional part
bits_scale = 26

# Create TenSEAL context
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[31, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, 31]
)


# set the scale
context.global_scale = pow(2, bits_scale)

# galois keys are required to do ciphertext rotations
context.generate_galois_keys()

# Initialisation du modèle chiffré
enc_model = EncFullyConnectedNet(model)
start = time.time()
# Test du modèle chiffré avec le sous-ensemble aléatoire
times = enc_test(context, enc_model, test_subset_loader, criterion)
end = time.time()
print("encrypted model testing takes",end - start,"s")


