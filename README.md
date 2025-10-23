# Projet CNN : Chats vs Chiens (Comparaison From Scratch vs Transfert Learning)

**Auteur :** Mouhamed DIOUF  
**GitHub :** [Mdcode002](https://github.com/Mdcode002)

---

## Objectif

Ce projet compare deux approches pour la classification d'images (Chats vs Chiens) en utilisant **PyTorch** :

- **Expérience A : CNN From Scratch** — Conception et entraînement d’un réseau simple.
- **Expérience B : Transfert Learning (ResNet18)** — Réutilisation d’un modèle pré-entraîné sur ImageNet.

L’objectif est de mesurer l’impact du transfert learning sur :
- la **vitesse de convergence**,
- la **performance (accuracy, recall, F1)**,
- la **robustesse** du modèle.

---

## Environnement & Dépendances

Exécution recommandée : **Google Colab (GPU activé)**

### Installation locale

```bash
pip install -r requirements.txt
```

**requirements.txt :**
```txt
numpy
matplotlib
torch>=2.0
torchvision>=0.15
scikit-learn
# torchmetrics==1.4.0 (optionnel)
```

**.gitignore :**
```gitignore
# Données
data/
Cat_Dog_data/

# Modèles sauvegardés
*.pt
*.pth
checkpoints/

# Environnements Python
*.env
venv/
.venv/

# Caches
__pycache__/
.ipynb_checkpoints/
```

---

## Organisation des Données

Jeu de données : **Kaggle Cats vs Dogs**

Structure :
```bash
Cat_Dog_data/
├── train/
│   ├── cat/
│   └── dog/
└── test/
    ├── cat/
    └── dog/
```
> 22 500 images pour l’entraînement et 2 500 pour le test.

---

## Exécution du Projet

Tout est contenu dans **notebook.ipynb**.

Reproductibilité :
```python
seed_everything(42)
```

### Pré-traitement & Augmentation
- **Train :** RandomRotation(30), RandomResizedCrop(224), RandomHorizontalFlip()
- **Val/Test :** Resize(255), CenterCrop(224)
- **Normalisation :** mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

---

## Expériences

### Expérience A : CNN From Scratch

- **Architecture :** 3 blocs conv + batchnorm + relu + maxpool + dropout (p=0.3)
- **Epochs :** 12 — **Batch size :** 32

**A.1 Adam**
```python
optimizer = Adam(lr=1e-3, weight_decay=1e-4)
scheduler = StepLR(step_size=5, gamma=0.5)
```

**A.2 SGD**
```python
optimizer = SGD(lr=0.03, momentum=0.9, weight_decay=5e-4)
scheduler = CosineAnnealingLR(T_max=12)
```

### Expérience B : Transfert Learning (ResNet18)

**Architecture :**
```python
model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(512, 2))
```

#### Phase 1 — Entraînement de la tête
- Gel des features
- Epochs : 6 — Optimiseur : Adam (LR=1e-3)

#### Phase 2 — Fine-tuning
- Dégel complet
- Epochs : 8 — Optimiseur : SGD (LR=1e-3, momentum=0.9)

---

## Résultats

| Modèle | Optimiseur | Scheduler | Val. Acc. | Test Acc. |
|---------|-------------|------------|------------|------------|
| CNN Scratch | Adam | StepLR | 76.67% | 79.92% |
| CNN Scratch | SGD | Cosine | 75.82% | — |
| ResNet18 (Head) | Adam | StepLR | 93.6% | — |
| ResNet18 (Finetune) | SGD | Cosine | 96.64% | **≈96.5%** |

**Analyse :**
- Le ResNet18 converge beaucoup plus vite (92% en 1 époque).
- Le modèle from scratch plafonne à ~80% malgré la régularisation.
- Le transfert learning offre une meilleure généralisation et stabilité.

---

## Rechargement & Évaluation

```python
def make_scratch():
    return SmallCNN(num_classes=2)

model = load_best('checkpoints/cnn_scratch_best.pth', make_scratch)
criterion = nn.CrossEntropyLoss()
loss, acc, _ = evaluate(model, test_loader, criterion)
print(f"[Reloaded scratch] TEST — acc={acc:.4f}")
```

---


## 📜 Licence

Ce projet est distribué sous la licence **MIT**.

---
