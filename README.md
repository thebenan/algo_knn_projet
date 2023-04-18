# L'algorithme KNN

## Classe KNNClass
La classe KNNClass implémente un modèle de classification basé sur l'algorithme KNN (k-Nearest Neighbors). Ce modèle permet de classer des vecteurs dans différentes classes en fonction de leur similarité avec les vecteurs de chaque classe. Les méthodes de cette classe sont :

* init(self, description: str="", data=[]): 
Cette méthode est le constructeur de la classe. Elle prend en entrée une chaîne de caractères qui décrit l'ensemble des classes et une liste de vecteurs. Elle initialise deux variables d'instance : la description de l'ensemble de classes et les données, représentées par une liste de dictionnaires où chaque dictionnaire contient un label de classe et les vecteurs associés à cette classe.

* add_class(self, label: str, vectors: list): 
Cette méthode permet d'ajouter une nouvelle classe au modèle. Elle prend en entrée le nom de la  classe à ajouter et une liste de vecteurs à associer à cette classe. Elle vérifie d'abord si la classe n'existe pas déjà dans le modèle, sinon elle l'ajoute.

* get_classes(self): 
Cette méthode permet de récupérer la liste des classes actuelles.

* add_vector(self, label: str, vector): 
Cette méthode permet d'ajouter un vecteur à une classe existante. Elle prend en entrée le nom de la classe et le vecteur à ajouter. Elle vérifie si la classe existe dans le modèle, puis ajoute le vecteur à cette classe.

* del_class(self, label:str): 
Cette méthode permet de supprimer une classe existante. Elle prend en entrée le nom de la classe à supprimer et vérifie si elle existe dans le modèle. Si c'est le cas, elle la supprime.

* save_as_json(self, filename:str): 
Cette méthode permet de sauvegarder les données du modèle sous forme de fichier JSON. Elle prend en entrée le nom du fichier de sortie et utilise la méthode json.dump() pour écrire les données dans ce fichier.

* load_as_json(self, filename:str): 
Cette méthode permet de charger les données d'un fichier JSON dans un modèle. Elle prend en entrée le nom du fichier à charger et utilise la méthode json.load() pour récupérer les données du fichier.

* classify(self, vector: dict, k: int, sim_func=None) -> List[Tuple[str, float]]: 
Cette méthode permet de classer un vecteur avec les vecteurs des classes existantes. Elle prend en entrée un vecteur sous forme de hashage, le nombre de voisins les plus proches à considérer et une fonction de similarité (par défaut, la fonction de calcul de cosinus). Elle calcule d'abord les similarités entre le nouveau vecteur et les vecteurs de chaque classe, puis trie les résultats en fonction de la similarité décroissante. Elle renvoie ensuite une liste triée de paires [label:str,sim:float] pour les classes candidates. La similarité est la moyenne des similarités obtenues sur les vecteurs retenus pour la classe correspondante dans les k plus proches voisins.

**Création d'un modèle KNN**

Instanciation de la classe en fournissant une description :
```python
from knnclass import KNNClass
test_knn = KNNClass(description="données de test")
```
