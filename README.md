# L'algorithme KNN
Dans le cadre de cette tâche, des documents texte de recettes sont traités. Ces documents sont stockés dans les dossiers "plats", "desserts" et "entrees", contenant chacun cinq fichiers de recettes. Les fichiers sont convertis en vecteurs TF-IDF et stockés dans la variable `data` de la classe `KNNClass`. Tout d'abord, les fichiers sont lus, tokenizés et vectorisés à l'aide des fonctions de la classe `TextVect`. Les vecteurs obtenus sont utilisés pour calculer le TF-IDF de chaque terme dans chaque document, permettant ainsi d'obtenir des vecteurs normalisés de poids de termes pour chaque document. Les résultats du TF-IDF sont stockés dans la variable `data` de la classe `KNNClass`. Ensuite, lorsqu'un utilisateur fournit un nouveau texte à classifier, celui-ci est également converti en un vecteur TF-IDF et comparé aux vecteurs stockés dans `data` à l'aide de la méthode classify KNN, en utilisant une fonction de similarité qui peut être sélectionnée parmi les trois options disponibles dans la classe `Similarity`.
En outre, la classe `Gestion` contient deux méthodes utiles pour manipuler les données stockées dans la classe `KNNClass`. La méthode `add_class_input` permet d'ajouter des données à la classe `KNNClass` en appelant la méthode `add_class` de la classe `KNNClass`, tandis que la méthode `del_class_input` permet de supprimer des données de la classe en appelant la méthode `del_class` de la classe `KNNClass`.







## Classe `KNNClass`
La classe KNNClass implémente un modèle de classification basé sur l'algorithme KNN (k-Nearest Neighbors). Ce modèle permet de classer des vecteurs dans différentes classes en fonction de leur similarité avec les vecteurs de chaque classe. Les méthodes de cette classe sont :

* `init(self, description: str="", data=[]):` 
Cette méthode est le constructeur de la classe. Elle prend en entrée une chaîne de caractères qui décrit l'ensemble des classes et une liste de vecteurs. Elle initialise deux variables d'instance : la description de l'ensemble de classes et les données, représentées par une liste de dictionnaires où chaque dictionnaire contient un label de classe et les vecteurs associés à cette classe.

* `add_class(self, label: str, vectors: list):`
Cette méthode permet d'ajouter une nouvelle classe au modèle. Elle prend en entrée le nom de la  classe à ajouter et une liste de vecteurs à associer à cette classe. Elle vérifie d'abord si la classe n'existe pas déjà dans le modèle, sinon elle l'ajoute.

* `get_classes(self):` 
Cette méthode permet de récupérer la liste des classes actuelles.

* `add_vector(self, label: str, vector):` 
Cette méthode permet d'ajouter un vecteur à une classe existante. Elle prend en entrée le nom de la classe et le vecteur à ajouter. Elle vérifie si la classe existe dans le modèle, puis ajoute le vecteur à cette classe.

* `del_class(self, label:str):` 
Cette méthode permet de supprimer une classe existante. Elle prend en entrée le nom de la classe à supprimer et vérifie si elle existe dans le modèle. Si c'est le cas, elle la supprime.

* `save_as_json(self, filename:str):` 
Cette méthode permet de sauvegarder les données du modèle sous forme de fichier JSON. Elle prend en entrée le nom du fichier de sortie et utilise la méthode `json.dump()` pour écrire les données dans ce fichier.

* `load_as_json(self, filename:str):` 
Cette méthode permet de charger les données d'un fichier JSON dans un modèle. Elle prend en entrée le nom du fichier à charger et utilise la méthode `json.load()` pour récupérer les données du fichier.

* `classify(self, vector: dict, k: int, sim_func=None) -> List[Tuple[str, float]]:` 
Cette méthode permet de classer un vecteur avec les vecteurs des classes existantes. Elle prend en entrée un vecteur sous forme de hashage, le nombre de voisins les plus proches à considérer et une fonction de similarité (définie dans la classe `Similarity`), (par défaut, la fonction de calcul de cosinus). Elle calcule d'abord les similarités entre le nouveau vecteur et les vecteurs de chaque classe, puis trie les résultats en fonction de la similarité décroissante. Elle renvoie ensuite une liste triée de paires [label:str,sim:float] pour les classes candidates. La similarité est la moyenne des similarités obtenues sur les vecteurs retenus pour la classe correspondante dans les k plus proches voisins.

**Création d'un modèle KNN**
Instanciation de la classe en fournissant une description :
```python
from knnclass import KNNClass
test_knn = KNNClass(description="données de test")
```



## Classe `Similarity`
La classe Similarity regroupe plusieurs fonctions de calcul de similarité entre deux vecteurs représentés sous forme de dictionnaires.
Ces fonctions peuvent être appelées dans la classe KNNClass pour la classification des vecteurs.

* `scalaire :`
Cette fonction calcule le produit scalaire de deux vecteurs représentés sous forme de dictionnaires. Elle parcourt les clés des deux dictionnaires et pour chaque clé qui est présente dans les deux dictionnaires, elle multiplie les valeurs correspondantes et ajoute le résultat à une liste. La somme des éléments de cette liste est retournée comme produit scalaire.

* `norme :`
Cette fonction calcule la norme d'un vecteur représenté sous forme de dictionnaire.  Elle parcourt les clés du dictionnaire et calcule la somme des carrés des valeurs correspondantes. Ensuite, elle calcule la racine carrée de cette somme et la retourne comme norme.

* `sim_cosinus :`
Cette fonction calcule la similarité cosinus entre deux vecteurs représentés sous forme de dictionnaires.  Elle utilise les fonctions scalaire et norme pour calculer le produit scalaire et les normes des deux vecteurs. La similarité cosinus est ensuite calculée en divisant le produit scalaire par le produit des normes.

* `sim_euclidienne :`
Cette fonction calcule la similarité euclidienne entre deux vecteurs donnés sous forme de dictionnaires. Elle parcourt les clés des deux dictionnaires et calcule la somme des carrés des écarts entre les valeurs correspondantes. Cette somme est normalisée et retournée comme similarité euclidienne.

* `sim_pearson :`
Cette fonction calcule la similarité de Pearson entre deux vecteurs donnés sous forme de dictionnaires. Elle calcule la moyenne de chaque vecteur, la somme des carrés des écarts à la moyenne pour les deux vecteurs, les écarts-types pour les deux vecteurs et la covariance pour les deux vecteurs. Enfin, elle calcule la similarité de Pearson en divisant la covariance par le produit des écarts-types.













## Classe `Gestion`
La classe Gestion contient des fonctions pour ajouter et supprimer des classes et des vecteurs d'un objet KNNClass. L'utilisation de ces fonctions sont facultatives, il est tout à fait possible d'executer les méthodes de la classe KNNClass sans passer par ces fonctions. 

* `add_class_input(knn_object):`
Cette fonction demande à l'utilisateur de saisir les informations nécessaires pour ajouter une nouvelle classe à l'objet KNNClass passé en paramètre (donc le label de la nouvelle classe ainsi que les vecteurs à ajouter). Les vecteurs doivent être saisis sous forme de dictionnaires. La fonction appelle ensuite la méthode `add_class()` de l'objet KNNClass avec les arguments saisis par l'utilisateur. Si l'ajout de la classe se passe bien, la fonction affiche un message de confirmation et renvoie True. Si une erreur se produit, la fonction affiche un message d'erreur correspondant.

* `del_class_input(knn_object):`
Cette fonction demande à l'utilisateur de saisir le label d'une classe à supprimer de l'objet KNNClass passé en paramètre. La fonction appelle ensuite la méthode `del_class()` de l'objet KNNClass avec le label saisi par l'utilisateur. Si la suppression de la classe se passe bien, la fonction affiche un message de confirmation. Si une erreur se produit, la fonction affiche un message d'erreur correspondant.

**Améliorations possibles pour la classe `Gestion` :**

- Il est possible de créer une fonction `add_vectors_input(knn_object)` qui demande à l'utilisateur de saisir les vecteurs à ajouter, ainsi que le label de la classe à laquelle ajouter ces vecteurs. Cette fonction pourrait faire appel à la méthode `add_vector()` de la classe KNNClass pour ajouter les vecteurs à la classe correspondante. Cette fonction enlevera la nécessité de repasser par `add_class_input()` ou d'appeler directement la méthode `add_vector()` pour ajouter des vecteurs supplémentaires à la classe.
- Une autre proposition serait d'ajouter une fonction `classify_input(knn_object)` à la classe Gestion qui appelle la méthode `classify` en demandant les entrées de la méthode. Cela faciliterait l'utilisation de la fonction `classify`.


